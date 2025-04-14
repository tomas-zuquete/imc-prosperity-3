import pandas as pd
import numpy as np
import json
# Removed matplotlib
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple, Any


class StrategyConfig:
    def __init__(self):
        # We'll dynamically set theta from the maximum drift
        self.theta = None

        # Example drift-based sizing (currently not strictly used, but a placeholder for expansions)
        self.drift_size_map = [
            (0.05, 0.2, (5, 10)),
            (0.2, 0.5, (10, 15)),
            (0.5, float("inf"), (15, 20))
        ]

        # Passive quoting thresholds
        self.passive_size_spread_2 = (10, 15)
        self.passive_size_spread_3 = (5, 10)

        # max position ±50 from limit orders.
        # Market orders do not "count" in the environment's limit
        self.max_position = 50

        # Define large trade: trade with volume 35
        self.large_trade_volume = 35

        # Timings
        self.signal_decay_ms = 200  # how often we can re-signal
        self.cooldown_ms = 100  # Strategy B cooldown
        # Estimated from probability vectors
        # limit order cancellation not allowed

        # Execution sizes
        self.execution_unit_size = 1

        # Passive fill probability logic
        self.min_fill_probability = 0.2

        # OBI threshold for Strategy A
        self.obi_threshold = 0.01

        # For trailing stop
        self.trailing_stop_distance = 2.0  # e.g. 2 ticks, assumed

        # For breakout
        self.breakout_threshold_up = 0.5
        self.breakout_threshold_down = 0.5

        self.daily_reentry_threshold = 0.3  # Adjust this to force reentry on a new day


# Data Structure
class LimitOrder:
    """Represent a resting limit order in the order book, uncancelable."""

    def __init__(self, day, ts, side, size, limit_price):
        self.day = day
        self.ts = ts
        self.side = side  # +1 = buy, -1 = sell
        self.size = size
        self.limit_price = limit_price
        self.is_filled = False
        self.fill_ts = None
        self.fill_price = None


class ExecutionRecord:
    """Track each fill event for PnL analysis."""

    def __init__(self, day, timestamp, ex_type, side, size, price):
        self.day = day
        self.timestamp = timestamp
        self.type = ex_type  # "market_order", "limit_fill", "impact_order", etc.
        self.side = side
        self.size = size
        self.price = price


###############################################################################
# TRADING STRATEGY
###############################################################################
# Renamed TradeKELP to Trader as required by IMC platform
class Trader:
    def __init__(self):
        """Initialize with default values until we can build state from data"""
        self.config = StrategyConfig()
        # Default theta if not dynamically set
        if self.config.theta is None:
            self.config.theta = 0.2
        
        # State variables needed between calls
        self.prices = None
        self.trades = None
        self.execution_log = []
        self.limit_orders = []
        self.net_position = 0
        self.position_entry_ts = None
        self.last_execution_ts = -np.inf
        self.last_limit_ts = -np.inf
        self.best_price_in_position = None
        
        # Day tracking
        self.current_day = None
        self.prev_day_close_mid = None
        self.prev_mid = None
        
        # Data for computing metrics
        self.obi_smooth = 0
        self.tfi = 0
        self.expected_drift = 0
        self.rolling_max_5 = None
        self.rolling_min_5 = None
        
        # Track price history
        self.price_history = []
        
        # For state persistence between calls
        self.trader_data_initialized = False

    def preprocess_data(self, state: TradingState):
        """Extract data from TradingState for strategy use"""
        # Get timestamp as day proxy
        timestamp = state.timestamp
        this_day = timestamp // 86400  # Simple day estimation from timestamp
        
        # Keep product reference - default to "KELP" if no products are found
        product = "KELP"
        if state.order_depths:
            product = list(state.order_depths.keys())[0]
        
        # Extract order book data
        bid_price_1 = None
        ask_price_1 = None
        mid_price = None
        spread = float('inf')
        
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            
            # Get best bid/ask and mid price
            if order_depth.buy_orders:
                bid_price_1 = max(order_depth.buy_orders.keys())
            
            if order_depth.sell_orders:
                ask_price_1 = min(order_depth.sell_orders.keys())
            
            # Calculate mid price only if both bid and ask are available
            if bid_price_1 is not None and ask_price_1 is not None:
                mid_price = (bid_price_1 + ask_price_1) / 2
                spread = ask_price_1 - bid_price_1
        
        # If we don't have a mid price but have a previous mid, use that
        if mid_price is None and self.prev_mid is not None:
            mid_price = self.prev_mid
        # If we still don't have a mid price, default to something reasonable
        if mid_price is None:
            mid_price = 2030  # Reasonable default for KELP
        
        # Calculate OBI (Order Book Imbalance)
        obi = 0
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            bid_vols = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            ask_vols = sum(abs(vol) for vol in order_depth.sell_orders.values()) if order_depth.sell_orders else 0
            obi = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6) if (bid_vols + ask_vols) > 0 else 0
        
        # Add the mid price to our history if it's valid
        if mid_price is not None:
            self.price_history.append(mid_price)
            if len(self.price_history) > 20:
                self.price_history = self.price_history[-20:]
        
        # Calculate basic metrics
        if len(self.price_history) >= 5:
            # Update rolling max/min for breakout detection
            self.rolling_max_5 = max(self.price_history[-5:])
            self.rolling_min_5 = min(self.price_history[-5:])
            
            # Calculate simple drift estimate (price change direction)
            if len(self.price_history) >= 6:
                past_5_avg = sum(self.price_history[-6:-1]) / 5
                self.expected_drift = mid_price - past_5_avg
            else:
                # Not enough history for proper drift calculation
                self.expected_drift = 0
        
        # Calculate TFI (Trade Flow Imbalance) from market trades
        if product in state.market_trades and state.market_trades[product]:
            latest_trades = state.market_trades[product]
            # Simple approximation - use latest trade direction and size
            latest_trade = latest_trades[-1]
            # Make sure we have a previous mid to compare with
            if self.prev_mid is not None:
                self.tfi = latest_trade.quantity if latest_trade.price > self.prev_mid else -latest_trade.quantity
            else:
                # Default to signed quantity based on trade price vs mid_price
                self.tfi = latest_trade.quantity if latest_trade.price > mid_price else -latest_trade.quantity
        
        # Smooth OBI (simple implementation)
        self.obi_smooth = 0.8 * self.obi_smooth + 0.2 * obi if self.trader_data_initialized else obi
        
        # Update day tracking
        if self.current_day is None:
            self.current_day = this_day
        elif this_day != self.current_day:
            # New day, reset some state
            self.prev_day_close_mid = self.prev_mid
            self.current_day = this_day
            self.last_execution_ts = -np.inf
            self.last_limit_ts = -np.inf
        
        # Save current mid for next iteration
        self.prev_mid = mid_price
        
        # Mark as initialized
        self.trader_data_initialized = True
        
        # Return processed data
        return {
            'product': product,
            'timestamp': timestamp,
            'day': this_day,
            'bid_price_1': bid_price_1,
            'ask_price_1': ask_price_1,
            'mid_price': mid_price,
            'spread': spread,
            'obi': obi,
            'obi_smooth': self.obi_smooth,
            'tfi': self.tfi,
            'expected_drift': self.expected_drift
        }

    def generate_signals(self, row, position, product, state):
        """Analyze market data and generate trading signals"""
        orders = []
        
        # Extract key data
        ts = row['timestamp']
        this_day = row['day']
        mid = row['mid_price']
        bid_price_1 = row['bid_price_1']
        ask_price_1 = row['ask_price_1']
        spread = row['spread']
        obi_smooth = row['obi_smooth']
        tfi = row['tfi']
        drift = row['expected_drift']
        
        # Safety check - if we don't have valid prices, return empty orders
        if bid_price_1 is None or ask_price_1 is None:
            return orders
        
        # ---- Daily Reentry Mechanism ----
        if position == 0 and ts > 3000 and self.prev_day_close_mid is not None:
            if abs(mid - self.prev_day_close_mid) > self.config.daily_reentry_threshold:
                side_reentry = 1 if mid > self.prev_day_close_mid else -1
                fill_price = ask_price_1 if side_reentry > 0 else bid_price_1
                print(f"[DAILY REENTRY] Day {this_day}: mid {mid:.2f} differs from prev close {self.prev_day_close_mid:.2f}")
                orders.append(Order(product, fill_price, self.config.execution_unit_size if side_reentry > 0 else -self.config.execution_unit_size))
                self.last_execution_ts = ts
                return orders
        
        # Process limit orders if we have a valid mid price
        if mid is not None:
            filled_order = self.fill_limit_orders(this_day, ts, mid, product)
            if filled_order:
                orders.append(filled_order)
                self.last_execution_ts = ts
                return orders
        
        # Check trailing stop (only if we have a position and valid prices)
        if position != 0 and mid is not None:
            if self.best_price_in_position is None:
                self.best_price_in_position = mid
            
            side_held = np.sign(position)
            if side_held > 0:  # Long position
                if mid > self.best_price_in_position:
                    self.best_price_in_position = mid
                if (self.best_price_in_position - mid) >= self.config.trailing_stop_distance:
                    fill_price = bid_price_1
                    orders.append(Order(product, fill_price, -abs(position)))
                    self.best_price_in_position = None
                    self.last_execution_ts = ts
                    return orders
            elif side_held < 0:  # Short position
                if mid < self.best_price_in_position:
                    self.best_price_in_position = mid
                if (mid - self.best_price_in_position) >= self.config.trailing_stop_distance:
                    fill_price = ask_price_1
                    orders.append(Order(product, fill_price, abs(position)))
                    self.best_price_in_position = None
                    self.last_execution_ts = ts
                    return orders
        
        # Check signal flip (exit positions if signal changes direction)
        if position != 0:
            side_signal = int(np.sign(drift)) if abs(drift) > self.config.theta else 0
            side_held = int(np.sign(position))
            if side_signal != 0 and side_signal != side_held:
                flatten_side = -side_held
                size_to_exit = abs(position)
                fill_price = ask_price_1 if flatten_side > 0 else bid_price_1
                orders.append(Order(product, fill_price, size_to_exit if flatten_side > 0 else -size_to_exit))
                self.last_execution_ts = ts
                return orders
        
        # Cooldown check
        if ts - self.last_execution_ts < self.config.signal_decay_ms:
            return orders
        
        # STRATEGY A: Aggressive Market Orders
        if (spread <= 2 and 
            abs(drift) > self.config.theta and 
            abs(obi_smooth) > self.config.obi_threshold and 
            np.sign(obi_smooth) == np.sign(tfi)):
            
            side = int(np.sign(drift))
            if abs(position + side * self.config.execution_unit_size) <= self.config.max_position:
                fill_price = ask_price_1 if side > 0 else bid_price_1
                orders.append(Order(product, fill_price, self.config.execution_unit_size if side > 0 else -self.config.execution_unit_size))
                self.last_execution_ts = ts
                return orders
        
        # STRATEGY B: Passive Limit Orders
        elif (0.005 <= abs(obi_smooth) <= 0.01 and
              (np.sign(obi_smooth) == np.sign(tfi) or abs(tfi) < 1e-6) and
              spread in [2, 3] and
              (ts - self.last_limit_ts >= self.config.cooldown_ms)):
            
            fill_prob = self.estimate_fill_probability(spread, obi_smooth)
            if fill_prob >= self.config.min_fill_probability:
                side = 1 if obi_smooth > 0 else -1
                if abs(position + side * self.config.execution_unit_size) <= self.config.max_position:
                    if spread == 3:
                        limit_price = ask_price_1 if side > 0 else bid_price_1
                    else:
                        limit_price = (ask_price_1 - 0.5) if side > 0 else (bid_price_1 + 0.5)
                    
                    limit_order = self.place_limit_order(this_day, ts, side, self.config.execution_unit_size, limit_price, product)
                    self.last_execution_ts = ts
                    self.last_limit_ts = ts
                    orders.append(limit_order)
                    return orders
        
        # STRATEGY C: Impact Trigger
        recent_trades = state.market_trades.get(product, [])
        if recent_trades:
            last_vol = recent_trades[-1].quantity
            if (last_vol >= self.config.large_trade_volume and
                np.sign(obi_smooth) == np.sign(tfi) and
                abs(drift) > self.config.theta):
                
                side = int(np.sign(tfi))
                if abs(position + side * self.config.execution_unit_size) <= self.config.max_position:
                    fill_price = ask_price_1 if side > 0 else bid_price_1
                    orders.append(Order(product, fill_price, self.config.execution_unit_size if side > 0 else -self.config.execution_unit_size))
                    self.last_execution_ts = ts
                    return orders
        
        # STRATEGY D: Breakout Approach
        if self.rolling_max_5 is not None and self.rolling_min_5 is not None:
            # Up breakout condition
            if (mid > self.rolling_max_5 + self.config.breakout_threshold_up and
                drift > self.config.theta and
                obi_smooth > 0 and
                tfi > 0):
                
                side = 1  # Buy
                if abs(position + side * self.config.execution_unit_size) <= self.config.max_position:
                    fill_price = ask_price_1
                    orders.append(Order(product, fill_price, self.config.execution_unit_size))
                    self.last_execution_ts = ts
                    return orders
                    
            # Down breakout condition
            elif (mid < self.rolling_min_5 - self.config.breakout_threshold_down and
                  drift < -self.config.theta and
                  obi_smooth < 0 and
                  tfi < 0):
                  
                side = -1  # Sell
                if abs(position + side * self.config.execution_unit_size) <= self.config.max_position:
                    fill_price = bid_price_1
                    orders.append(Order(product, fill_price, -self.config.execution_unit_size))
                    self.last_execution_ts = ts
                    return orders
        
        return orders

    def fill_limit_orders(self, day, now_ts, now_mid, product):
        """Check and fill limit orders that can execute"""
        for lo in self.limit_orders:
            if lo.is_filled:
                continue
            if lo.side > 0 and now_mid <= lo.limit_price:
                lo.is_filled = True
                lo.fill_ts = now_ts
                lo.fill_price = lo.limit_price
                return Order(product, lo.fill_price, lo.size)
            elif lo.side < 0 and now_mid >= lo.limit_price:
                lo.is_filled = True
                lo.fill_ts = now_ts
                lo.fill_price = lo.limit_price
                return Order(product, lo.fill_price, -lo.size)
        return None

    def place_limit_order(self, day, ts, side, size, limit_price, product):
        """Create a new limit order"""
        lo = LimitOrder(day, ts, side, size, limit_price)
        self.limit_orders.append(lo)
        print(f"[PASSIVE] limit order placed: day={day}, side={side}, price={limit_price:.2f}")
        return Order(product, limit_price, size if side > 0 else -size)

    def estimate_fill_probability(self, spread, obi_smooth):
        """Estimate probability of getting filled on a passive order"""
        if spread == 2:
            return 0.6 if abs(obi_smooth) > 0.005 else 0.3
        elif spread == 3:
            return 0.3 if abs(obi_smooth) > 0.005 else 0.1
        return 0

    def serialize_state(self):
        """Convert strategy state to a string for persistence between calls"""
        state = {
            'net_position': self.net_position,
            'position_entry_ts': self.position_entry_ts,
            'last_execution_ts': float(self.last_execution_ts),
            'last_limit_ts': float(self.last_limit_ts),
            'best_price_in_position': self.best_price_in_position,
            'current_day': self.current_day,
            'prev_day_close_mid': self.prev_day_close_mid,
            'prev_mid': self.prev_mid,
            'obi_smooth': float(self.obi_smooth),
            'tfi': float(self.tfi),
            'expected_drift': float(self.expected_drift),
            'rolling_max_5': self.rolling_max_5,
            'rolling_min_5': self.rolling_min_5,
            'price_history': self.price_history,
            'trader_data_initialized': self.trader_data_initialized,
            # Serialize limit orders
            'limit_orders': [(lo.day, lo.ts, lo.side, lo.size, lo.limit_price, lo.is_filled, 
                             lo.fill_ts, lo.fill_price) for lo in self.limit_orders]
        }
        return json.dumps(state)

    def restore_state(self, trader_data_str):
        """Restore strategy state from a string"""
        if not trader_data_str:
            return
            
        try:
            state = json.loads(trader_data_str)
            
            # Restore basic state variables
            self.net_position = state.get('net_position', 0)
            self.position_entry_ts = state.get('position_entry_ts')
            self.last_execution_ts = state.get('last_execution_ts', -np.inf)
            self.last_limit_ts = state.get('last_limit_ts', -np.inf)
            self.best_price_in_position = state.get('best_price_in_position')
            self.current_day = state.get('current_day')
            self.prev_day_close_mid = state.get('prev_day_close_mid')
            self.prev_mid = state.get('prev_mid')
            self.obi_smooth = state.get('obi_smooth', 0)
            self.tfi = state.get('tfi', 0)
            self.expected_drift = state.get('expected_drift', 0)
            self.rolling_max_5 = state.get('rolling_max_5')
            self.rolling_min_5 = state.get('rolling_min_5')
            self.price_history = state.get('price_history', [])
            self.trader_data_initialized = state.get('trader_data_initialized', False)
            
            # Restore limit orders
            limit_orders_data = state.get('limit_orders', [])
            self.limit_orders = []
            for lo_data in limit_orders_data:
                lo = LimitOrder(lo_data[0], lo_data[1], lo_data[2], lo_data[3], lo_data[4])
                lo.is_filled = lo_data[5]
                lo.fill_ts = lo_data[6]
                lo.fill_price = lo_data[7]
                self.limit_orders.append(lo)
                
        except Exception as e:
            print(f"Error restoring state: {e}")

    # Added run method required by IMC platform
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main method called by the exchange to get orders.
        It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent.
        """
        try:
            # Restore state from previous iteration
            self.restore_state(state.traderData)
            
            # Process market data
            market_data = self.preprocess_data(state)
            
            # Get the product we're trading
            product = market_data['product']
            
            # Get current position from state
            position = state.position.get(product, 0)
            self.net_position = position  # Update internal tracking
            
            # Generate trading signals
            orders = self.generate_signals(market_data, position, product, state)
            
            # Prepare result in expected format
            result = {product: orders}
            
            # No conversions for KELP
            conversions = 0
            
            # Serialize state for next iteration
            trader_data = self.serialize_state()
            
            return result, conversions, trader_data
            
        except Exception as e:
            print(f"Error in strategy execution: {e}")
            # Return empty result in case of error
            return {}, 0, self.serialize_state() if hasattr(self, 'serialize_state') else "{}"
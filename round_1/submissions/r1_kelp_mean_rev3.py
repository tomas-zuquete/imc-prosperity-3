import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState, OrderDepth, UserId

# Constants to avoid typos
SUBMISSION = "SUBMISSION"
KELP = "KELP"

PRODUCTS = [
    KELP
]

DEFAULT_PRICES = {
    KELP: 2030  # Starting middle price based on historical data
}

class Trader:
    def __init__(self) -> None:
        print("Initializing Trader...")

        # Position limits
        self.position_limit = {
            KELP: 50
        }

        # Keep track of round
        self.round = 0

        # Values to compute pnl
        self.cash = 0

        # Keep track of all past prices for each product
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []
            
        # Keep track of past positions for each product
        self.past_positions = dict()
        for product in PRODUCTS:
            self.past_positions[product] = []

        # Keep track of past orders to avoid over-trading
        self.last_order_round = dict()
        for product in PRODUCTS:
            self.last_order_round[product] = 0

        # Keep track of EMAs for mean reversion strategy
        self.ema_short = dict()
        self.ema_long = dict()
        for product in PRODUCTS:
            self.ema_short[product] = None
            self.ema_long[product] = None

        # Parameters for EMAs - modified for better performance based on logs
        self.ema_short_param = 2 / (3 + 1)  # 3-period EMA (faster)
        self.ema_long_param = 2 / (10 + 1)  # 10-period EMA

        # Parameters for Z-score calculation
        self.z_score_window = 10  # Reduced window size for faster adaptation
        self.rolling_mean = {KELP: None}
        self.rolling_std = {KELP: None}
        
        # Trading hours with superior performance (based on analysis)
        self.peak_hours = [3, 10, 13]
        
        # Mean reversion threshold parameters
        self.z_score_entry = 2.5     # Increased for stronger signals only
        self.z_score_exit = 0.8      # Increased to hold positions longer
        self.ema_threshold = 0.0001  # Small threshold for signal detection
        
        # Risk management parameter - use smaller positions
        self.risk_factor = 1.0       # Very conservative position sizing
        
        # Hour tracking
        self.current_hour = {KELP: 0}
        
        # Volatility tracking
        self.volatility = {KELP: 0.001}
        
        # Price direction tracking
        self.price_direction = {KELP: None}
        
        # Track average spread
        self.avg_spread = {KELP: None}
        self.spread_history = {KELP: []}
        
        # Track entry prices for each position
        self.position_entry_price = {KELP: None}
        
        # Track momentum
        self.momentum = {KELP: 0}
        
        # Risk management parameters - tighter profit taking/stop loss
        self.profit_target_percent = 0.0005  # 0.05% profit target
        self.stop_loss_percent = 0.001       # 0.1% stop loss
        
        # Don't trade until we have enough data
        self.min_data_points = 15
        
        # Simple market making parameters
        self.mm_spread_multiplier = 1.0  # Place orders at current spread
        
        # Wait several rounds between trades
        self.min_rounds_between_trades = 5
        
        # Mandatory cooling-off period after a losing trade
        self.cooling_off = False
        self.cooling_off_rounds = 0
        self.max_cooling_off_rounds = 10
        
        # Keep track of positive vs negative return rounds
        self.positive_rounds = 0
        self.negative_rounds = 0
        
        # Keep record of successful trade times
        self.successful_hours = {}
        
        # Last trade result
        self.last_trade_pnl = 0
        
        # Strategy mode
        self.mode = "OBSERVATION"  # Start in observation mode
        
        # Limit trade frequency 
        self.trade_frequency_limiter = 0
        self.rounds_since_last_trade = 0

    # Utils
    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    def get_bid_ask_mid_price(self, product, state: TradingState):
        """
        Returns best bid, best ask, and mid price for a product
        """
        if product not in state.order_depths or not state.order_depths[product].buy_orders or not state.order_depths[product].sell_orders:
            if self.ema_short[product] is not None:
                return None, None, self.ema_short[product]
            return None, None, DEFAULT_PRICES[product]

        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders

        if not market_bids or not market_asks:
            if self.ema_short[product] is not None:
                return None, None, self.ema_short[product]
            return None, None, DEFAULT_PRICES[product]

        best_bid = max(market_bids.keys())
        best_ask = min(market_asks.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Update spread history
        current_spread = best_ask - best_bid
        self.spread_history[product].append(current_spread)
        if len(self.spread_history[product]) > 30:
            self.spread_history[product] = self.spread_history[product][-30:]
        
        # Calculate average spread
        self.avg_spread[product] = sum(self.spread_history[product]) / len(self.spread_history[product])
        
        return best_bid, best_ask, mid_price

    def update_emas(self, product, mid_price):
        """
        Update the exponential moving averages for the product
        """
        if mid_price is None:
            return
            
        # Update short EMA
        if self.ema_short[product] is None:
            self.ema_short[product] = mid_price
        else:
            self.ema_short[product] = mid_price * self.ema_short_param + (1 - self.ema_short_param) * self.ema_short[product]
            
        # Update long EMA
        if self.ema_long[product] is None:
            self.ema_long[product] = mid_price
        else:
            self.ema_long[product] = mid_price * self.ema_long_param + (1 - self.ema_long_param) * self.ema_long[product]

    def update_price_history(self, product, mid_price):
        """
        Update price history and calculate volatility
        """
        if mid_price is None:
            return
            
        self.past_prices[product].append(mid_price)
        
        # Keep only the last 50 prices
        if len(self.past_prices[product]) > 50:
            self.past_prices[product] = self.past_prices[product][-50:]
            
        # Update price direction and momentum
        if len(self.past_prices[product]) >= 3:
            prev_price = self.past_prices[product][-2]
            if mid_price > prev_price:
                self.price_direction[product] = 1
                # Consecutive up move
                if self.momentum[product] >= 0:
                    self.momentum[product] += 1
                else:
                    self.momentum[product] = 1
            elif mid_price < prev_price:
                self.price_direction[product] = -1
                # Consecutive down move
                if self.momentum[product] <= 0:
                    self.momentum[product] -= 1
                else:
                    self.momentum[product] = -1
            else:
                # No change, reduce momentum
                self.momentum[product] = int(self.momentum[product] * 0.5)
            
        # Calculate recent volatility (standard deviation of returns)
        if len(self.past_prices[product]) >= 5:
            prices = self.past_prices[product][-5:]
            returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            self.volatility[product] = max(math.sqrt(variance), 0.0001)  # Minimum volatility floor

    def update_z_score(self, product, mid_price):
        """
        Update rolling mean and standard deviation for z-score calculation
        """
        if mid_price is None:
            return
            
        # Use as many prices as available, up to z_score_window
        window = min(len(self.past_prices[product]), self.z_score_window)
        if window < 5:  # Need at least 5 prices to calculate meaningful statistics
            return
            
        prices = self.past_prices[product][-window:]
        self.rolling_mean[product] = sum(prices) / len(prices)
        
        variance = sum((p - self.rolling_mean[product]) ** 2 for p in prices) / len(prices)
        self.rolling_std[product] = max(math.sqrt(variance), 0.1)  # Minimum std dev floor to avoid division by zero
    
    def calculate_z_score(self, product, mid_price):
        """
        Calculate z-score for current price
        """
        if self.rolling_mean[product] is None or self.rolling_std[product] is None or mid_price is None:
            return 0
            
        return (mid_price - self.rolling_mean[product]) / self.rolling_std[product]
        
    def update_hour(self, product, timestamp):
        """
        Calculate the hour of day (0-23) from timestamp
        """
        hour = (timestamp % 86400) // 3600  # 86400 seconds in a day, 3600 seconds in an hour
        self.current_hour[product] = hour
        
    def should_take_profit_or_stop_loss(self, product, current_price, current_position):
        """
        Determine if we should take profit or cut losses
        """
        if self.position_entry_price[product] is None or current_position == 0:
            return False, "HOLD"
            
        entry_price = self.position_entry_price[product]
        
        # Calculate % gain/loss
        if current_position > 0:  # Long position
            pct_change = (current_price - entry_price) / entry_price
            if pct_change >= self.profit_target_percent:
                return True, "TAKE_PROFIT"
            elif pct_change <= -self.stop_loss_percent:
                return True, "STOP_LOSS"
        else:  # Short position
            pct_change = (entry_price - current_price) / entry_price
            if pct_change >= self.profit_target_percent:
                return True, "TAKE_PROFIT"
            elif pct_change <= -self.stop_loss_percent:
                return True, "STOP_LOSS"
                
        return False, "HOLD"
        
    def identify_signal(self, product, mid_price, current_position):
        """
        Identify trading signal based on market analysis and strategy
        """
        # Cooling off period after losing trade
        if self.cooling_off:
            self.cooling_off_rounds += 1
            if self.cooling_off_rounds >= self.max_cooling_off_rounds:
                self.cooling_off = False
                self.cooling_off_rounds = 0
            else:
                print(f"In cooling-off period: {self.cooling_off_rounds}/{self.max_cooling_off_rounds}")
                return "WAIT"
            
        # Wait for enough data
        if len(self.past_prices[product]) < self.min_data_points:
            print(f"Waiting for more data points: {len(self.past_prices[product])}/{self.min_data_points}")
            return "WAIT"
            
        # Wait for enough rounds between trades
        if self.round - self.last_order_round[product] < self.min_rounds_between_trades:
            print(f"Waiting for trading cooldown: {self.round - self.last_order_round[product]}/{self.min_rounds_between_trades}")
            return "WAIT"
            
        # Check if we should take profit or cut losses
        should_exit, exit_reason = self.should_take_profit_or_stop_loss(product, mid_price, current_position)
        if should_exit:
            print(f"Exit signal: {exit_reason}")
            return "EXIT"
            
        # Calculate EMA and Z-score indicators
        if self.ema_short[product] is None or self.ema_long[product] is None:
            return "NEUTRAL"
            
        diff_pct = (self.ema_short[product] - self.ema_long[product]) / self.ema_long[product]
        z_score = self.calculate_z_score(product, mid_price)
        
        # For debugging
        print(f"Z-score: {z_score:.4f}, EMA diff: {diff_pct*100:.4f}%, Momentum: {self.momentum[product]}")
        
        # In observation mode, just watch and record statistics
        if self.mode == "OBSERVATION" and self.round < 20:
            return "NEUTRAL"
            
        # Profitable hour bias - trade more during historically profitable hours
        profitable_hour_bias = 1.0
        if self.current_hour[product] in self.successful_hours:
            success_rate = self.successful_hours[product]/self.rounds_since_last_trade
            if success_rate > 0.6:  # More than 60% win rate
                profitable_hour_bias = 1.5
                
        # Very simple and conservative strategy - only trade on extreme signals
        # and only in the right direction for each signal type
        
        if z_score < -self.z_score_entry * profitable_hour_bias:
            # Deep discount - buy only if momentum is positive (price starting to rise)
            if self.momentum[product] > 0:
                return "BUY" 
        elif z_score > self.z_score_entry * profitable_hour_bias:
            # Very overpriced - sell only if momentum is negative (price starting to fall)
            if self.momentum[product] < 0:
                return "SELL"
        elif abs(z_score) < self.z_score_exit:
            # Price has returned to mean - exit positions
            if current_position != 0:
                return "EXIT"
        
        # Market making strategy
        if self.avg_spread[product] is not None and self.avg_spread[product] > 1.0 and current_position == 0:
            # Only make market when there's no existing position
            return "MAKE_MARKET"
                
        return "NEUTRAL"

    def calculate_position_size(self, product, signal, current_position, best_bid, best_ask):
        """
        Calculate optimal position size based on signal, volatility, and limits
        """
        position_limit = self.position_limit[product]
        
        # Special cases
        if signal in ("EXIT", "WAIT", "NEUTRAL"):
            if signal == "EXIT":
                return -current_position
            return 0
            
        # Very small position sizes to minimize risk
        if signal == "BUY":
            target_position = 3  # Limited buying
        elif signal == "SELL":
            target_position = -3  # Limited selling
        elif signal == "MAKE_MARKET":
            # For market making, use even smaller positions
            return 0  # No position change - handled separately
        else:
            target_position = 0
            
        # Limit the size to position limits
        target_position = max(-position_limit, min(position_limit, target_position))
        
        # Calculate position change needed
        position_delta = target_position - current_position
        
        return position_delta
        
    def create_market_making_orders(self, product, best_bid, best_ask, mid_price):
        """
        Create market making orders just inside the spread
        """
        orders = []
        
        if best_bid is None or best_ask is None:
            return orders
            
        spread = best_ask - best_bid
        
        # Only make markets if spread is wider than 1
        if spread <= 1:
            return orders
            
        # Place buy order just above best bid 
        buy_price = best_bid + 1
        # Place sell order just below best ask
        sell_price = best_ask - 1
        
        # Use very small size for market making
        size = 1
        
        # Add orders if they're inside the spread
        if buy_price < sell_price:
            orders.append(Order(product, buy_price, size))
            orders.append(Order(product, sell_price, -size))
            
        return orders
        
    def kelp_strategy(self, state: TradingState):
        """
        Implement combined mean reversion and market making strategy for KELP
        """
        product = KELP
        
        # Get current position
        current_position = self.get_position(product, state)
        
        # Update position history
        self.past_positions[product].append(current_position)
        if len(self.past_positions[product]) > 10:
            self.past_positions[product] = self.past_positions[product][-10:]
        
        # Get market data
        best_bid, best_ask, mid_price = self.get_bid_ask_mid_price(product, state)
        
        # Skip if no valid price
        if mid_price is None:
            return []
            
        # Update market data
        self.update_price_history(product, mid_price)
        self.update_emas(product, mid_price)
        self.update_z_score(product, mid_price)
        self.update_hour(product, state.timestamp)
        
        # Analyze trades to see if we entered a new position
        position_changed = False
        if len(self.past_positions[product]) >= 2:
            if self.past_positions[product][-1] != self.past_positions[product][-2]:
                position_changed = True
                
                # If we're entering a new position, record the entry price
                if self.past_positions[product][-2] == 0 and self.past_positions[product][-1] != 0:
                    self.position_entry_price[product] = mid_price
                
                # If we're exiting a position, record profitability
                if self.past_positions[product][-2] != 0 and self.past_positions[product][-1] == 0:
                    # Reset entry price
                    self.position_entry_price[product] = None
                    
                    # Record trading hour success
                    hour = self.current_hour[product]
                    if hour not in self.successful_hours:
                        self.successful_hours[hour] = 0
                        
                    # Check if this was a profitable trade
                    if self.last_trade_pnl > 0:
                        self.successful_hours[hour] += 1
                        self.positive_rounds += 1
                        # Reset cooling off after a winning trade
                        self.cooling_off = False
                        self.cooling_off_rounds = 0
                    else:
                        self.negative_rounds += 1
                        # Enter cooling off period after a losing trade
                        self.cooling_off = True
                        self.cooling_off_rounds = 0
        
        # Get trading signal
        signal = self.identify_signal(product, mid_price, current_position)
        
        # Create orders list
        orders = []
        
        # Handle market making separately
        if signal == "MAKE_MARKET":
            orders = self.create_market_making_orders(product, best_bid, best_ask, mid_price)
            if orders:
                print(f"Adding market making orders")
                self.last_order_round[product] = self.round
                self.rounds_since_last_trade = 0
            return orders
            
        # Calculate position change for other signals
        position_delta = self.calculate_position_size(product, signal, current_position, best_bid, best_ask)
        
        # Don't trade if no position change needed
        if position_delta == 0:
            self.rounds_since_last_trade += 1
            print(f"No position change needed. Signal: {signal}")
            return []
        
        print(f"Taking action: {signal}, Position Delta: {position_delta}, Current Position: {current_position}")
            
        # Execute the position change
        if position_delta > 0:  # Need to buy
            # Only place order at the current best ask
            if product in state.order_depths and state.order_depths[product].sell_orders:
                price = min(state.order_depths[product].sell_orders.keys())
                volume = state.order_depths[product].sell_orders[price]
                
                # Limit quantity to available volume
                buy_quantity = min(position_delta, -volume)  # volume is negative for sell orders
                if buy_quantity > 0:
                    orders.append(Order(product, price, buy_quantity))
                    print(f"Adding BUY order: {product} @ {price} x {buy_quantity}")
                    
                    # Update last order round
                    self.last_order_round[product] = self.round
                    self.rounds_since_last_trade = 0
        
        elif position_delta < 0:  # Need to sell
            # Only place order at the current best bid
            if product in state.order_depths and state.order_depths[product].buy_orders:
                price = max(state.order_depths[product].buy_orders.keys())
                volume = state.order_depths[product].buy_orders[price]
                
                # Limit quantity to available volume
                sell_quantity = min(-position_delta, volume)  # Convert to positive, volume is positive for buy orders
                if sell_quantity > 0:
                    orders.append(Order(product, price, -sell_quantity))  # Negative for sell
                    print(f"Adding SELL order: {product} @ {price} x {sell_quantity}")
                    
                    # Update last order round
                    self.last_order_round[product] = self.round
                    self.rounds_since_last_trade = 0
        
        # Print strategy information
        z_score = self.calculate_z_score(product, mid_price)
        ema_diff = ((self.ema_short.get(product, 0) - self.ema_long.get(product, 0)) / 
                   self.ema_long.get(product, 1)) * 100 if self.ema_long.get(product, 0) else 0
        
        print(f"{product} - Signal: {signal}, Z-score: {z_score:.2f}, Hour: {self.current_hour.get(product, 0)}, " +
              f"Position: {current_position}, Target: {current_position + position_delta}, " +
              f"EMA Diff: {ema_diff:.4f}%")
              
        return orders
                    
    def update_pnl(self, state: TradingState):
        """
        Updates the pnl.
        """
        prev_pnl = self.cash
        
        def update_cash():
            # Update cash based on trades
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price

        def get_value_on_positions():
            value = 0
            for product in state.position:
                _, _, mid_price = self.get_bid_ask_mid_price(product, state)
                if mid_price is not None:
                    value += state.position[product] * mid_price
            return value

        # Update cash
        update_cash()
        current_pnl = self.cash + get_value_on_positions()
        
        # Calculate PnL change
        self.last_trade_pnl = current_pnl - prev_pnl
        
        return current_pnl

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Any, Any]:
        """
        Main method called by the exchange to get orders.
        It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent.
        """
        self.round += 1
        pnl = self.update_pnl(state)
        
        print(f"Log round {self.round}")
        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)

        print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            _, _, mid_price = self.get_bid_ask_mid_price(product, state)
            print(f"\tProduct {product}, Position {self.get_position(product, state)}, " +
                  f"Midprice {mid_price}, Value {self.get_position(product, state) * mid_price if mid_price else 0}")
        print(f"\tPnL {pnl}")

        # Initialize the method output dict as an empty dict
        result = {}

        try:
            result[KELP] = self.kelp_strategy(state)
        except Exception as e:
            print(f"Error in kelp strategy: {e}")
        
        # No conversions for KELP
        conversions = 0

        return result, conversions, None
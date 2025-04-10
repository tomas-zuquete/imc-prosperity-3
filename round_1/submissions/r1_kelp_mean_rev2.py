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

        # Keep track of EMAs for mean reversion strategy
        self.ema_short = dict()
        self.ema_long = dict()
        for product in PRODUCTS:
            self.ema_short[product] = None
            self.ema_long[product] = None

        # Parameters for EMAs - modified for better performance based on logs
        self.ema_short_param = 2 / (3 + 1)  # 3-period EMA (faster)
        self.ema_long_param = 2 / (15 + 1)  # 15-period EMA (slower)

        # Parameters for Z-score calculation
        self.z_score_window = 15
        self.rolling_mean = {KELP: None}
        self.rolling_std = {KELP: None}
        
        # Trading hours with superior performance (based on analysis)
        self.peak_hours = [3, 10, 13]
        
        # Mean reversion threshold parameters - more conservative settings
        self.z_score_entry = 1.5  # Z-score threshold for entry (increased from 0.3)
        self.z_score_exit = 0.5   # Z-score threshold for exit (increased from 0.1)
        self.ema_threshold = 0.0001  # Minimum EMA difference to identify signal
        
        # Risk management parameter
        self.risk_factor = 0.4  # Decreased from 0.8 for more conservative positions
        
        # Hour tracking
        self.current_hour = {KELP: 0}
        
        # Volatility tracking
        self.volatility = {KELP: 0.001}  # Starting volatility estimate
        
        # Price direction tracking
        self.price_direction = {KELP: 0}  # 0: neutral, 1: up, -1: down
        
        # Profit taking parameters
        self.profit_target_percent = 0.001  # Take profit at 0.1% gain
        self.stop_loss_percent = 0.002      # Cut losses at 0.2% loss
        
        # Trade execution tracking
        self.last_trade_price = {KELP: None}
        self.position_entry_price = {KELP: None}
        
        # Flag to wait for more data
        self.wait_for_data = True
        self.min_data_points = 10
        
        # Trade only when trends are clear
        self.min_trend_strength = 0.00005

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
            
        # Update price direction
        if len(self.past_prices[product]) >= 3:
            prev_price = self.past_prices[product][-2]
            if mid_price > prev_price:
                self.price_direction[product] = 1
            elif mid_price < prev_price:
                self.price_direction[product] = -1
            else:
                self.price_direction[product] = 0
            
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
        Identify trading signal using mean reversion strategy
        """
        # Wait for enough data
        if len(self.past_prices[product]) < self.min_data_points:
            print(f"Waiting for more data points: {len(self.past_prices[product])}/{self.min_data_points}")
            return "WAIT"
            
        # Check if we should take profit or cut losses
        should_exit, exit_reason = self.should_take_profit_or_stop_loss(product, mid_price, current_position)
        if should_exit:
            print(f"Exit signal: {exit_reason}")
            return "EXIT"
        
        if self.ema_short[product] is None or self.ema_long[product] is None:
            return "NEUTRAL"
            
        # Calculate percentage difference between short and long EMAs
        diff_pct = (self.ema_short[product] - self.ema_long[product]) / self.ema_long[product]
        
        # Check if trend is strong enough to trade
        if abs(diff_pct) < self.min_trend_strength:
            return "NEUTRAL"
        
        # CORRECTLY IMPLEMENTED mean reversion (we were previously trading in the wrong direction)
        # For mean reversion: BUY when price is BELOW mean (negative z-score) and SELL when price is ABOVE mean (positive z-score)
        signal = "NEUTRAL"
        z_score = self.calculate_z_score(product, mid_price)
        
        # For debugging
        print(f"Z-score: {z_score:.4f}, EMA diff: {diff_pct*100:.4f}%")
        
        # Trading logic with trend confirmation
        # Only trade when z-score and trend direction agree
        if z_score < -self.z_score_entry:
            # Price is significantly below mean - buy signal for mean reversion
            if self.price_direction[product] == 1:  # Price is moving up (recovery confirmation)
                signal = "STRONG_BUY"
            else:
                signal = "BUY"
        elif z_score > self.z_score_entry:
            # Price is significantly above mean - sell signal for mean reversion
            if self.price_direction[product] == -1:  # Price is moving down (decline confirmation)
                signal = "STRONG_SELL"
            else:
                signal = "SELL"
        elif abs(z_score) < self.z_score_exit:
            # Price is close to mean - exit positions
            signal = "EXIT"
            
        return signal
        
    def calculate_position_size(self, product, signal, current_position, best_bid, best_ask):
        """
        Calculate optimal position size based on signal, volatility, and limits
        """
        position_limit = self.position_limit[product]
        
        # Special case for exit signal
        if signal == "EXIT":
            return -current_position  # Close entire position
            
        # Special case for wait signal
        if signal == "WAIT":
            return 0
        
        # Adjust position size based on volatility (lower for higher volatility)
        volatility_factor = 1 / (1 + self.volatility[product] * 20)  # Increased impact of volatility
        
        # Adjust based on hour of day
        hour_factor = 1.0
        if self.current_hour[product] in self.peak_hours:
            hour_factor = 1.2  # 20% boost during peak hours
        
        # Adjust based on spread
        spread = best_ask - best_bid if best_ask is not None and best_bid is not None else 0
        spread_factor = 1.0
        if spread > 0:
            # Reduce position size for large spreads
            avg_price = (best_ask + best_bid) / 2
            relative_spread = spread / avg_price
            if relative_spread > 0.001:  # Greater than 0.1%
                spread_factor = 0.5  # Reduce position size by 50%
        
        # Base position size with all factors
        base_size = math.floor(position_limit * self.risk_factor * volatility_factor * hour_factor * spread_factor)
        
        # Adjust based on signal strength
        if signal == "STRONG_BUY":
            target_position = base_size
        elif signal == "BUY":
            target_position = math.floor(base_size * 0.5)  # 50% size for regular signals
        elif signal == "STRONG_SELL":
            target_position = -base_size
        elif signal == "SELL":
            target_position = -math.floor(base_size * 0.5)  # 50% size for regular signals
        else:  # NEUTRAL
            target_position = 0
        
        # Limit the size to position limits
        target_position = max(-position_limit, min(position_limit, target_position))
        
        # Calculate position change needed
        position_delta = target_position - current_position
        
        return position_delta
        
    def kelp_strategy(self, state: TradingState):
        """
        Implement mean reversion strategy for KELP
        """
        product = KELP
        
        # Get current position
        current_position = self.get_position(product, state)
        
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
        if current_position != 0 and self.position_entry_price[product] is None:
            self.position_entry_price[product] = mid_price
            
        # Or if we closed out a position
        if current_position == 0:
            self.position_entry_price[product] = None
            
        # Get trading signal
        signal = self.identify_signal(product, mid_price, current_position)
        
        # Calculate position change
        position_delta = self.calculate_position_size(product, signal, current_position, best_bid, best_ask)
        
        # Don't trade if no position change needed
        if position_delta == 0:
            print(f"No position change needed. Signal: {signal}")
            return []
        
        print(f"Taking action: {signal}, Position Delta: {position_delta}, Current Position: {current_position}")
            
        # Create orders list
        orders = []
        
        # Execute the position change
        if position_delta > 0:  # Need to buy
            # Only use market orders to guarantee execution
            if product in state.order_depths and state.order_depths[product].sell_orders:
                available_sells = sorted(state.order_depths[product].sell_orders.items())
                
                remaining_to_buy = position_delta
                for price, volume in available_sells:
                    # Volume is negative for sell orders
                    buy_quantity = min(remaining_to_buy, -volume)
                    if buy_quantity > 0:
                        orders.append(Order(product, price, buy_quantity))
                        print(f"Adding BUY order: {product} @ {price} x {buy_quantity}")
                        remaining_to_buy -= buy_quantity
                        
                        # Update last trade price
                        self.last_trade_price[product] = price
                        if current_position == 0:  # Starting a new position
                            self.position_entry_price[product] = price
                        
                    # Break after first level to avoid overpaying
                    break
                        
        elif position_delta < 0:  # Need to sell
            # Only use market orders to guarantee execution
            if product in state.order_depths and state.order_depths[product].buy_orders:
                available_buys = sorted(state.order_depths[product].buy_orders.items(), reverse=True)
                
                remaining_to_sell = -position_delta  # Convert to positive
                for price, volume in available_buys:
                    # Volume is positive for buy orders
                    sell_quantity = min(remaining_to_sell, volume)
                    if sell_quantity > 0:
                        orders.append(Order(product, price, -sell_quantity))  # Negative for sell
                        print(f"Adding SELL order: {product} @ {price} x {sell_quantity}")
                        remaining_to_sell -= sell_quantity
                        
                        # Update last trade price
                        self.last_trade_price[product] = price
                        if current_position == 0:  # Starting a new position
                            self.position_entry_price[product] = price
                    
                    # Break after first level to avoid getting underpaid
                    break
        
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
        return self.cash + get_value_on_positions()

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
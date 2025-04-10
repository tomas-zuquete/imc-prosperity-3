import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState, OrderDepth, UserId

# Constants to avoid typos
SUBMISSION = "SUBMISSION"
SQUID_INK = "SQUID_INK"

PRODUCTS = [
    SQUID_INK
]

DEFAULT_PRICES = {
    SQUID_INK: 2030  # Starting middle price based on historical data
}

class Trader:
    def __init__(self) -> None:
        print("Initializing Trader...")

        # Position limits
        self.position_limit = {
            SQUID_INK: 50
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

        # Parameters for EMAs
        self.ema_short_param = 2 / (5 + 1)  # 5-period EMA
        self.ema_long_param = 2 / (20 + 1)  # 20-period EMA

        # Parameters for Z-score calculation
        self.z_score_window = 20
        self.rolling_mean = {SQUID_INK: None}
        self.rolling_std = {SQUID_INK: None}
        
        # Trading hours with superior performance (based on analysis)
        self.peak_hours = [3, 10, 13]
        
        # Mean reversion threshold parameters - made more aggressive to start trading earlier
        self.z_score_entry = 0.3  # Z-score threshold for entry (reduced from 0.8)
        self.z_score_exit = 0.1   # Z-score threshold for exit (reduced from 0.2)
        self.ema_threshold = 0.0005  # Minimum EMA difference to identify signal (reduced from 0.001)
        
        # Risk management parameter - increased to take more aggressive positions
        self.risk_factor = 0.8  # Default position sizing factor (increased from 0.5)
        
        # Hour tracking
        self.current_hour = {SQUID_INK: 0}
        
        # Volatility tracking
        self.volatility = {SQUID_INK: 0.01}  # Starting volatility estimate
        
        # Flag to force initial trades to start building history
        self.initial_trades_done = False

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
            
        # Calculate recent volatility (standard deviation of returns)
        if len(self.past_prices[product]) >= 5:  # Reduced from 20 to calculate volatility earlier
            prices = self.past_prices[product][-5:]
            returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            self.volatility[product] = max(math.sqrt(variance), 0.001)  # Minimum volatility floor

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
        
    def identify_signal(self, product, mid_price):
        """
        Identify trading signal using mean reversion strategy
        """
        # If we don't have enough data yet, start with some initial positions
        if len(self.past_prices[product]) < 10 and not self.initial_trades_done:
            # Take a small position to build history
            return "INITIAL_BUY"
            
        if self.ema_short[product] is None or self.ema_long[product] is None:
            return "NEUTRAL"
            
        # Calculate percentage difference between short and long EMAs
        diff_pct = (self.ema_short[product] - self.ema_long[product]) / self.ema_long[product]
        
        # REVERSED logic for mean reversion (buy when short EMA below long EMA)
        signal = "NEUTRAL"
        if diff_pct < -self.ema_threshold:
            signal = "BUY"
        elif diff_pct > self.ema_threshold:
            signal = "SELL"
            
        # Refine signal with z-score
        z_score = self.calculate_z_score(product, mid_price)
        
        # For debugging, always print z-score
        print(f"Z-score: {z_score:.4f}, EMA diff: {diff_pct*100:.4f}%")
        
        # Strong signals when z-score confirms
        if signal == "BUY" and z_score < -self.z_score_entry:
            return "STRONG_BUY"
        elif signal == "SELL" and z_score > self.z_score_entry:
            return "STRONG_SELL"
        elif signal in ("BUY", "SELL") and -self.z_score_exit < z_score < self.z_score_exit:
            return "EXIT"  # Exit positions when close to mean
            
        # If price moved significantly without EMA confirmation, still act on it
        if z_score < -self.z_score_entry * 1.5:  # Even more extreme negative z-score
            return "STRONG_BUY"
        elif z_score > self.z_score_entry * 1.5:  # Even more extreme positive z-score
            return "STRONG_SELL"
            
        return signal
        
    def calculate_position_size(self, product, signal, current_position):
        """
        Calculate optimal position size based on signal, volatility, and limits
        """
        position_limit = self.position_limit[product]
        
        # For initial trades, take a small position
        if signal == "INITIAL_BUY":
            self.initial_trades_done = True
            return min(10, position_limit - current_position)  # Buy 10 or up to limit
        
        # Special case for exit signal
        if signal == "EXIT":
            return -current_position  # Close entire position
        
        # Adjust position size based on volatility (lower for higher volatility)
        volatility_factor = 1 / (1 + self.volatility[product] * 5)  # Reduced multiplier from 10 to 5
        
        # Adjust based on hour of day
        hour_factor = 1.0
        if self.current_hour[product] in self.peak_hours:
            hour_factor = 1.2  # 20% boost during peak hours
        
        # Base position size
        base_size = math.floor(position_limit * self.risk_factor * volatility_factor * hour_factor)
        
        # Adjust based on signal strength
        if signal == "STRONG_BUY":
            target_position = base_size
        elif signal == "BUY":
            target_position = math.floor(base_size * 0.7)  # 70% size for regular signals (increased from 0.5)
        elif signal == "STRONG_SELL":
            target_position = -base_size
        elif signal == "SELL":
            target_position = -math.floor(base_size * 0.7)  # 70% size for regular signals (increased from 0.5)
        else:  # NEUTRAL
            target_position = 0
        
        # Limit the size to position limits
        target_position = max(-position_limit, min(position_limit, target_position))
        
        # Calculate position change needed
        position_delta = target_position - current_position
        
        return position_delta
        
    def SQUID_INK_strategy(self, state: TradingState):
        """
        Implement mean reversion strategy for SQUID_INK
        """
        product = SQUID_INK
        
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
        
        # Get trading signal
        signal = self.identify_signal(product, mid_price)
        
        # Calculate position change
        position_delta = self.calculate_position_size(product, signal, current_position)
        
        # Don't trade if no position change needed
        if position_delta == 0:
            print(f"No position change needed. Signal: {signal}")
            return []
        
        print(f"Taking action: {signal}, Position Delta: {position_delta}, Current Position: {current_position}")
            
        # Create orders list
        orders = []
        
        # Execute the position change
        if position_delta > 0:  # Need to buy
            # Buy using all available sell orders from best to worst price
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
                        
                    if remaining_to_buy <= 0:
                        break
            
            # If there are still units to buy, place a limit order at best bid + 1
            if remaining_to_buy > 0 and best_bid is not None:
                limit_price = best_bid + 1
                orders.append(Order(product, limit_price, remaining_to_buy))
                print(f"Adding LIMIT BUY order: {product} @ {limit_price} x {remaining_to_buy}")
                        
        elif position_delta < 0:  # Need to sell
            # Sell using all available buy orders from best to worst price
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
                        
                    if remaining_to_sell <= 0:
                        break
                        
            # If there are still units to sell, place a limit order at best ask - 1
            if remaining_to_sell > 0 and best_ask is not None:
                limit_price = best_ask - 1
                orders.append(Order(product, limit_price, -remaining_to_sell))
                print(f"Adding LIMIT SELL order: {product} @ {limit_price} x {remaining_to_sell}")
        
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
            result[SQUID_INK] = self.SQUID_INK_strategy(state)
        except Exception as e:
            print(f"Error in SQUID_INK strategy: {e}")
        
        # No conversions for SQUID_INK
        conversions = 0

        return result, conversions, None
import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState
import pandas as pd

# Constants
SUBMISSION = "SUBMISSION"

# Product names
RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

MM_PRODUCTS={
    "RESIN": "RAINFOREST_RESIN",
    "KELP": "KELP",
    "SQUID_INK": "SQUID_INK",
    "CROISSANTS": "CROISSANTS",
    "JAMS": "JAMS",
    "DJEMBES": "DJEMBES",
    "PICNIC_BASKET1": "PICNIC_BASKET1",
    "PICNIC_BASKET2": "PICNIC_BASKET2",
    "VOLCANIC_ROCK": "VOLCANIC_ROCK",
    "VOLCANIC_ROCK_VOUCHER_9500": "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750": "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000": "VOLCANIC_ROCK_VOUCHER_10000",
    "VOLCANIC_ROCK_VOUCHER_10250": "VOLCANIC_ROCK_VOUCHER_10250",
    "VOLCANIC_ROCK_VOUCHER_10500": "VOLCANIC_ROCK_VOUCHER_10500"
}

# List of all products
PRODUCTS = [
    RESIN,
    KELP,
    SQUID_INK,
    CROISSANTS,
    JAMS,
    DJEMBES,
    PICNIC_BASKET1,
    PICNIC_BASKET2,
    VOLCANIC_ROCK,
    VOLCANIC_ROCK_VOUCHER_9500,
    VOLCANIC_ROCK_VOUCHER_9750,
    VOLCANIC_ROCK_VOUCHER_10000,
    VOLCANIC_ROCK_VOUCHER_10250,
    VOLCANIC_ROCK_VOUCHER_10500
]

# Volcanic products
VOLCANIC_PRODUCTS = [
    VOLCANIC_ROCK,
    VOLCANIC_ROCK_VOUCHER_9500,
    VOLCANIC_ROCK_VOUCHER_9750,
    VOLCANIC_ROCK_VOUCHER_10000,
    VOLCANIC_ROCK_VOUCHER_10250,
    VOLCANIC_ROCK_VOUCHER_10500
]

# Default prices for reference
DEFAULT_PRICES = {
    RESIN: 10000,
    KELP: 2023,
    SQUID_INK: 1972,
    CROISSANTS: 500,
    JAMS: 500,
    DJEMBES: 500,
    PICNIC_BASKET1: 500,
    PICNIC_BASKET2: 500,
    VOLCANIC_ROCK: 10000,
    VOLCANIC_ROCK_VOUCHER_9500: 1000,
    VOLCANIC_ROCK_VOUCHER_9750: 750,
    VOLCANIC_ROCK_VOUCHER_10000: 500,
    VOLCANIC_ROCK_VOUCHER_10250: 275,
    VOLCANIC_ROCK_VOUCHER_10500: 100
}

# Helper functions for technical indicators
def compute_SMA(prices: List[float], window: int) -> float:
    if len(prices) < window:
        return sum(prices) / len(prices)
    return sum(prices[-window:]) / window

def compute_STD(prices: List[float], window: int) -> float:
    sma = compute_SMA(prices, window)
    var = sum((p - sma) ** 2 for p in prices[-window:]) / window
    return math.sqrt(var)

def norm_cdf(x):
    """
    Cumulative distribution function for the standard normal distribution
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price
    
    Args:
        S: Current price of underlying asset
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility of underlying asset
        
    Returns:
        Call option price
    """
    if T <= 0:
        # If expired, return intrinsic value
        return max(0, S - K)
        
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def black_scholes_call_delta(S, K, T, r, sigma):
    """
    Calculate Black-Scholes delta for call option
    
    Args:
        S: Current price of underlying asset
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility of underlying asset
        
    Returns:
        Call option delta
    """
    if T <= 0:
        # If expired, delta is either 0 or 1
        return 1.0 if S > K else 0.0
        
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

class Trader:
    def __init__(self) -> None:
        print("Initializing Trader...")

        # Position limits for each product
        self.position_limit = {
            RESIN: 50,
            KELP: 50, 
            SQUID_INK: 50,
            CROISSANTS: 250,
            JAMS: 350,
            DJEMBES: 60,
            PICNIC_BASKET1: 60,
            PICNIC_BASKET2: 100,
            VOLCANIC_ROCK: 400,
            VOLCANIC_ROCK_VOUCHER_9500: 200,
            VOLCANIC_ROCK_VOUCHER_9750: 200,
            VOLCANIC_ROCK_VOUCHER_10000: 200,
            VOLCANIC_ROCK_VOUCHER_10250: 200,
            VOLCANIC_ROCK_VOUCHER_10500: 200
        }

        self.round = 0
        self.cash = 0

        # Price history
        self.new_history = {
            RESIN: [],
            KELP: [],
            SQUID_INK: [],
            CROISSANTS: [], 
            JAMS: [], 
            DJEMBES: [],
            PICNIC_BASKET1: [],
            PICNIC_BASKET2: [],
            VOLCANIC_ROCK: [],
            VOLCANIC_ROCK_VOUCHER_9500: [],
            VOLCANIC_ROCK_VOUCHER_9750: [],
            VOLCANIC_ROCK_VOUCHER_10000: [],
            VOLCANIC_ROCK_VOUCHER_10250: [],
            VOLCANIC_ROCK_VOUCHER_10500: []
        }
        
        # EMA price tracking
        self.ema_param = 0.5
        self.ema_prices = {}
        for product in PRODUCTS:
            self.ema_prices[product] = None

        # Volcanic rock specific tracking
        self.volatility = 0.3  # Initial estimate
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Any, Any]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)

        print(f"Log round {self.round}")

        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)

        print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            if product in state.position:
                print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}")
        print(f"\tPnL {pnl}")

        # Initialize the method output dict as an empty dict
        result = {}
        
        # # Other strategies you may want to activate
        # try:
        #     result[SQUID_INK] = self.ink_strategy(state)
        # except Exception as e:
        #     print("Error occurred while executing ink_strategy:")
        #     print(f"Exception Type: {type(e).__name__}")
        #     print(f"Exception Message: {str(e)}")
        
        #############################################################################################################################

        # Trade each market-amking product individually
        try:
            result[RESIN] = self.resin_mm_strategy(state)
        except Exception as e:
            print(f"Error occurred while executing resin_mm_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}") 
            
        try:
            result[SQUID_INK] = self.squid_ink_mm_strategy(state)
        except Exception as e:
            print(f"Error occurred while executing resin_mm_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}") 
        
        try:
            result[PICNIC_BASKET1] = self.picnic_1_mm_strategy(state)
        except Exception as e:
            print(f"Error occurred while executing resin_mm_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            
        try:
            result[PICNIC_BASKET2] = self.picnic_2_mm_strategy(state)
        except Exception as e:
            print(f"Error occurred while executing resin_mm_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}") 
            
        conversions = 0
        return result, conversions, None

    # Utility methods
    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    def get_mid_price(self, product, state: TradingState):
        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask) / 2

    def get_value_on_product(self, product, state: TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)

    def update_pnl(self, state: TradingState):
        """
        Updates the pnl.
        """
        def update_cash():
            # Update cash
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
                value += self.get_value_on_product(product, state)
            return value

        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()

    def update_ema_prices(self, state: TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            if product in state.order_depths:
                mid_price = self.get_mid_price(product, state)
                if mid_price is None:
                    continue

                # Update ema price
                if self.ema_prices[product] is None:
                    self.ema_prices[product] = mid_price
                else:
                    self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

    # Resin pnl - 1.91k
    def resin_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for resin rock with more aggressive parameters
        
        Args:
            state: Current trading state
            
        Returns:
            List of orders for volcanic rock
        """
        product = RESIN
        pos = self.get_position(product, state)
        
        # Get order book data
        order_depth = state.order_depths[product]
        
        # Calculate fair value using mid price
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
        else:
            mid = self.get_mid_price(product, state)
        
        # Store price for volatility calculation
        if product not in self.new_history:
            self.new_history[product] = []
        self.new_history[product].append(mid)
        
        # Get position limit
        position_limit = self.position_limit[product]
        
        # Calculate available capacity
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        orders = []
        
        # Market making approach - try to capture bid-ask spread
        if order_depth.buy_orders and order_depth.sell_orders:
            # If spread is wide enough, place orders inside the spread
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            
            # Only provide liquidity if spread is favorable
            if spread > 2:
                # Place orders inside the spread
                our_bid = best_bid + 1
                our_ask = best_ask - 1
                
                # Ensure we're not crossing the spread
                if our_bid >= our_ask:
                    our_bid = best_bid
                    our_ask = best_ask
                
                # Calculate quantities based on existing orders and our capacity
                best_bid_quantity = abs(sum(order_depth.buy_orders.values()))
                best_ask_quantity = abs(sum(order_depth.sell_orders.values()))
                
                # Start with base sizes
                bid_size = min(40, buy_capacity)
                ask_size = min(40, sell_capacity)
                
                # Add bid order if we have capacity
                if buy_capacity > 0:
                    orders.append(Order(product, our_bid, bid_size))
                
                # Add ask order if we have capacity
                if sell_capacity > 0:
                    orders.append(Order(product, our_ask, -ask_size))
        
        # Directional trading based on position
        # If we have a significant position, try to revert to neutral
        if pos > position_limit * 0.3:  # If we're long more than 30% of capacity
            # Try to sell at market
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(abs(pos), abs(bid_volume))
                orders.append(Order(product, best_bid, -sell_size))
        
        elif pos < -position_limit * 0.3:  # If we're short more than 30% of capacity
            # Try to buy at market
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_volume = order_depth.sell_orders[best_ask]
                buy_size = min(abs(pos), abs(ask_volume))
                orders.append(Order(product, best_ask, buy_size))
        
        # If few or no orders, add orders near mid price
        if len(orders) < 2:
            # Calculate bid and ask prices around mid
            bid_price = int(mid - 2)
            ask_price = int(mid + 2)
            
            # Add missing orders
            if not any(o.price == bid_price and o.quantity > 0 for o in orders) and buy_capacity > 0:
                orders.append(Order(product, bid_price, min(40, buy_capacity)))
            
            if not any(o.price == ask_price and o.quantity < 0 for o in orders) and sell_capacity > 0:
                orders.append(Order(product, ask_price, -min(40, sell_capacity)))
        
        # Print debug info
        print(f"[{product}] pos={pos}, mid={mid:.1f}, orders={[(o.price, o.quantity) for o in orders]}")
        
        return orders
    
    # Squid pnl - 313 - combined - 2224
    def squid_ink_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for resin rock with more aggressive parameters
        
        Args:
            state: Current trading state
            
        Returns:
            List of orders for volcanic rock
        """
        product = SQUID_INK
        pos = self.get_position(product, state)
        
        # Get order book data
        order_depth = state.order_depths[product]
        
        # Calculate fair value using mid price
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
        else:
            mid = self.get_mid_price(product, state)
        
        # Store price for volatility calculation
        if product not in self.new_history:
            self.new_history[product] = []
        self.new_history[product].append(mid)
        
        # Get position limit
        position_limit = self.position_limit[product]
        
        # Calculate available capacity
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        orders = []
        
        # Market making approach - try to capture bid-ask spread
        if order_depth.buy_orders and order_depth.sell_orders:
            # If spread is wide enough, place orders inside the spread
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            
            # Only provide liquidity if spread is favorable
            if spread > 2:
                # Place orders inside the spread
                our_bid = best_bid + 1
                our_ask = best_ask - 1
                
                # Ensure we're not crossing the spread
                if our_bid >= our_ask:
                    our_bid = best_bid
                    our_ask = best_ask
                
                # Calculate quantities based on existing orders and our capacity
                best_bid_quantity = abs(sum(order_depth.buy_orders.values()))
                best_ask_quantity = abs(sum(order_depth.sell_orders.values()))
                
                # Start with base sizes
                bid_size = min(50, buy_capacity)
                ask_size = min(50, sell_capacity)
                
                # Add bid order if we have capacity
                if buy_capacity > 0:
                    orders.append(Order(product, our_bid, bid_size))
                
                # Add ask order if we have capacity
                if sell_capacity > 0:
                    orders.append(Order(product, our_ask, -ask_size))
        
        # Directional trading based on position
        # If we have a significant position, try to revert to neutral
        if pos > position_limit * 0.4:  # If we're long more than 30% of capacity
            # Try to sell at market
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(abs(pos), abs(bid_volume))
                orders.append(Order(product, best_bid, -sell_size))
        
        elif pos < -position_limit * 0.4:  # If we're short more than 30% of capacity
            # Try to buy at market
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_volume = order_depth.sell_orders[best_ask]
                buy_size = min(abs(pos), abs(ask_volume))
                orders.append(Order(product, best_ask, buy_size))
        
        # If few or no orders, add orders near mid price
        if len(orders) < 2:
            # Calculate bid and ask prices around mid
            bid_price = int(mid - 2)
            ask_price = int(mid + 2)
            
            # Add missing orders
            if not any(o.price == bid_price and o.quantity > 0 for o in orders) and buy_capacity > 0:
                orders.append(Order(product, bid_price, min(40, buy_capacity)))
            
            if not any(o.price == ask_price and o.quantity < 0 for o in orders) and sell_capacity > 0:
                orders.append(Order(product, ask_price, -min(40, sell_capacity)))
        
        # Print debug info
        print(f"[{product}] pos={pos}, mid={mid:.1f}, orders={[(o.price, o.quantity) for o in orders]}")
        
        return orders
     
    # Picnic 1 pnl - 243 - combined - 2467
    def picnic_1_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for resin rock with more aggressive parameters
        
        Args:
            state: Current trading state
            
        Returns:
            List of orders for volcanic rock
        """
        product = PICNIC_BASKET1
        pos = self.get_position(product, state)
        
        # Get order book data
        order_depth = state.order_depths[product]
        
        # Calculate fair value using mid price
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
        else:
            mid = self.get_mid_price(product, state)
        
        # Store price for volatility calculation
        if product not in self.new_history:
            self.new_history[product] = []
        self.new_history[product].append(mid)
        
        # Get position limit
        position_limit = self.position_limit[product]
        
        # Calculate available capacity
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        orders = []
        
        # Market making approach - try to capture bid-ask spread
        if order_depth.buy_orders and order_depth.sell_orders:
            # If spread is wide enough, place orders inside the spread
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            
            # Only provide liquidity if spread is favorable
            if spread > 2:
                # Place orders inside the spread
                our_bid = best_bid + 1
                our_ask = best_ask - 1
                
                # Ensure we're not crossing the spread
                if our_bid >= our_ask:
                    our_bid = best_bid
                    our_ask = best_ask
                
                # Calculate quantities based on existing orders and our capacity
                best_bid_quantity = abs(sum(order_depth.buy_orders.values()))
                best_ask_quantity = abs(sum(order_depth.sell_orders.values()))
                
                # Start with base sizes
                bid_size = min(20, buy_capacity)
                ask_size = min(20, sell_capacity)
                
                # Add bid order if we have capacity
                if buy_capacity > 0:
                    orders.append(Order(product, our_bid, bid_size))
                
                # Add ask order if we have capacity
                if sell_capacity > 0:
                    orders.append(Order(product, our_ask, -ask_size))
        
        # Directional trading based on position
        # If we have a significant position, try to revert to neutral
        if pos > position_limit * 0.4:  # If we're long more than 30% of capacity
            # Try to sell at market
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(abs(pos), abs(bid_volume))
                orders.append(Order(product, best_bid, -sell_size))
        
        elif pos < -position_limit * 0.4:  # If we're short more than 30% of capacity
            # Try to buy at market
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_volume = order_depth.sell_orders[best_ask]
                buy_size = min(abs(pos), abs(ask_volume))
                orders.append(Order(product, best_ask, buy_size))
        
        # If few or no orders, add orders near mid price
        if len(orders) < 2:
            # Calculate bid and ask prices around mid
            bid_price = int(mid - 2)
            ask_price = int(mid + 2)
            
            # Add missing orders
            if not any(o.price == bid_price and o.quantity > 0 for o in orders) and buy_capacity > 0:
                orders.append(Order(product, bid_price, min(40, buy_capacity)))
            
            if not any(o.price == ask_price and o.quantity < 0 for o in orders) and sell_capacity > 0:
                orders.append(Order(product, ask_price, -min(40, sell_capacity)))
        
        # Print debug info
        print(f"[{product}] pos={pos}, mid={mid:.1f}, orders={[(o.price, o.quantity) for o in orders]}")
        
        return orders
    
    # Picnic 2 pnl - 532 - combined - 2938
    def picnic_2_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for resin rock with more aggressive parameters
        
        Args:
            state: Current trading state
            
        Returns:
            List of orders for volcanic rock
        """
        product = PICNIC_BASKET2
        pos = self.get_position(product, state)
        
        # Get order book data
        order_depth = state.order_depths[product]
        
        # Calculate fair value using mid price
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
        else:
            mid = self.get_mid_price(product, state)
        
        # Store price for volatility calculation
        if product not in self.new_history:
            self.new_history[product] = []
        self.new_history[product].append(mid)
        
        # Get position limit
        position_limit = self.position_limit[product]
        
        # Calculate available capacity
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        orders = []
        
        # Market making approach - try to capture bid-ask spread
        if order_depth.buy_orders and order_depth.sell_orders:
            # If spread is wide enough, place orders inside the spread
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            
            # Only provide liquidity if spread is favorable
            if spread > 3:
                # Place orders inside the spread
                our_bid = best_bid + 10
                our_ask = best_ask - 10
                
                # Ensure we're not crossing the spread
                if our_bid >= our_ask:
                    our_bid = best_bid
                    our_ask = best_ask
                
                # Calculate quantities based on existing orders and our capacity
                best_bid_quantity = abs(sum(order_depth.buy_orders.values()))
                best_ask_quantity = abs(sum(order_depth.sell_orders.values()))
                
                # Start with base sizes
                bid_size = min(10, buy_capacity)
                ask_size = min(10, sell_capacity)
                
                # Add bid order if we have capacity
                if buy_capacity > 1:
                    orders.append(Order(product, our_bid, bid_size))
                
                # Add ask order if we have capacity
                if sell_capacity > 1:
                    orders.append(Order(product, our_ask, -ask_size))
        
        # Directional trading based on position
        # If we have a significant position, try to revert to neutral
        if pos > position_limit * 0.15:  # If we're long more than 30% of capacity
            # Try to sell at market
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(abs(pos), abs(bid_volume))
                orders.append(Order(product, best_bid, -sell_size))
        
        elif pos < -position_limit * 0.15:  # If we're short more than 30% of capacity
            # Try to buy at market
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_volume = order_depth.sell_orders[best_ask]
                buy_size = min(abs(pos), abs(ask_volume))
                orders.append(Order(product, best_ask, buy_size))
        
        # If few or no orders, add orders near mid price
        if len(orders) < 4:
            # Calculate bid and ask prices around mid
            bid_price = int(mid - 2)
            ask_price = int(mid + 2)
            
            # Add missing orders
            if not any(o.price == bid_price and o.quantity > 0 for o in orders) and buy_capacity > 0:
                orders.append(Order(product, bid_price, min(40, buy_capacity)))
            
            if not any(o.price == ask_price and o.quantity < 0 for o in orders) and sell_capacity > 0:
                orders.append(Order(product, ask_price, -min(40, sell_capacity)))
        
        # Print debug info
        print(f"[{product}] pos={pos}, mid={mid:.1f}, orders={[(o.price, o.quantity) for o in orders]}")
        
        return orders
    
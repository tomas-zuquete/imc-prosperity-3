import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState
import pandas as pd

# Constants
SUBMISSION = "SUBMISSION"

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
    VOLCANIC_ROCK_VOUCHER_9500: 500,
    VOLCANIC_ROCK_VOUCHER_9750: 500,
    VOLCANIC_ROCK_VOUCHER_10000: 500,
    VOLCANIC_ROCK_VOUCHER_10250: 500,
    VOLCANIC_ROCK_VOUCHER_10500: 500
}

def norm_pdf(x: float) -> float:
    """Approximate the standard normal PDF using a simple formula."""
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def norm_cdf(x: float) -> float:
    """Approximate the standard normal CDF using a polynomial approximation."""
    # Hart's approximation for the standard normal CDF (accurate for practical purposes)
    if x < -10:
        return 0.0
    if x > 10:
        return 1.0
    
    # Constants for Hart's approximation
    a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    t = 1 / (1 + p * abs(x))
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
    z = norm_pdf(x)
    cdf = 1 - z * poly
    
    return cdf if x >= 0 else 1 - cdf

# Utility functions
def compute_SMA(prices: List[float], window: int) -> float:
    if len(prices) < window:
        return sum(prices) / len(prices) if prices else 0
    return sum(prices[-window:]) / window

def compute_STD(prices: List[float], window: int) -> float:
    sma = compute_SMA(prices, window)
    if len(prices) < window:
        return 0
    var = sum((p - sma) ** 2 for p in prices[-window:]) / window
    return math.sqrt(var)

def compute_RSI(prices: List[float], window: int = 14) -> float:
    if len(prices) < window + 1:
        return 50.0
    gains = [max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices))][-window:]
    losses = [max(prices[i - 1] - prices[i], 0) for i in range(1, len(prices))][-window:]
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))

def compute_stochastic(prices: List[float], window: int = 14) -> float:
    if len(prices) < window:
        return 50.0
    lowest = min(prices[-window:])
    highest = max(prices[-window:])
    if highest == lowest:
        return 50.0
    return (prices[-1] - lowest) / (highest - lowest) * 100.0

def compute_ATR(prices: List[float], window: int = 14) -> float:
    if len(prices) < window + 1:
        return 0.0
    tr_values = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))][-window:]
    return sum(tr_values) / window

def update_EMA(prev_ema: float, price: float, window: int) -> float:
    alpha = 2 / (window + 1)
    return alpha * price + (1 - alpha) * prev_ema

# Black-Scholes
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> Tuple[float, float, float]:
     
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        price = max(S - K, 0)  # Intrinsic value of the call
        delta = 1 if S > K else 0  # Delta is 1 if in-the-money, else 0
        gamma = 0  # No curvature for expired or invalid options
        return price, delta, gamma

    try:
        # Calculate d1 and d2
        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T
        if sigma_sqrt_T == 0:
            price = max(S - K, 0)
            delta = 1 if S > K else 0
            gamma = 0
            return price, delta, gamma
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        # Calculate price, delta, and gamma
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = math.exp(-q * T) * norm_cdf(d1)
        
        # Gamma calculation: protect against S * sigma * sqrt(T) being zero
        denominator = S * sigma_sqrt_T
        gamma = math.exp(-q * T) * norm_pdf(d1) / denominator if denominator != 0 else 0

        return price, delta, gamma

    except (ValueError, ZeroDivisionError, OverflowError) as e:
        # Handle numerical errors (e.g., log of zero, overflow)
        print(f"Error in Black-Scholes: {e}, S={S}, K={K}, T={T}, sigma={sigma}")
        price = max(S - K, 0)  # Fallback to intrinsic value
        delta = 1 if S > K else 0
        gamma = 0
        return price, delta, gamma

class Trader:
    def __init__(self) -> None:
        print("Initializing Trader...")

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
        self.ou_params = {"mu": -0.000013, "theta": 0.01, "sigma": 0.000025}
        self.kelp_ou_params = {"mu": 0.000006, "theta": 0.01, "sigma": 0.000713}
        self.past_prices = {product: [] for product in PRODUCTS}
        self.ema_param = 0.5
        self.new_history = {CROISSANTS: [], JAMS: [], DJEMBES: []}
        self.basket1_history = []
        self.basket1_ema = None
        self.ema_prices = {product: None for product in PRODUCTS}
        self.ink_prev_mid_price = None
        self.kelp_prev_mid_price = None
        self.ink_mid_prices = []
        self.kelp_mid_prices = []

        # Voucher-specific state
        self.voucher_strikes = {
            VOLCANIC_ROCK_VOUCHER_9500: 9500,
            VOLCANIC_ROCK_VOUCHER_9750: 9750,
            VOLCANIC_ROCK_VOUCHER_10000: 10000,
            VOLCANIC_ROCK_VOUCHER_10250: 10250,
            VOLCANIC_ROCK_VOUCHER_10500: 10500
        }
        self.voucher_volatility = 0.0009450871502416238  
        self.risk_free_rate = 0.0435
        self.time_to_maturity = 7 / 365  # 7 days from round 1
        self.base_spread = 0.25  # Base spread for vouchers

    def get_position(self, product, state: TradingState) -> int:
        return state.position.get(product, 0)

    def get_order_ratio(self, product, state: TradingState) -> float:
        market_bids = state.order_depths.get(product, {}).buy_orders.keys()
        market_asks = state.order_depths.get(product, {}).sell_orders.keys()
        if len(market_asks) > 0 and len(market_bids) > 0:
            return (sum(market_bids) - sum(market_asks)) / (sum(market_bids) + sum(market_asks))
        return 0

    def get_mid_price(self, product, state: TradingState) -> float:
        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        if not market_bids or not market_asks:
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask) / 2

    def get_value_on_product(self, product, state: TradingState) -> float:
        if product in self.voucher_strikes:
            S = self.get_mid_price(VOLCANIC_ROCK, state)
            K = self.voucher_strikes[product]
            price, _, _ = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, self.voucher_volatility, q=0)
            return self.get_position(product, state) * price
        return self.get_position(product, state) * self.get_mid_price(product, state)

    def update_pnl(self, state: TradingState) -> float:
        def update_cash():
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
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

        update_cash()
        return self.cash + get_value_on_positions()

    def update_ema_prices(self, state: TradingState):
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if product in self.voucher_strikes:
                S = self.get_mid_price(VOLCANIC_ROCK, state)
                K = self.voucher_strikes[product]
                mid_price, _, _ = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, self.voucher_volatility, q=0)
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1 - self.ema_param) * self.ema_prices[product]

    def implied_volatility(self, S, K, T, r, option_price, q=0) -> float:
        sigma = 0.3
        for _ in range(50):
            price, _, _ = black_scholes_call(S, K, T, r, sigma, q)
        
            if abs(price) < 1e-6:
                return 0.3
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            vega = S * norm_pdf(d1) * math.sqrt(T)
            if vega < 1e-6:
                break
            diff = price - option_price
            if abs(diff) < 1e-6:
                break
            sigma -= diff / vega
            if sigma < 0.01 or sigma > 1.0:
                break
        return max(sigma, 0.1)

    def update_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return max(self.voucher_volatility, 0.5)
        log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        if log_returns:
            self.voucher_volatility = max(np.std(log_returns) * math.sqrt(252), 0.1)
        return self.voucher_volatility
  
    def croissants_strategy(self, state: TradingState) -> List[Order]:
        product = CROISSANTS
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        self.new_history[product].append(mid)
        prices = self.new_history[product]
        window = 20
        sma20 = compute_SMA(prices, window) if len(prices) >= window else compute_SMA(prices, len(prices))
        std20 = compute_STD(prices, window) if len(prices) >= window else compute_STD(prices, len(prices))
        lower_band = sma20 - 2 * std20
        rsi = compute_RSI(prices, 14)
        signal = 0
        if mid <= lower_band and rsi < 28:
            signal = 1
        elif mid >= sma20 or rsi > 72:
            signal = -1
        if signal == 1:
            bid_price = int(mid - 1)
            ask_price = int(mid + 2)
        elif signal == -1:
            bid_price = int(mid - 2)
            ask_price = int(mid + 1)
        else:
            bid_price = int(mid - 1)
            ask_price = int(mid + 1)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    def jams_strategy(self, state: TradingState) -> List[Order]:
        product = JAMS
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        self.new_history[product].append(mid)
        prices = self.new_history[product]
        stoch_val = compute_stochastic(prices, 14)
        signal = 0
        if stoch_val < 25:
            signal = 1
        elif stoch_val > 75:
            signal = -1
        if signal == 1:
            bid_price = int(mid - 1)
            ask_price = int(mid + 2)
        elif signal == -1:
            bid_price = int(mid - 2)
            ask_price = int(mid + 1)
        else:
            bid_price = int(mid - 1)
            ask_price = int(mid + 1)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    def djembes_strategy(self, state: TradingState) -> List[Order]:
        product = DJEMBES
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        self.new_history[product].append(mid)
        prices = self.new_history[product]
        rsi = compute_RSI(prices, 14)
        atr = compute_ATR(prices, 14)
        atr_med = np.median(prices[-14:]) if len(prices) >= 14 else atr
        signal = 0
        if rsi < 28 and atr > atr_med:
            signal = 1
        elif rsi > 72 and atr > atr_med:
            signal = -1
        if signal == 1:
            bid_price = int(mid - 1)
            ask_price = int(mid + 2)
        elif signal == -1:
            bid_price = int(mid - 2)
            ask_price = int(mid + 1)
        else:
            bid_price = int(mid - 1)
            ask_price = int(mid + 1)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

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

    def kelp_strategy(self, state: TradingState) -> List[Order]:
        mid_price = self.get_mid_price(KELP, state)
        self.kelp_mid_prices.append(mid_price)
        perc_diff = 0
        if self.kelp_prev_mid_price and self.kelp_prev_mid_price != 0:
            perc_diff = (mid_price - self.kelp_prev_mid_price) / self.kelp_prev_mid_price
        self.kelp_prev_mid_price = mid_price
        if len(self.kelp_mid_prices) >= 20:
            X = [(self.kelp_mid_prices[i] - self.kelp_mid_prices[i-1]) / self.ink_mid_prices[i-1]
                for i in range(1, len(self.kelp_mid_prices)) if self.kelp_mid_prices[i-1] != 0]
            if X:
                mu, theta, sigma = self.fit_ou(X)
                self.kelp_ou_params.update({"mu": mu, "theta": theta, "sigma": sigma})
                print(f"Updated OU params: mu={mu:.6f}, theta={theta:.4f}, sigma={sigma:.6f}")
        mu, theta, sigma = self.kelp_ou_params["mu"], self.kelp_ou_params["theta"], self.kelp_ou_params["sigma"]
        position_kelp = self.get_position(KELP, state)
        z_score = (perc_diff - mu) / (sigma / np.sqrt(2 * theta)) if sigma > 0 else 0
        orders = []
        max_trade_size = 50
        bid_volume = self.position_limit[KELP] - position_kelp
        ask_volume = -self.position_limit[KELP] - position_kelp
        best_ask = min(state.order_depths[KELP].sell_orders.keys(), default=int(mid_price + 1))
        best_bid = max(state.order_depths[KELP].buy_orders.keys(), default=int(mid_price - 1))
        if perc_diff != 0:
            if z_score >= 0.5 and ask_volume > 0:
                size = min(max_trade_size, ask_volume, abs(state.order_depths[KELP].buy_orders.get(best_bid, 0)))
                if size > 0:
                    orders.append(Order(KELP, best_bid, -size))
            elif z_score <= -0.5 and bid_volume > 0:
                size = min(max_trade_size, bid_volume, abs(state.order_depths[KELP].sell_orders.get(best_ask, 0)))
                if size > 0:
                    orders.append(Order(KELP, best_ask, size))
            else:
                fair_price = mid_price * (1 + mu)
                bid_price = int(fair_price - 1)
                ask_price = int(fair_price + 1)
                orders.append(Order(KELP, bid_price, bid_volume))
                orders.append(Order(KELP, ask_price, ask_volume))
        print(f"z_score: {z_score:.2f}, perc_diff: {perc_diff:.6f}, orders: {[('Kelp', o.price, o.quantity) for o in orders]}")
        return orders
 

    # def volcanic_rock_vouchers_strategy(self, state: TradingState) -> Dict[str, List[Order]]:
    #     """Market-making strategy for VOLCANIC_ROCK vouchers."""
    #     result = {}
    #     S = self.get_mid_price(VOLCANIC_ROCK, state) or DEFAULT_PRICES[VOLCANIC_ROCK]
    #     self.past_prices[VOLCANIC_ROCK].append(S)
    #     if len(self.past_prices[VOLCANIC_ROCK]) > 25: #50
    #         self.update_volatility(self.past_prices[VOLCANIC_ROCK][-25:])  

    #     total_delta = 0
        
    #     rock_pos = self.get_position(VOLCANIC_ROCK, state)
    #     rock_bid_volume = self.position_limit[VOLCANIC_ROCK] - rock_pos
    #     rock_ask_volume = -self.position_limit[VOLCANIC_ROCK] - rock_pos
    #     hedging_orders = []

    #     trade_size = 50 #200
 
    #     for product in self.voucher_strikes:
    #         K = self.voucher_strikes[product]
    #         pos = self.get_position(product, state)

    #         # Use market-based pricing
    #         sigma = self.voucher_volatility
    #         market_price = None
    #         best_bid = None
    #         best_ask = None
    #         if product in state.order_depths and state.order_depths[product].buy_orders and state.order_depths[product].sell_orders:
    #             market_price = self.get_mid_price(product, state) or DEFAULT_PRICES[PRODUCTS]
    #             best_bid = max(state.order_depths[product].buy_orders.keys())
    #             best_ask = min(state.order_depths[product].sell_orders.keys())
    #             sigma = self.implied_volatility(S, K, self.time_to_maturity, self.risk_free_rate, market_price)
    #             sigma = max(sigma, 1e-6) if sigma is not None else 1e-6
    #         price, delta, gamma = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, sigma, q=0)

    #         spread = 1.5 + min(0.3 * gamma, 0.4)
    #         inventory_adjust = 0.02 * pos # 0.2
    #         fair_bid = (price - spread) / 2 - inventory_adjust
    #         fair_ask = (price + spread) / 2 - inventory_adjust

    #         # Match order book if competitive
    #         bid_price = int(best_bid) if best_bid and fair_bid <= best_bid else max(int(fair_bid), 1)
    #         ask_price = int(best_ask) if best_ask and fair_ask >= best_ask else int(fair_ask)

    #         bid_volume = self.position_limit[product] - pos
    #         ask_volume = -self.position_limit[product] - pos
    #         print("OK2")
    #         orders = []
    #         if bid_volume > 0:
    #             orders.append(Order(product, bid_price, min(bid_volume, trade_size)))
    #         if ask_volume < 0:
    #             orders.append(Order(product, ask_price, max(ask_volume, -trade_size)))
    #         result[product] = orders

    #         total_delta += pos * delta 
    #         print("total_delta: ", total_delta)

    #         # Diagnostics
    #         market_mid = market_price if market_price else price
    #         print(f"{product}: bid={bid_price}, ask={ask_price}, best_bid={best_bid or 'N/A'}, best_ask={best_ask or 'N/A'}, pos={pos}, fair={price:.2f}, market={market_mid:.2f}, sigma={sigma:.3f}")

    #     voucher_value = 0
    #     hedging_pnl = 0
    #     for product in self.voucher_strikes:
    #         value = self.get_value_on_product(product, state)
    #         voucher_value += value
    #         pos = self.get_position(product, state)
    #         print(f"{product}: pos={pos}, value={value:.2f}")

    #     for product in state.own_trades:
    #         if product in self.voucher_strikes or product == VOLCANIC_ROCK:
    #             for trade in state.own_trades[product]:
    #                 if trade.timestamp == state.timestamp - 100:
    #                     mid = self.get_mid_price(product, state)
    #                     profit = (mid - trade.price) * trade.quantity if trade.buyer == SUBMISSION else (trade.price - mid) * trade.quantity
    #                     if product == VOLCANIC_ROCK:
    #                         hedging_pnl += profit
    #                     print(f"Trade: {product}, price={trade.price}, qty={trade.quantity}, side={'buy' if trade.buyer == SUBMISSION else 'sell'}, profit={profit:.2f}")
    #     print(f"Cash: {self.cash}, Voucher Value: {voucher_value:.2f}, Hedging P&L: {hedging_pnl:.2f}, Total Delta: {total_delta:.2f}")

    #     # Delta hedging
    #     target_rock_pos = -total_delta
    #     rock_trade = int(target_rock_pos - rock_pos)
       
    #     best_ask = min(state.order_depths.get(VOLCANIC_ROCK, {}).sell_orders.keys(), default=S)
    #     best_bid = max(state.order_depths.get(VOLCANIC_ROCK, {}).buy_orders.keys(), default=S)
    #     spread = best_ask - best_bid
    #     if spread > 5:
    #         best_ask = best_bid = S
    #     rock_trade = max(min(rock_trade, 400), -400)  
    #     if rock_trade > 0 and rock_bid_volume >= rock_trade:
    #         hedging_orders.append(Order(VOLCANIC_ROCK, int(best_ask), rock_trade))
    #         print(f"Hedge: Buy {rock_trade} VOLCANIC_ROCK at {best_ask}, delta={total_delta:.2f}")
    #     elif rock_trade < 0 and rock_ask_volume <= rock_trade:
    #         hedging_orders.append(Order(VOLCANIC_ROCK, int(best_bid), -rock_trade))
    #         print(f"Hedge: Sell {-rock_trade} VOLCANIC_ROCK at {best_bid}, delta={total_delta:.2f}")

    #     result[VOLCANIC_ROCK] = hedging_orders
    #     return result
   
   
    
    def volcanic_rock_vouchers_strategy(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Implements the volatility smile trading strategy for volcanic rock vouchers
        with improved risk management and hedging
        
        This strategy:
        1. Models the volatility smile curve with m_t = log(K/St)/sqrt(TTE)
        2. Identifies relatively mispriced options from the curve
        3. Trades based on deviations from the modeled curve
        4. Implements proper delta hedging with more frequent adjustments
        5. Adds position balancing and risk limits
        """
        result = {}
        
        # Get underlying price
        S = self.get_mid_price(VOLCANIC_ROCK, state) or DEFAULT_PRICES[VOLCANIC_ROCK]
        self.past_prices[VOLCANIC_ROCK].append(S)
        
        # Update volatility estimate
        if len(self.past_prices[VOLCANIC_ROCK]) > 25:
            self.update_volatility(self.past_prices[VOLCANIC_ROCK][-25:])
        
        # Initialize delta tracking
        total_delta = 0
        
        # Get rock position for hedging
        rock_pos = self.get_position(VOLCANIC_ROCK, state)
        rock_bid_volume = self.position_limit[VOLCANIC_ROCK] - rock_pos
        rock_ask_volume = -self.position_limit[VOLCANIC_ROCK] - rock_pos
        hedging_orders = []
        
        # Define maximum notional exposure per product for risk management
        max_notional_per_product = 100000  # Maximum value exposure per product
        
        # Collect data for volatility smile modeling
        moneyness_points = []
        iv_points = []
        product_data = {}
        
        # First pass: collect data points for volatility smile modeling
        for product in self.voucher_strikes:
            # Skip if product not in order book
            if product not in state.order_depths:
                continue
                
            K = self.voucher_strikes[product]
            pos = self.get_position(product, state)
            
            # Calculate time to expiry (decreases by 1 day per round)
            # Round 1: 7 days, Round 2: 6 days, etc.
            days_elapsed = min(5, max(0, self.round // 100))  # Rough estimation of rounds passed
            remaining_days = 7 - days_elapsed
            tte = max(0.01, remaining_days / 365.0)  # In years, minimum of 0.01 to avoid division by zero
            self.time_to_maturity = tte
            
            # Calculate moneyness
            if tte > 0 and S > 0 and K > 0:
                moneyness = math.log(K / S) / math.sqrt(tte)
            else:
                moneyness = 0
                
            # Get market price if available
            market_price = None
            if (product in state.order_depths and 
                state.order_depths[product].buy_orders and 
                state.order_depths[product].sell_orders):
                best_bid = max(state.order_depths[product].buy_orders.keys())
                best_ask = min(state.order_depths[product].sell_orders.keys())
                market_price = (best_bid + best_ask) / 2
                
                # Calculate implied volatility from market price
                iv = self.implied_volatility(S, K, tte, self.risk_free_rate, market_price)
                iv = max(0.01, min(iv, 1.0))  # Constrain IV to reasonable values
                
                # Store valid data points for smile modeling
                if not math.isnan(iv) and iv > 0:
                    moneyness_points.append(moneyness)
                    iv_points.append(iv)
                    
                    # Store product data for trading decisions
                    product_data[product] = {
                        'strike': K,
                        'moneyness': moneyness,
                        'market_iv': iv,
                        'market_price': market_price,
                        'position': pos,
                        'best_bid': best_bid,
                        'best_ask': best_ask
                    }
        
        # Fit volatility smile curve if we have enough data points
        predicted_ivs = {}
        if len(moneyness_points) >= 3:
            try:
                # Fit quadratic curve: iv = a*m^2 + b*m + c
                z = np.polyfit(moneyness_points, iv_points, 2)
                a, b, c = z
                
                # Smooth volatility parameters using exponential smoothing
                if hasattr(self, 'prev_smile_params'):
                    alpha = 0.3  # Smoothing factor
                    a = alpha * a + (1-alpha) * self.prev_smile_params['a']
                    b = alpha * b + (1-alpha) * self.prev_smile_params['b']
                    c = alpha * c + (1-alpha) * self.prev_smile_params['c']
                
                self.prev_smile_params = {"a": a, "b": b, "c": c}
                
                print(f"Volatility smile parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")
                
                # Calculate predicted IV for each voucher
                for product, data in product_data.items():
                    m = data['moneyness']
                    predicted_iv = a * m**2 + b * m + c
                    data['predicted_iv'] = predicted_iv
                    data['iv_deviation'] = data['market_iv'] - predicted_iv
                    data['relative_deviation'] = data['iv_deviation'] / predicted_iv if predicted_iv > 0 else 0
                    predicted_ivs[product] = predicted_iv
            except Exception as e:
                print(f"Error fitting volatility smile: {e}")
        
        # Second pass: make trading decisions based on volatility smile model
        # Optimize trade sizes based on position limits, deviation magnitude, and risk
        for product in self.voucher_strikes:
            if product not in product_data:
                continue
                
            K = self.voucher_strikes[product]
            pos = self.get_position(product, state)
            data = product_data[product]
            
            # If we have a smile model, use it for trading decisions
            if product in predicted_ivs:
                # Calculate theoretical price using predicted IV
                price_using_model_iv, delta, gamma = black_scholes_call(
                    S, K, self.time_to_maturity, self.risk_free_rate, 
                    predicted_ivs[product], q=0)
                    
                # Calculate price using market IV for comparison
                price_using_market_iv, delta, gamma = black_scholes_call(
                    S, K, self.time_to_maturity, self.risk_free_rate, 
                    data['market_iv'], q=0)
                    
                # Calculate spread based on option characteristics
                # Higher gamma (curvature) means higher risk, so increase spread
                spread = 1.5 + min(0.3 * abs(gamma), 0.4)
                
                # Adjust price based on position - mean reversion component
                # More aggressive inventory adjustment
                inventory_adjust = 0.02 * pos
                
                # Calculate trading threshold based on moneyness (higher for far OTM/ITM)
                base_threshold = 0.03
                moneyness_adj = min(3.0, 1.0 + abs(data['moneyness']))
                threshold = base_threshold * moneyness_adj
                
                # Trading signals based on implied volatility deviation
                orders = []
                relative_deviation = data['relative_deviation']
                
                # OPTIMIZED TRADE SIZING WITH POSITION BALANCING
                # Calculate position limits and available capacity
                position_limit = self.position_limit[product]
                buy_capacity = position_limit - pos
                sell_capacity = position_limit + pos
                
                # Position balancing - reduce size as position grows
                position_ratio = pos / position_limit
                position_balance_factor = max(0.2, 1.0 - abs(position_ratio))
                
                # Calculate notional exposure and limits
                current_notional = abs(pos * data['market_price'])
                notional_remaining = max(0, max_notional_per_product - current_notional)
                
                # Base size - scales with deviation magnitude
                # For strong signals, use larger sizes
                if abs(relative_deviation) > threshold * 2:
                    base_size = 150  # Maximum size for strong signals
                elif abs(relative_deviation) > threshold * 1.5:
                    base_size = 100  # Large size for clear signals
                elif abs(relative_deviation) > threshold:
                    base_size = 70  # Medium size for threshold signals
                else:
                    base_size = 30   # Small size for weak signals
                    
                # More granular size scaling based on deviation magnitude
                deviation_factor = min(3.0, 1.0 + (abs(relative_deviation) / threshold) * 2)
                base_size = int(min(base_size, position_limit * 0.15) * deviation_factor)
                
                # Scale size based on time to expiry - reduce size as expiry approaches
                # This reduces risk near expiration when volatility is less predictable
                tte_factor = min(1.0, self.time_to_maturity * 365 / 3)  # Scale down when less than 3 days left
                sized_scaled_by_time = int(base_size * tte_factor)
                
                # Apply position balance factor to avoid extreme concentration
                sized_with_balance = int(sized_scaled_by_time * position_balance_factor)
                    
                # Buy signals - Market IV lower than model (underpriced option)
                if relative_deviation < -threshold:
                    # Size position based on deviation, capacity, and risk limits
                    max_qty_by_notional = int(notional_remaining / data['market_price']) if data['market_price'] > 0 else buy_capacity
                    qty = min(sized_with_balance, buy_capacity, max_qty_by_notional)
                    
                    # Calculate price - slightly above best bid to improve execution
                    if data['best_bid'] is not None:
                        # Market making approach - place inside the spread
                        fair_bid = data['best_bid'] + 1
                    else:
                        # Use model price if no market orders
                        fair_bid = max(int(price_using_model_iv - spread/2), 1)
                    
                    # Add buy order if quantity is positive
                    if qty > 0:
                        orders.append(Order(product, fair_bid, qty))
                        print(f"BUY {product}: {qty} @ {fair_bid} (IV deviation: {relative_deviation:.4f})")
                    
                # Sell signals - Market IV higher than model (overpriced option)
                elif relative_deviation > threshold:
                    # Size position based on deviation, capacity, and risk limits
                    max_qty_by_notional = int(notional_remaining / data['market_price']) if data['market_price'] > 0 else sell_capacity
                    qty = min(sized_with_balance, sell_capacity, max_qty_by_notional)
                    
                    # Calculate price - slightly below best ask to improve execution
                    if data['best_ask'] is not None:
                        # Market making approach - place inside the spread
                        fair_ask = data['best_ask'] - 1
                    else:
                        # Use model price if no market orders
                        fair_ask = max(int(price_using_model_iv + spread/2), 1)
                    
                    # Add sell order if quantity is positive
                    if qty > 0:
                        orders.append(Order(product, fair_ask, -qty))
                        print(f"SELL {product}: {qty} @ {fair_ask} (IV deviation: {relative_deviation:.4f})")
                
                # Market making orders if no strong signal
                else:
                    # Only add market making orders if we don't have extreme positions
                    if abs(pos) < position_limit * 0.7:
                        # Calculate fair prices with tighter spread for market making
                        mm_spread = spread * 0.7
                        fair_bid = max(int(price_using_model_iv - mm_spread/2 - inventory_adjust), 1)
                        fair_ask = max(int(price_using_model_iv + mm_spread/2 - inventory_adjust), fair_bid + 1)
                        
                        # Smaller market making orders with position balancing
                        mm_size = int(30 * position_balance_factor)
                        
                        # Only place orders if we have capacity and size is meaningful
                        if buy_capacity >= mm_size and mm_size > 0:
                            orders.append(Order(product, fair_bid, mm_size))
                        if sell_capacity >= mm_size and mm_size > 0:
                            orders.append(Order(product, fair_ask, -mm_size))
                        
                        print(f"MM {product}: BID {mm_size} @ {fair_bid}, ASK {mm_size} @ {fair_ask}")
                
                # Update total delta for hedging
                total_delta += pos * delta
                result[product] = orders
                
            else:
                # Fallback strategy if we couldn't model the smile
                # Use standard Black-Scholes with our volatility estimate
                price, delta, gamma = black_scholes_call(
                    S, K, self.time_to_maturity, self.risk_free_rate, 
                    self.voucher_volatility, q=0)
                    
                spread = 2.0
                inventory_adjust = 0.02 * pos
                
                # Simple market making approach
                orders = []
                bid_capacity = self.position_limit[product] - pos
                ask_capacity = self.position_limit[product] + pos
                
                # Calculate prices
                fair_bid = max(int(price - spread/2 - inventory_adjust), 1)
                fair_ask = max(int(price + spread/2 - inventory_adjust), fair_bid + 1)
                
                # Use best market prices if available
                if 'best_bid' in data and 'best_ask' in data:
                    best_bid = data['best_bid']
                    best_ask = data['best_ask']
                    
                    # Only replace if market is more competitive
                    if best_bid and best_bid >= fair_bid:
                        fair_bid = int(best_bid)
                    if best_ask and best_ask <= fair_ask:
                        fair_ask = int(best_ask)
                
                # Position balancing - reduce size as position grows
                position_ratio = pos / self.position_limit[product]
                position_balance_factor = max(0.2, 1.0 - abs(position_ratio))
                
                # Use moderate size for fallback strategy with position balancing
                mm_size = int(min(40, min(bid_capacity, ask_capacity)) * position_balance_factor)
                
                # Add orders if we have capacity and size is meaningful
                if bid_capacity >= mm_size and mm_size > 0:
                    orders.append(Order(product, fair_bid, mm_size))
                if ask_capacity >= mm_size and mm_size > 0:
                    orders.append(Order(product, fair_ask, -mm_size))
                    
                # Update total delta for hedging
                total_delta += pos * delta
                result[product] = orders
                
                print(f"Fallback {product}: BID {mm_size} @ {fair_bid}, ASK {mm_size} @ {fair_ask}")
        
        # Delta hedging with underlying VOLCANIC_ROCK - more frequent with smaller sizes
        print(f"Total delta: {total_delta:.2f}, Current rock position: {rock_pos}")
        
        # Calculate target rock position for delta neutrality
        target_rock_pos = -int(total_delta)
        rock_trade = target_rock_pos - rock_pos
        
        # Implement hedging with lower threshold for more frequent adjustments
        if abs(rock_trade) > 50:  # Lower threshold for more frequent hedging
            # Cap position to avoid overexposure
            rock_trade = max(min(rock_trade, 200), -200)  # More conservative position limit
            
            # Get best market prices
            best_ask = min(state.order_depths.get(VOLCANIC_ROCK, {}).sell_orders.keys(), default=S+1)
            best_bid = max(state.order_depths.get(VOLCANIC_ROCK, {}).buy_orders.keys(), default=S-1)
            
            # Spread check to avoid trading at extreme prices
            spread = best_ask - best_bid
            if spread > 8:  # If spread is too wide, use mid price instead
                best_ask = best_bid = S
            
            # Buy VOLCANIC_ROCK to hedge short option delta
            if rock_trade > 0 and rock_bid_volume >= rock_trade:
                hedging_orders.append(Order(VOLCANIC_ROCK, int(best_ask), rock_trade))
                print(f"Hedge: Buy {rock_trade} VOLCANIC_ROCK at {best_ask}")
                
            # Sell VOLCANIC_ROCK to hedge long option delta
            elif rock_trade < 0 and rock_ask_volume <= rock_trade:
                hedging_orders.append(Order(VOLCANIC_ROCK, int(best_bid), rock_trade))
                print(f"Hedge: Sell {-rock_trade} VOLCANIC_ROCK at {best_bid}")
                
        result[VOLCANIC_ROCK] = hedging_orders
        return result


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Any, Any]:
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
            print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}")
        print(f"\tPnL {pnl}")

        self.below_110 = False
        result = {}

        

        # try:
        #     result[KELP] = self.kelp_strategy(state)
        # except Exception as e:
        #     print(f"Error in kelp_strategy: {type(e).__name__}: {str(e)}")

        

        # try:
        #     result[CROISSANTS] = self.croissants_strategy(state)
        # except Exception as e:
        #     print(f"Error in croissants_strategy: {type(e).__name__}: {str(e)}")

        # try:
        #     result[JAMS] = self.jams_strategy(state)
        # except Exception as e:
        #     print(f"Error in jams_strategy: {type(e).__name__}: {str(e)}")

        # try:
        #     result[DJEMBES] = self.djembes_strategy(state)
        # except Exception as e:
        #     print(f"Error in djembes_strategy: {type(e).__name__}: {str(e)}")


################################ strats that work         
        # try:
        #     result[RESIN] = self.resin_mm_strategy(state)
        # except Exception as e:
        #     print(f"Error in resin_strategy: {type(e).__name__}: {str(e)}")
            
            
        # try:
        #     result[SQUID_INK] = self.squid_ink_mm_strategy(state)
        # except Exception as e:
        #     print(f"Error in ink_strategy: {type(e).__name__}: {str(e)}")
        

        # try:
        #     result[PICNIC_BASKET1] = self.picnic_1_mm_strategy(state)
        # except Exception as e:
        #     print(f"Error in picnic_basket1_strategy: {type(e).__name__}: {str(e)}")

        # try:
        #     result[PICNIC_BASKET2] = self.picnic_2_mm_strategy(state)
        # except Exception as e:
        #     print(f"Error in picnic_basket2_strategy: {type(e).__name__}: {str(e)}")

        try:
            voucher_results = self.volcanic_rock_vouchers_strategy(state)
            result.update(voucher_results)
        except Exception as e:
            print(f"Error in volcanic_rock_vouchers_strategy: {type(e).__name__}: {str(e)}")

        conversions = 1
        return result, conversions, None
import math
import traceback
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState
import pandas as pd
import json
import jsonpickle

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
        
        # New volatility surface data structure
        self.vol_surface = {product: self.voucher_volatility for product in self.voucher_strikes}
        
        # Delta hedging corridor parameters
        self.hedge_lower_threshold = 0.8  # Only hedge if below 80% hedged
        self.hedge_upper_threshold = 1.2  # Only hedge if above 120% hedged
        self.max_hedge_size = 100  # Maximum size for hedge trades
        
        # Last hedged delta to track changes
        self.last_hedged_delta = 0
        self.last_volcanic_rock_position = 0

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
            sigma = self.vol_surface[product]  # Use strike-specific volatility
            price, _, _ = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, sigma, q=0)
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
                sigma = self.vol_surface[product]  # Use strike-specific volatility
                mid_price, _, _ = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, sigma, q=0)
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
        """Calculate volatility using EWMA model, which responds faster to recent changes."""
        if len(prices) < 2:
            return max(self.voucher_volatility, 0.5)
            
        # Calculate log returns
        log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        if not log_returns:
            return max(self.voucher_volatility, 0.1)
            
        # EWMA parameters
        lambda_param = 0.94  # Standard value for RiskMetrics
        
        # Initialize variance with square of first return
        variance = log_returns[0] ** 2
        
        # Update variance with EWMA
        for ret in log_returns[1:]:
            variance = lambda_param * variance + (1 - lambda_param) * (ret ** 2)
        
        # Convert to annualized volatility
        daily_vol = math.sqrt(variance)
        annualized_vol = daily_vol * math.sqrt(252)  # Assuming 252 trading days
        
        # Update the base volatility
        self.voucher_volatility = max(annualized_vol, 0.1)
        
        return self.voucher_volatility
  
    def calculate_volatility_surface(self, state: TradingState):
        """Create a volatility surface across different strikes."""
        S = self.get_mid_price(VOLCANIC_ROCK, state)
        base_vol = self.voucher_volatility
        
        # Dictionary to store strike-specific volatilities
        vol_surface = {}
        
        for product in self.voucher_strikes:
            K = self.voucher_strikes[product]
            # Calculate moneyness (K/S ratio)
            moneyness = K / S if S > 0 else 1.0
            
            # Implement volatility smile - higher volatility for strikes further from ATM
            # This is a simple quadratic formula; more sophisticated models exist
            vol_adjustment = 0.5 * (moneyness - 1.0) ** 2
            
            # Skew the smile slightly based on market trend
            if len(self.past_prices[VOLCANIC_ROCK]) > 20:
                recent_return = (self.past_prices[VOLCANIC_ROCK][-1] / 
                                self.past_prices[VOLCANIC_ROCK][-20] - 1)
                skew_factor = -0.1 * recent_return  # Negative correlation between returns and skew
            else:
                skew_factor = 0
            
            # Calculate final volatility for this strike
            strike_vol = base_vol * (1 + vol_adjustment + skew_factor)
            
            # Ensure reasonable bounds
            strike_vol = max(0.0001, min(strike_vol, 0.05))
            vol_surface[product] = strike_vol
        
        return vol_surface
        
    def calculate_dynamic_spread(self, product, state, price, gamma, pos):
        """Calculate optimal spread based on market conditions."""
        # Base spread component
        base_spread = 1.0
        
        # Volatility component - wider spreads in volatile markets
        if product in self.new_history and len(self.new_history[product]) > 30:
            recent_prices = self.new_history[product][-30:]
            recent_returns = [recent_prices[i]/recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
            vol_estimate = np.std(recent_returns) * 100  # Scale up for readability
            vol_component = 0.5 * vol_estimate
        else:
            vol_component = 0.2  # Default if not enough history
        
        # Gamma component - options with high gamma need wider spreads
        gamma_component = 0.3 * gamma
        
        # Inventory risk component - widen spread on the side we're exposed
        inventory_factor = 0.01 * abs(pos)
        inventory_component = inventory_factor * (1 if pos > 0 else -1)
        
        # Market depth component - check order book depth
        if product in state.order_depths:
            depth = state.order_depths[product]
            bid_depth = sum(abs(qty) for qty in depth.buy_orders.values())
            ask_depth = sum(abs(qty) for qty in depth.sell_orders.values())
            avg_depth = (bid_depth + ask_depth) / 2
            
            # Wider spreads in thin markets
            depth_component = max(0, 0.5 * (1 - avg_depth / 50))
        else:
            depth_component = 0.3  # Default if no order book
        
        # Calculate total spread
        total_spread = base_spread + vol_component + gamma_component + depth_component
        
        # Add inventory skew to shift midpoint
        bid_spread = total_spread - inventory_component
        ask_spread = total_spread + inventory_component
        
        return max(0.5, bid_spread), max(0.5, ask_spread)

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

    # def picnic_basket1_strategy(self, state: TradingState) -> List[Order]:
    #     product = PICNIC_BASKET1
    #     pos = self.get_position(product, state)
    #     mid = self.get_mid_price(product, state)
    #     crois_mid = self.get_mid_price(CROISSANTS, state)
    #     jams_mid = self.get_mid_price(JAMS, state)
    #     djembes_mid = self.get_mid_price(DJEMBES, state)
    #     fair_val = (6 * crois_mid + 3 * jams_mid + 1 * djembes_mid) / 10.0
    #     composite_mid = fair_val
    #     self.basket1_history.append(composite_mid)
    #     if len(self.basket1_history) < 50:
    #         basket_ema = sum(self.basket1_history) / len(self.basket1_history)
    #     else:
    #         alpha = 2 / (50 + 1)
    #         if self.basket1_ema is None:
    #             basket_ema = sum(self.basket1_history[-50:]) / 50
    #         else:
    #             basket_ema = alpha * composite_mid + (1 - alpha) * self.basket1_ema
    #         self.basket1_ema = basket_ema
    #     signal = 1 if composite_mid > basket_ema else -1
    #     delta = 1
    #     if signal == 1:
    #         bid_price = int(composite_mid - delta)
    #         ask_price = int(composite_mid + delta + 1)
    #     else:
    #         bid_price = int(composite_mid - delta - 1)
    #         ask_price = int(composite_mid + delta)
    #     bid_volume = self.position_limit[product] - pos
    #     ask_volume = -self.position_limit[product] - pos
    #     return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    # def picnic_basket2_strategy(self, state: TradingState) -> List[Order]:
    #     product = PICNIC_BASKET2
    #     pos = self.get_position(product, state)
    #     mid = self.get_mid_price(product, state)
    #     crois_mid = self.get_mid_price(CROISSANTS, state)
    #     jams_mid = self.get_mid_price(JAMS, state)
    #     fair_val = (4 * crois_mid + 2 * jams_mid) / 6.0
    #     delta = 1
    #     bid_price = int(min(mid, fair_val) - delta)
    #     ask_price = int(max(mid, fair_val) + delta)
    #     bid_volume = self.position_limit[product] - pos
    #     ask_volume = -self.position_limit[product] - pos
    #     return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    # def resin_strategy(self, state: TradingState) -> List[Order]:
    #     position_resin = self.get_position(RESIN, state)
    #     bid_volume = self.position_limit[RESIN] - position_resin
    #     ask_volume = -self.position_limit[RESIN] - position_resin
    #     orders = []
    #     order_ratio = self.get_order_ratio(RESIN, state)
    #     mid_price = self.get_mid_price(RESIN, state)
    #     best_ask = min(state.order_depths[RESIN].sell_orders.keys(), default=int(mid_price + 1))
    #     best_bid = max(state.order_depths[RESIN].buy_orders.keys(), default=int(mid_price - 1))
    #     best_ask_amount = state.order_depths[RESIN].sell_orders.get(best_ask, 0)
    #     best_bid_amount = state.order_depths[RESIN].buy_orders.get(best_bid, 0)
    #     if order_ratio > 0.3:
    #         orders.append(Order(RESIN, int(best_ask) - 1, bid_volume))  # Buy
    #     elif -1 <= order_ratio <= -0.3:
    #         orders.append(Order(RESIN, int(best_bid) + 1, ask_volume))  # Sell
    #     else:
    #         adjustment = round((DEFAULT_PRICES[RESIN] - mid_price) * 0.15)
    #         extra_adjustment_bid = 1 if position_resin < -5 else 0
    #         extra_adjustment_ask = -1 if position_resin > 5 else 0
    #         orders.append(Order(RESIN, min(DEFAULT_PRICES[RESIN] - 1, best_bid + adjustment + extra_adjustment_bid), bid_volume))
    #         orders.append(Order(RESIN, max(DEFAULT_PRICES[RESIN] + 1, best_ask + adjustment + extra_adjustment_ask), ask_volume))
    #     return orders

    # def ink_strategy(self, state: TradingState) -> List[Order]:
    #     mid_price = self.get_mid_price(SQUID_INK, state)
    #     self.ink_mid_prices.append(mid_price)
    #     perc_diff = 0
    #     if self.ink_prev_mid_price and self.ink_prev_mid_price != 0:
    #         perc_diff = (mid_price - self.ink_prev_mid_price) / self.ink_prev_mid_price
    #     self.ink_prev_mid_price = mid_price
    #     if len(self.ink_mid_prices) >= 20 and state.timestamp % 20 == 0:
    #         X = [(self.ink_mid_prices[i] - self.ink_mid_prices[i-1]) / self.ink_mid_prices[i-1]
    #             for i in range(1, len(self.ink_mid_prices)) if self.ink_mid_prices[i-1] != 0]
    #         if X:
    #             mu, theta, sigma = self.fit_ou(X)
    #             self.ou_params.update({"mu": mu, "theta": theta, "sigma": sigma})
    #             print(f"Updated OU params: mu={mu:.6f}, theta={theta:.4f}, sigma={sigma:.6f}")
    #     mu, theta, sigma = self.ou_params["mu"], self.ou_params["theta"], self.ou_params["sigma"]
    #     position_ink = self.get_position(SQUID_INK, state)
    #     z_score = (perc_diff - mu) / (sigma / np.sqrt(2 * theta)) if sigma > 0 else 0
    #     orders = []
    #     max_trade_size = 50
    #     bid_volume = self.position_limit[SQUID_INK] - position_ink
    #     ask_volume = -self.position_limit[SQUID_INK] - position_ink
    #     best_ask = min(state.order_depths[SQUID_INK].sell_orders.keys(), default=int(mid_price + 1))
    #     best_bid = max(state.order_depths[SQUID_INK].buy_orders.keys(), default=int(mid_price - 1))
    #     if perc_diff != 0:
    #         if z_score >= 4 and ask_volume > 0:
    #             size = min(max_trade_size, ask_volume, abs(state.order_depths[SQUID_INK].buy_orders.get(best_bid, 0)))
    #             if size > 0:
    #                 orders.append(Order(SQUID_INK, best_bid, -size))
    #         elif z_score <= -4 and bid_volume > 0:
    #             size = min(max_trade_size, bid_volume, abs(state.order_depths[SQUID_INK].sell_orders.get(best_ask, 0)))
    #             if size > 0:
    #                 orders.append(Order(SQUID_INK, best_ask, size))
    #         else:
    #             fair_price = mid_price * (1 + mu)
    #             bid_price = int(fair_price - 2)
    #             ask_price = int(fair_price + 2)
    #             orders.append(Order(SQUID_INK, bid_price, bid_volume))
    #             orders.append(Order(SQUID_INK, ask_price, ask_volume))
    #     print(f"z_score: {z_score:.2f}, perc_diff: {perc_diff:.6f}, orders: {[('Ink', o.price, o.quantity) for o in orders]}")
    #     return orders
import math
import jsonpickle
import numpy as np
from typing import Dict, List, Any, Tuple
from datamodel import Order, OrderDepth, TradingState, Trade, UserId

# Constants
SUBMISSION = "SUBMISSION"

# Products
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
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
    MAGNIFICENT_MACARONS,
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

# Basket components mapping 
BASKET1_COMPONENTS = {
    CROISSANTS: 3,
    JAMS: 2
}

BASKET2_COMPONENTS = {
    DJEMBES: 4,
    JAMS: 2
}

DEFAULT_PRICES = {
    MAGNIFICENT_MACARONS: 10000,
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

# Utility functions for statistical calculations
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

def compute_SMA(prices: List[float], window: int) -> float:
    if len(prices) < window:
        return sum(prices) / len(prices) if prices else 0
    return sum(prices[-window:]) / window

def compute_EMA(prices: List[float], window: int) -> float:
    if len(prices) == 0:
        return 0
    if len(prices) == 1:
        return prices[0]
    
    alpha = 2 / (window + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = price * alpha + ema * (1 - alpha)
    return ema

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

# Enhanced Black-Scholes option pricing model with Greeks
def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> Tuple[float, float, float, float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        price = max(S - K, 0)  # Intrinsic value of the call
        delta = 1 if S > K else 0  # Delta is 1 if in-the-money, else 0
        gamma = 0  # No curvature for expired or invalid options
        vega = 0   # No sensitivity to volatility
        theta = 0  # No time decay
        return price, delta, gamma, vega, theta

    try:
        # Calculate d1 and d2
        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T
        if sigma_sqrt_T == 0:
            price = max(S - K, 0)
            delta = 1 if S > K else 0
            gamma = 0
            vega = 0
            theta = 0
            return price, delta, gamma, vega, theta
            
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        # Calculate price and Greeks
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = math.exp(-q * T) * norm_cdf(d1)
        
        # Gamma calculation: protect against S * sigma * sqrt(T) being zero
        denominator = S * sigma_sqrt_T
        gamma = math.exp(-q * T) * norm_pdf(d1) / denominator if denominator != 0 else 0
        
        # Vega (sensitivity to volatility) - divided by 100 to get per 1% change
        vega = S * math.exp(-q * T) * sqrt_T * norm_pdf(d1) / 100
        
        # Theta (time decay) - divided by 365 to get daily decay
        part1 = -S * sigma * math.exp(-q * T) * norm_pdf(d1) / (2 * sqrt_T)
        part2 = -r * K * math.exp(-r * T) * norm_cdf(d2)
        part3 = q * S * math.exp(-q * T) * norm_cdf(d1)
        theta = (part1 + part2 + part3) / 365

        return price, delta, gamma, vega, theta

    except (ValueError, ZeroDivisionError, OverflowError) as e:
        # Handle numerical errors (e.g., log of zero, overflow)
        print(f"Error in Black-Scholes: {e}, S={S}, K={K}, T={T}, sigma={sigma}")
        price = max(S - K, 0)  # Fallback to intrinsic value
        delta = 1 if S > K else 0
        gamma = 0
        vega = 0
        theta = 0
        return price, delta, gamma, vega, theta


class Trader:
    def __init__(self) -> None:
        print("Initializing Trader...")
        
        # Position limits for all products
        self.position_limit = {
            MAGNIFICENT_MACARONS: 70,  # From macarons trader
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
        
        # General trading state
        self.round = 0
        self.cash = 0
        self.positions = {}
        self.past_prices = {product: [] for product in PRODUCTS}
        self.vols = {product: 0.01 for product in PRODUCTS}  # Store volatility estimates for each product
        self.price_history = {}
        self.position_history = []
        self.pnl_history = []
        self.last_pnl = 0
        self.best_pnl = 0
        self.new_history = {}
        
        # Market state tracking
        self.ema_prices = {product: None for product in PRODUCTS}
        self.ema_param = 0.5
        
        # Adaptive parameters for market making
        self.spread_multipliers = {product: 1.0 for product in PRODUCTS}  # Adjusts based on volatility
        self.vol_window = {product: [] for product in PRODUCTS}  # Tracks recent volatility
        self.market_liquidity = {product: 1.0 for product in PRODUCTS}  # Tracks liquidity depth
        
        # Ornstein-Uhlenbeck parameters for mean-reverting assets
        self.ou_params = {"mu": -0.000013, "theta": 0.01, "sigma": 0.000025}
        self.kelp_ou_params = {"mu": 0.000006, "theta": 0.01, "sigma": 0.000713}
        
        # Product-specific state (from macarons trader)
        self.market_phase = "discovery"
        self.phase_counter = 0
        self.trades_this_round = 0
        self.last_mid = 0
        self.preferred_position = -40  # Maintain a net short bias for MAGNIFICENT_MACARONS
        
        # Product-specific price tracking
        self.ink_prev_mid_price = None
        self.kelp_prev_mid_price = None
        self.ink_mid_prices = []
        self.kelp_mid_prices = []
        self.basket1_history = []
        self.basket1_ema = None
        
        # Basket fair value tracking
        self.basket1_fair_value = None
        self.basket2_fair_value = None
        self.basket_arbitrage_history = {
            PICNIC_BASKET1: [],
            PICNIC_BASKET2: []
        }
        
        # Voucher-specific state for options trading
        self.voucher_strikes = {
            VOLCANIC_ROCK_VOUCHER_9500: 9500,
            VOLCANIC_ROCK_VOUCHER_9750: 9750,
            VOLCANIC_ROCK_VOUCHER_10000: 10000,
            VOLCANIC_ROCK_VOUCHER_10250: 10250,
            VOLCANIC_ROCK_VOUCHER_10500: 10500
        }
        self.voucher_volatility = 0.0009450871502416238
        self.volatility_surface = {}  # Store volatility by strike
        self.realized_vols = []  # Track realized volatility
        self.risk_free_rate = 0.0435
        self.time_to_maturity = 7 / 365  # 7 days from round 1
        self.base_spread = 0.25  # Base spread for vouchers
        self.voucher_gamma_exposure = 0  # Track total gamma exposure
        self.voucher_vega_exposure = 0   # Track total vega exposure
        
        # Performance tracking
        self.product_pnl = {product: 0 for product in PRODUCTS}
        self.trade_counts = {product: 0 for product in PRODUCTS}

    def run(self, state: TradingState):
        """Main entry point for the trading algorithm"""
        # Initialize or restore state
        if state.traderData:
            try:
                trader_state = jsonpickle.decode(state.traderData)
                self.__dict__.update(trader_state)
            except Exception as e:
                print(f"Error restoring state: {e}")
        
        # Update positions from current state
        for product, pos in state.position.items():
            self.positions[product] = pos
        
        # Increment round count and calculate pnl
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)
        self.update_volatilities(state)
        
        # Update time to maturity for options
        if self.round > 1:
            self.time_to_maturity = max(0, self.time_to_maturity - (1/365))
        
        # Debug output
        print(f"Log round {self.round}")
        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)
        print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            if product in state.order_depths:
                print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}")
        print(f"\tPnL {pnl}")
        
        # Initialize result container
        result = {}
        
        # Update basket fair values
        self.update_basket_fair_values(state)
        
        # Process each product with its respective strategy if it exists in order_depths
        for product in state.order_depths:
            if product == MAGNIFICENT_MACARONS:
                self.trades_this_round = 0
                orders = self.trade_macarons(
                    product,
                    state.order_depths[product],
                    self.positions.get(product, 0)
                )
                result[product] = orders
            
            elif product == RESIN:
                try:
                    result[product] = self.resin_mm_strategy(state)
                except Exception as e:
                    print(f"Error in resin_strategy: {type(e).__name__}: {str(e)}")
            
            elif product == SQUID_INK:
                try:
                    result[product] = self.squid_ink_mm_strategy(state)
                except Exception as e:
                    print(f"Error in ink_strategy: {type(e).__name__}: {str(e)}")
            
            elif product == PICNIC_BASKET1:
                try:
                    result[product] = self.enhanced_basket1_strategy(state)
                except Exception as e:
                    print(f"Error in picnic_basket1_strategy: {type(e).__name__}: {str(e)}")
            
            elif product == PICNIC_BASKET2:
                try:
                    result[product] = self.enhanced_basket2_strategy(state)
                except Exception as e:
                    print(f"Error in picnic_basket2_strategy: {type(e).__name__}: {str(e)}")
        
        # Process volcanic rock vouchers separately since they're interdependent
        try:
            if VOLCANIC_ROCK in state.order_depths or any(product in state.order_depths for product in self.voucher_strikes):
                voucher_results = self.enhanced_volcanic_rock_vouchers_strategy(state)
                result.update(voucher_results)
        except Exception as e:
            print(f"Error in volcanic_rock_vouchers_strategy: {type(e).__name__}: {str(e)}")
            
        # Process other products if needed
        try:
            if CROISSANTS in state.order_depths:
                result[CROISSANTS] = self.enhanced_croissants_strategy(state)
        except Exception as e:
            print(f"Error in croissants_strategy: {type(e).__name__}: {str(e)}")
            
        try:
            if JAMS in state.order_depths:
                result[JAMS] = self.enhanced_jams_strategy(state)
        except Exception as e:
            print(f"Error in jams_strategy: {type(e).__name__}: {str(e)}")
            
        try:
            if DJEMBES in state.order_depths:
                result[DJEMBES] = self.enhanced_djembes_strategy(state)
        except Exception as e:
            print(f"Error in djembes_strategy: {type(e).__name__}: {str(e)}")
            
        try:
            if KELP in state.order_depths:
                result[KELP] = self.enhanced_kelp_strategy(state)
        except Exception as e:
            print(f"Error in kelp_strategy: {type(e).__name__}: {str(e)}")
        
        # Determine if conversion is needed
        conversion = 0
        for product in state.position:
            product_conversion = self.determine_conversion(product, state.position.get(product, 0))
            if product_conversion > 0:
                conversion = product_conversion
                break
        
        # Save state
        trader_data = jsonpickle.encode(self.__dict__)
        
        return result, conversion, trader_data
        
    # Common utility methods
    def get_position(self, product, state: TradingState) -> int:
        """Get current position for a product"""
        return state.position.get(product, 0)
    
    def get_order_ratio(self, product, state: TradingState) -> float:
        """Calculate order imbalance ratio"""
        if product not in state.order_depths:
            return 0
            
        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()
        if len(market_asks) > 0 and len(market_bids) > 0:
            return (sum(market_bids) - sum(market_asks)) / (sum(market_bids) + sum(market_asks))
        return 0

    def get_mid_price(self, product, state: TradingState) -> float:
        """Calculate mid price for a product"""
        default_price = self.ema_prices.get(product)
        if default_price is None:
            default_price = DEFAULT_PRICES.get(product, 10000)

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
        """Calculate value of position in a product"""
        if not product:
            return 0
            
        if product in self.voucher_strikes:
            S = self.get_mid_price(VOLCANIC_ROCK, state)
            K = self.voucher_strikes[product]
            price, _, _, _, _ = black_scholes_greeks(S, K, self.time_to_maturity, self.risk_free_rate, self.voucher_volatility, q=0)
            return self.get_position(product, state) * price
        return self.get_position(product, state) * self.get_mid_price(product, state)
    
    def update_pnl(self, state: TradingState) -> float:
        """Update cash balance and calculate total PnL"""
        def update_cash():
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        continue
                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                        # Track product-specific PnL
                        if product not in self.product_pnl:
                            self.product_pnl[product] = 0
                        self.product_pnl[product] -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price
                        if product not in self.product_pnl:
                            self.product_pnl[product] = 0
                        self.product_pnl[product] += trade.quantity * trade.price
                    
                    # Increment trade count
                    if product not in self.trade_counts:
                        self.trade_counts[product] = 0
                    self.trade_counts[product] += 1

        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value

        update_cash()
        return self.cash + get_value_on_positions()
    
    def update_ema_prices(self, state: TradingState):
        """Update exponential moving average prices for all products"""
        for product in PRODUCTS:
            if product not in state.order_depths:
                continue
                
            mid_price = self.get_mid_price(product, state)
            if product in self.voucher_strikes:
                S = self.get_mid_price(VOLCANIC_ROCK, state)
                K = self.voucher_strikes[product]
                mid_price, _, _, _, _ = black_scholes_greeks(S, K, self.time_to_maturity, self.risk_free_rate, self.voucher_volatility, q=0)
                
            if product not in self.ema_prices or self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1 - self.ema_param) * self.ema_prices[product]
                
            # Store historical prices for volatility calculation
            if product not in self.past_prices:
                self.past_prices[product] = []
            self.past_prices[product].append(mid_price)
            # Keep only recent history
            if len(self.past_prices[product]) > 50:
                self.past_prices[product] = self.past_prices[product][-50:]
    
    def update_volatilities(self, state: TradingState):
        """Update volatility estimates for all products"""
        for product in PRODUCTS:
            if product not in self.past_prices or len(self.past_prices[product]) < 3:
                continue
                
            prices = self.past_prices[product]
            # Calculate returns
            returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
            if returns:
                # Calculate volatility as standard deviation of returns
                vol = np.std(returns) * math.sqrt(252)  # Annualized
                self.vols[product] = vol
                
                # For market making, adjust spread multipliers based on volatility
                normalized_vol = vol / 0.01  # Normalize against a baseline volatility of 1%
                self.spread_multipliers[product] = max(0.5, min(3.0, normalized_vol))
                
                # Record for voucher volatility surface
                if product == VOLCANIC_ROCK:
                    self.realized_vols.append(vol)
                    if len(self.realized_vols) > 10:
                        self.realized_vols = self.realized_vols[-10:]
    
    def update_basket_fair_values(self, state: TradingState):
        """Calculate fair values for basket products based on components"""
        # For Basket 1: Croissants and Jams
        croissants_price = self.get_mid_price(CROISSANTS, state)
        jams_price = self.get_mid_price(JAMS, state)
        
        if croissants_price and jams_price:
            # Calculate theoretical fair value (3 croissants + 2 jams)
            fair_value1 = (3 * croissants_price + 2 * jams_price)
            self.basket1_fair_value = fair_value1
            
            # Calculate actual price
            actual_price1 = self.get_mid_price(PICNIC_BASKET1, state)
            
            # Record arbitrage opportunity
            if actual_price1:
                arb_spread = actual_price1 - fair_value1
                self.basket_arbitrage_history[PICNIC_BASKET1].append(arb_spread)
                if len(self.basket_arbitrage_history[PICNIC_BASKET1]) > 20:
                    self.basket_arbitrage_history[PICNIC_BASKET1] = self.basket_arbitrage_history[PICNIC_BASKET1][-20:]
        
        # For Basket 2: Djembes and Jams
        djembes_price = self.get_mid_price(DJEMBES, state)
        
        if djembes_price and jams_price:
            # Calculate theoretical fair value (4 djembes + 2 jams)
            fair_value2 = (4 * djembes_price + 2 * jams_price)
            self.basket2_fair_value = fair_value2
            
            # Calculate actual price
            actual_price2 = self.get_mid_price(PICNIC_BASKET2, state)
            
            # Record arbitrage opportunity
            if actual_price2:
                arb_spread = actual_price2 - fair_value2
                self.basket_arbitrage_history[PICNIC_BASKET2].append(arb_spread)
                if len(self.basket_arbitrage_history[PICNIC_BASKET2]) > 20:
                    self.basket_arbitrage_history[PICNIC_BASKET2] = self.basket_arbitrage_history[PICNIC_BASKET2][-20:]
    
    def get_best_bid(self, order_depth):
        """Get highest bid price and quantity"""
        if not order_depth.buy_orders:
            return None, None
        best_bid = max(order_depth.buy_orders.keys())
        return best_bid, order_depth.buy_orders[best_bid]
    
    def get_best_ask(self, order_depth):
        """Get lowest ask price and quantity"""
        if not order_depth.sell_orders:
            return None, None
        best_ask = min(order_depth.sell_orders.keys())
        return best_ask, order_depth.sell_orders[best_ask]
    
    def determine_conversion(self, product, position):
        """Determine if conversion is needed"""
        if product == MAGNIFICENT_MACARONS:
            # If position is approaching limits, convert to free up capacity
            if position <= -65:  # More conservative threshold
                return 15  # Convert more short to buy and free up capacity
            elif position >= 65:  # More conservative threshold
                return 15  # Convert more long to free up capacity
                
            # If we have a moderate position but want to increase profits in the market phase
            if self.market_phase == "aggressive_short" and position <= -50:
                return 15  # Convert more shorts to establish new ones
                
            # If we have a position opposite to our strategy
            if self.market_phase in ["aggressive_short", "opportunistic"] and position >= 10:
                return 15  # Convert more long position (against our strategy)
        
        # For picnic baskets, convert if they're significantly mispriced
        if product == PICNIC_BASKET1 and abs(position) > 40:
            return 10  # Convert to potentially arbitrage with components
            
        if product == PICNIC_BASKET2 and abs(position) > 75:
            return 10  # Convert to potentially arbitrage with components
        
        return 0
            
    def calculate_trend(self, product):
        """Calculate price trend (negative = downtrend)"""
        prices = self.price_history.get(product, [])
        
        if len(prices) < 5:
            return 0  # Not enough data
            
        # Calculate short-term trend (last 5 prices)
        recent_prices = prices[-5:]
        if recent_prices[0] == 0:  # Avoid division by zero
            return 0
            
        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
    def update_volatility(self, prices: List[float]) -> float:
        """Update volatility estimate based on price history"""
        if len(prices) < 2:
            return max(self.voucher_volatility, 0.5)
        log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        if log_returns:
            volatility = max(np.std(log_returns) * math.sqrt(252), 0.1)
            self.voucher_volatility = volatility * 0.2 + self.voucher_volatility * 0.8  # Smooth volatility updates
        return self.voucher_volatility
        
    def implied_volatility(self, S, K, T, r, option_price, q=0) -> float:
        """Calculate implied volatility using numerical methods"""
        sigma = 0.3
        for _ in range(50):
            price, _, _, _, _ = black_scholes_greeks(S, K, T, r, sigma, q)
        
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
    
    def calculate_volatility_smile(self, state):
        """Calculate volatility smile across strikes"""
        if VOLCANIC_ROCK not in state.order_depths:
            return
            
        S = self.get_mid_price(VOLCANIC_ROCK, state)
        base_vol = self.voucher_volatility
        
        # Update volatility surface for each strike
        for product in self.voucher_strikes:
            if product not in state.order_depths:
                continue
                
            K = self.voucher_strikes[product]
            moneyness = K / S
            
            # Find market price if available
            mid_price = self.get_mid_price(product, state)
            
            if not mid_price:
                # Theoretical volatility smile: higher vol for out-of-the-money options
                vol_adjustment = 0.0001 * abs(moneyness - 1.0) * 100
                self.volatility_surface[K] = base_vol + vol_adjustment
            else:
                # Calculate implied volatility from market price
                iv = self.implied_volatility(S, K, self.time_to_maturity, self.risk_free_rate, mid_price)
                if iv > 0:
                    # Blend with existing estimate for stability
                    if K in self.volatility_surface:
                        self.volatility_surface[K] = 0.3 * iv + 0.7 * self.volatility_surface[K]
                    else:
                        self.volatility_surface[K] = iv
    
    def calculate_market_depth(self, product, state: TradingState):
        """Calculate market depth to adjust trade sizes"""
        if product not in state.order_depths:
            return 1.0
            
        order_depth = state.order_depths[product]
        
        total_bid_quantity = sum(abs(qty) for qty in order_depth.buy_orders.values())
        total_ask_quantity = sum(abs(qty) for qty in order_depth.sell_orders.values())
        
        # Calculate average depth
        avg_depth = (total_bid_quantity + total_ask_quantity) / 2
        
        # Normalize against expected baseline
        expected_depth = 50  # Example baseline
        
        # Return normalized depth (minimum 0.2, maximum 2.0)
        return max(0.2, min(2.0, avg_depth / expected_depth))
    
    def fit_ou(self, X: List[float]) -> Tuple[float, float, float]:
        """Fit Ornstein-Uhlenbeck parameters to price changes"""
        if len(X) < 20:
            return self.ou_params["mu"], self.ou_params["theta"], self.ou_params["sigma"]
        X = np.array(X)
        n = len(X)
        dt = 1.0
        mu = X.mean()
        X_t = X[:-1]
        X_t1 = X[1:]
        cov = np.cov(X_t, X_t1)[0, 1]
        var = np.var(X_t)
        theta = -np.log(cov / var) / dt if var > 0 else 0.1
        theta = max(0.01, min(theta, 1.0))
        drift = mu + (X_t - mu) * np.exp(-theta * dt)
        residuals = X_t1 - drift
        sigma = np.sqrt(2 * theta * np.var(residuals) / (1 - np.exp(-2 * theta * dt)))
        sigma = max(1e-6, sigma)
        return mu, theta, sigma
    
    def enhanced_jams_strategy(self, state: TradingState) -> List[Order]:
        """Enhanced Jams strategy using stochastic oscillator with trend confirmation"""
        product = JAMS
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        
        # Position management parameters
        position_limit = self.position_limit[product]
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        # Initialize price history if needed
        if product not in self.new_history:
            self.new_history[product] = []
        self.new_history[product].append(mid)
        
        prices = self.new_history[product]
        
        # Calculate technical indicators
        stoch = compute_stochastic(prices, 14)  # Fast stochastic
        
        # Calculate slow stochastic (3-period SMA of fast stochastic)
        stoch_history = []
        for i in range(min(14, len(prices))):
            stoch_subset = prices[-(i+1):]
            stoch_history.append(compute_stochastic(stoch_subset, min(14, len(stoch_subset))))
            
        slow_stoch = compute_SMA(stoch_history, 3) if len(stoch_history) >= 3 else stoch
        
        # Calculate EMA for trend confirmation
        ema20 = compute_EMA(prices, 20) if len(prices) >= 10 else mid
        
        # Calculate momentum for trend strength
        momentum = (prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))] if len(prices) > 1 else 0
        
        # Custom signal generation for Jams
        signal = 0
        
        # 1. Stochastic crossover signals
        if stoch < 20 and slow_stoch < 20:
            signal += 2  # Strong oversold
        elif stoch > 80 and slow_stoch > 80:
            signal -= 2  # Strong overbought
            
        # 2. Stochastic reversal signals
        if stoch < 20 and stoch > slow_stoch:
            signal += 1  # Potential bullish reversal
        elif stoch > 80 and stoch < slow_stoch:
            signal -= 1  # Potential bearish reversal
            
        # 3. Trend confirmation
        if mid > ema20 and signal > 0:
            signal += 0.5  # Uptrend confirms buy signal
        elif mid < ema20 and signal < 0:
            signal -= 0.5  # Downtrend confirms sell signal
            
        # 4. Momentum confirmation
        if momentum > 0.01 and signal > 0:
            signal += 0.5  # Strong momentum confirms buy
        elif momentum < -0.01 and signal < 0:
            signal -= 0.5  # Strong momentum confirms sell
            
        # Calculate dynamic pricing and sizing
        # Base prices
        base_bid = int(mid - 1)
        base_ask = int(mid + 1)
        
        # Adjust prices based on signal
        price_adjustment = int(signal * 0.5)
        bid_price = base_bid + price_adjustment
        ask_price = base_ask + price_adjustment
        
        # Basic size
        base_size = 25
        
        # Scale based on signal strength
        signal_factor = min(abs(signal), 3) / 2  # Scale from 0 to 1.5
        
        # Position impact
        position_factor = abs(pos / position_limit) if position_limit > 0 else 0
        
        # Calculate final sizes
        if signal > 0:  # Buy signal
            bid_size = int(base_size * (1 + signal_factor) * (1 - position_factor * 0.7))
            ask_size = int(base_size * 0.7)  # Reduced sell size
        elif signal < 0:  # Sell signal
            bid_size = int(base_size * 0.7)  # Reduced buy size
            ask_size = int(base_size * (1 + signal_factor) * (1 - position_factor * 0.7))
        else:  # Neutral
            bid_size = int(base_size * (1 - position_factor * 0.5))
            ask_size = int(base_size * (1 - position_factor * 0.5))
            
        # Ensure within capacity
        bid_size = min(bid_size, buy_capacity)
        ask_size = min(ask_size, sell_capacity)
        
        # Generate orders
        orders = []
        
        # Add limit orders
        if buy_capacity > 0:
            orders.append(Order(product, bid_price, bid_size))
        
        if sell_capacity > 0:
            orders.append(Order(product, ask_price, -ask_size))
        
        # Add aggressive orders for very strong signals
        if signal > 2.5 and buy_capacity > bid_size:
            # Very strong buy - add market order
            if product in state.order_depths and state.order_depths[product].sell_orders:
                best_ask = min(state.order_depths[product].sell_orders.keys())
                ask_volume = abs(state.order_depths[product].sell_orders[best_ask])
                market_buy_size = min(int(base_size * 0.7), buy_capacity - bid_size, ask_volume)
                if market_buy_size > 0:
                    orders.append(Order(product, best_ask, market_buy_size))
                    print(f"{product} MARKET BUY: {market_buy_size} @ {best_ask} due to strong signal {signal:.1f}")
                    
        elif signal < -2.5 and sell_capacity > ask_size:
            # Very strong sell - add market order
            if product in state.order_depths and state.order_depths[product].buy_orders:
                best_bid = max(state.order_depths[product].buy_orders.keys())
                bid_volume = state.order_depths[product].buy_orders[best_bid]
                market_sell_size = min(int(base_size * 0.7), sell_capacity - ask_size, bid_volume)
                if market_sell_size > 0:
                    orders.append(Order(product, best_bid, -market_sell_size))
                    print(f"{product} MARKET SELL: {market_sell_size} @ {best_bid} due to strong signal {signal:.1f}")
        
        # Debug info
        print(f"{product}: pos={pos}, mid={mid:.1f}, stoch={stoch:.1f}, slow={slow_stoch:.1f}, "
              f"signal={signal:.1f}, orders={[(o.price, o.quantity) for o in orders]}")
              
        return orders
    
    def enhanced_croissants_strategy(self, state: TradingState) -> List[Order]:
        """Enhanced Croissants strategy using multiple technical indicators with adaptive parameters"""
        product = CROISSANTS
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        
        # Position management parameters
        position_limit = self.position_limit[product]
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        # Initialize price history if needed
        if product not in self.new_history:
            self.new_history[product] = []
        self.new_history[product].append(mid)
        
        # Calculate technical indicators
        prices = self.new_history[product]
        
        # Lookback windows
        short_window = 5
        medium_window = 14
        long_window = 20
        
        # Calculate moving averages
        short_sma = compute_SMA(prices, short_window) if len(prices) >= short_window else mid
        medium_sma = compute_SMA(prices, medium_window) if len(prices) >= medium_window else mid
        long_sma = compute_SMA(prices, long_window) if len(prices) >= long_window else mid
        
        # Calculate exponential moving average
        ema = compute_EMA(prices, medium_window)
        
        # Calculate price volatility
        std = compute_STD(prices, medium_window) if len(prices) >= medium_window else 0
        
        # Calculate Bollinger Bands
        upper_band = medium_sma + 2 * std
        lower_band = medium_sma - 2 * std
        
        # Calculate RSI
        rsi = compute_RSI(prices, 14)
        
        # Momentum indicators
        momentum = (prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))] if len(prices) > 1 else 0
        
        # Determine buy/sell signals based on multiple indicators
        buy_signals = 0
        sell_signals = 0
        
        # 1. Bollinger Band signals
        if mid <= lower_band:
            buy_signals += 2  # Strong buy signal at lower band
        elif mid >= upper_band:
            sell_signals += 2  # Strong sell signal at upper band
            
        # 2. RSI signals
        if rsi < 30:
            buy_signals += 1  # Oversold
        elif rsi > 70:
            sell_signals += 1  # Overbought
            
        # 3. Moving average crossover signals
        if short_sma > medium_sma and medium_sma > long_sma:
            buy_signals += 1  # Uptrend
        elif short_sma < medium_sma and medium_sma < long_sma:
            sell_signals += 1  # Downtrend
            
        # 4. Price relative to moving averages
        if mid > long_sma:
            buy_signals += 0.5  # Above long-term average
        elif mid < long_sma:
            sell_signals += 0.5  # Below long-term average
            
        # 5. Momentum signals
        if momentum > 0.01:
            buy_signals += 0.5  # Positive momentum
        elif momentum < -0.01:
            sell_signals += 0.5  # Negative momentum
            
        # Calculate net signal
        net_signal = buy_signals - sell_signals
        
        # Determine order sizes based on signal strength
        base_size = 20  # Base order size
        
        # Scale size by signal strength (stronger signals = larger positions)
        signal_factor = min(abs(net_signal) / 2, 1.5)  # Cap at 150% of base size
        
        # Dynamic pricing based on signal strength and direction
        bid_base = int(mid - 1)  # Base bid price
        ask_base = int(mid + 1)  # Base ask price
        
        # Adjust prices based on signal direction
        signal_price_adj = int(net_signal)
        bid_price = bid_base + signal_price_adj
        ask_price = ask_base + signal_price_adj
        
        # Factor in position to size
        position_factor = abs(pos / position_limit) if position_limit > 0 else 0
        
        # Reduce buying when long, reduce selling when short
        if net_signal > 0:  # Buy signal
            bid_size = int(base_size * signal_factor * (1 - position_factor * 0.7))
            ask_size = int(base_size * 0.5)  # Smaller sell orders during buy signals
        elif net_signal < 0:  # Sell signal
            bid_size = int(base_size * 0.5)  # Smaller buy orders during sell signals
            ask_size = int(base_size * signal_factor * (1 - position_factor * 0.7))
        else:  # Neutral signal
            bid_size = int(base_size * (1 - position_factor * 0.5))
            ask_size = int(base_size * (1 - position_factor * 0.5))
        
        # Ensure sizes respect capacity limits
        bid_size = min(bid_size, buy_capacity)
        ask_size = min(ask_size, sell_capacity)
        
        # Generate orders
        orders = []
        
        # Add limit orders based on strategy
        if buy_capacity > 0:
            orders.append(Order(product, bid_price, bid_size))
        
        if sell_capacity > 0:
            orders.append(Order(product, ask_price, -ask_size))
        
        # Add aggressive orders for strong signals
        if net_signal > 2 and buy_capacity > bid_size:
            # Strong buy signal - add market order
            if product in state.order_depths and state.order_depths[product].sell_orders:
                best_ask = min(state.order_depths[product].sell_orders.keys())
                ask_volume = abs(state.order_depths[product].sell_orders[best_ask])
                market_buy_size = min(int(base_size * 0.5), buy_capacity - bid_size, ask_volume)
                if market_buy_size > 0:
                    orders.append(Order(product, best_ask, market_buy_size))
                    print(f"{product} MARKET BUY: {market_buy_size} @ {best_ask} due to strong signal {net_signal:.1f}")
                    
        elif net_signal < -2 and sell_capacity > ask_size:
            # Strong sell signal - add market order
            if product in state.order_depths and state.order_depths[product].buy_orders:
                best_bid = max(state.order_depths[product].buy_orders.keys())
                bid_volume = state.order_depths[product].buy_orders[best_bid]
                market_sell_size = min(int(base_size * 0.5), sell_capacity - ask_size, bid_volume)
                if market_sell_size > 0:
                    orders.append(Order(product, best_bid, -market_sell_size))
                    print(f"{product} MARKET SELL: {market_sell_size} @ {best_bid} due to strong signal {net_signal:.1f}")
        
        # Debug info
        print(f"{product}: pos={pos}, mid={mid:.1f}, rsi={rsi:.1f}, bands=[{lower_band:.1f},{upper_band:.1f}], "
              f"signal={net_signal:.1f}, orders={[(o.price, o.quantity) for o in orders]}")
              
        return orders
    
    def trade_macarons(self, product, order_depth, position):
        """Main trading strategy for MAGNIFICENT_MACARONS"""
        # Initialize orders list
        orders = []
        
        # Get best prices from order book
        best_bid, best_bid_amount = self.get_best_bid(order_depth)
        best_ask, best_ask_amount = self.get_best_ask(order_depth)
        
        # If market is one-sided, return empty orders
        if not best_bid or not best_ask:
            return orders
        
        # Calculate mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Initialize price history if needed
        if product not in self.price_history:
            self.price_history[product] = []
        
        # Record price
        self.price_history[product].append(mid_price)
        
        # Keep only the most recent 20 prices
        if len(self.price_history[product]) > 20:
            self.price_history[product] = self.price_history[product][-20:]
        
        # Calculate price trend (negative value = downtrend)
        price_trend = self.calculate_trend(product)
        
        # Record position
        self.position_history.append(position)
        
        # Keep only most recent 10 positions
        if len(self.position_history) > 10:
            self.position_history = self.position_history[-10:]
        
        # Update market phase
        self.update_market_phase(price_trend, position)
        
        # CORE TRADING STRATEGY
        if self.market_phase == "discovery":
            # In discovery phase, take small positions to test the market
            orders = self.execute_discovery_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount)
        
        elif self.market_phase == "aggressive_short":
            # In aggressive shorting phase, build significant short position
            orders = self.execute_aggressive_short_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount)
        
        elif self.market_phase == "defensive":
            # In defensive phase, reduce exposure and protect capital
            orders = self.execute_defensive_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount)
        
        elif self.market_phase == "opportunistic":
            # In opportunistic phase, look for quick profits in both directions
            orders = self.execute_opportunistic_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, price_trend)
        
        # Safety mechanism - verify we're not exceeding position limits
        self.trades_this_round += len(orders)
        self.last_mid = mid_price
        
        return orders
    
    def update_market_phase(self, price_trend, position):
        """Update market phase based on current conditions"""
        # Increment phase counter
        self.phase_counter += 1
        
        # If we're just starting or have only been in discovery for a short time
        if self.market_phase == "discovery" and self.phase_counter < 5:
            return  # Stay in discovery phase
            
        # Transition from discovery phase
        if self.market_phase == "discovery" and self.phase_counter >= 5:
            if price_trend < -0.001:  # Downtrend detected
                self.market_phase = "aggressive_short"
                self.phase_counter = 0
                print("PHASE CHANGE: Discovery -> Aggressive Short (downtrend detected)")
            else:
                self.market_phase = "opportunistic"
                self.phase_counter = 0
                print("PHASE CHANGE: Discovery -> Opportunistic (no clear downtrend)")
            return
            
        # Transitions from aggressive short phase
        if self.market_phase == "aggressive_short":
            if position <= -50:  # If we've built up a significant short position
                self.market_phase = "defensive"
                self.phase_counter = 0
                print("PHASE CHANGE: Aggressive Short -> Defensive (large short position established)")
            elif price_trend > 0.002:  # If market trend reverses
                self.market_phase = "defensive"
                self.phase_counter = 0
                print("PHASE CHANGE: Aggressive Short -> Defensive (uptrend detected)")
            return
            
        # Transitions from defensive phase
        if self.market_phase == "defensive" and self.phase_counter >= 3:
            self.market_phase = "opportunistic"
            self.phase_counter = 0
            print("PHASE CHANGE: Defensive -> Opportunistic (defense period complete)")
            return
            
        # Transitions from opportunistic phase
        if self.market_phase == "opportunistic":
            if price_trend < -0.002:  # Strong downtrend detected
                self.market_phase = "aggressive_short"
                self.phase_counter = 0
                print("PHASE CHANGE: Opportunistic -> Aggressive Short (downtrend detected)")
            return
    
    def execute_discovery_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount):
        """Initial discovery phase - take small positions to test market direction"""
        orders = []
        
        # If we have a long position, try to reduce it
        if position > 0:
            # Sell at best bid to reduce long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DISCOVERY: Selling {sell_amount} @ {best_bid} to reduce long position")
        
        # If position is neutral or only slightly short, build moderate short position
        if position > -10:
            # Sell at bid to establish short position
            sell_amount = min(best_bid_amount, 20 + position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DISCOVERY: Selling {sell_amount} @ {best_bid} to establish short position")
        
        return orders
    
    def execute_aggressive_short_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount):
        """Aggressively build short position when market is declining"""
        orders = []
        
        # If we have a long position, exit immediately
        if position > 0:
            # Sell at best bid to exit long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"AGGRESSIVE SHORT: Selling {sell_amount} @ {best_bid} to exit long position")
                return orders  # Exit after handling long position
        
        # If we have room to add more short position
        if position > -self.position_limit.get(product, 70):
            target_short = min(self.position_limit.get(product, 70), 50)  # Target 50 short or max position
            
            if position > -target_short:
                # Calculate how much to sell
                sell_amount = min(best_bid_amount, target_short + position)
                if sell_amount > 0:
                    orders.append(Order(product, best_bid, -sell_amount))
                    print(f"AGGRESSIVE SHORT: Selling {sell_amount} @ {best_bid} (Current: {position}, Target: {-target_short})")
        
        return orders
    
    def execute_defensive_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount):
        """Defensive strategy - reduce exposure and protect capital"""
        orders = []
        
        # If we're too heavily short, reduce position
        if position < -50:
            # Buy at best ask to reduce short position
            buy_amount = min(-best_ask_amount, -position - 40)  # Target -40 position
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                print(f"DEFENSIVE: Buying {buy_amount} @ {best_ask} to reduce short position")
        
        # If we're long, get neutral
        elif position > 0:
            # Sell at best bid to exit long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DEFENSIVE: Selling {sell_amount} @ {best_bid} to exit long position")
        
        return orders
    
    def execute_opportunistic_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, price_trend):
        """Opportunistic strategy - seek quick profits while maintaining short bias"""
        orders = []
        
        # If deeply underinvested (not enough short position), add more shorts
        if position > -20:
            # Add more shorts
            sell_amount = min(best_bid_amount, 20 + position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"OPPORTUNISTIC: Selling {sell_amount} @ {best_bid} to establish minimum short position")
            return orders
            
        # If we're too heavily short in a possible uptrend, reduce position somewhat
        if position < -60 and price_trend > 0:
            # Buy to reduce short position
            buy_amount = min(-best_ask_amount, -position - 40)  # Target -40 position
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                print(f"OPPORTUNISTIC: Buying {buy_amount} @ {best_ask} to reduce heavy short in uptrend")
        
        # If we're near our preferred position, consider market making
        if abs(position - self.preferred_position) < 10:
            # Calculate the spread
            spread = best_ask - best_bid
            
            # Only market make if spread is decent
            if spread >= 2:
                # Place orders on both sides with a slight edge
                if position > self.preferred_position - 15:  # Only sell if not too short
                    sell_price = best_ask - 1  # Undercut the ask
                    sell_amount = min(5, self.position_limit.get(product, 70) + position)
                    if sell_amount > 0:
                        orders.append(Order(product, sell_price, -sell_amount))
                        print(f"OPPORTUNISTIC MM: Selling {sell_amount} @ {sell_price}")
                
                if position < self.preferred_position + 15:  # Only buy if not too long
                    buy_price = best_bid + 1  # Improve the bid
                    buy_amount = min(5, self.position_limit.get(product, 70) - position)
                    if buy_amount > 0:
                        orders.append(Order(product, buy_price, buy_amount))
                        print(f"OPPORTUNISTIC MM: Buying {buy_amount} @ {buy_price}")
        
        return orders
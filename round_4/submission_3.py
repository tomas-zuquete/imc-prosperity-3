import math
import jsonpickle
import numpy as np
from typing import Dict, List, Any, Tuple
from round_5.datamodel import Order, OrderDepth, TradingState, Trade, UserId

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

# Black-Scholes option pricing model
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
        self.price_history = {}
        self.position_history = []
        self.pnl_history = []
        self.last_pnl = 0
        self.best_pnl = 0
        self.new_history = {}
        
        # Market state tracking
        self.ema_prices = {product: None for product in PRODUCTS}
        self.ema_param = 0.5
        
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
        
        # Voucher-specific state for options trading
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
        
        # Add PnL to history for tracking performance
        self.pnl_history.append(pnl)
        if len(self.pnl_history) > 50:
            self.pnl_history = self.pnl_history[-50:]
            
        # Calculate performance metrics
        pnl_growth = 0
        if len(self.pnl_history) > 5:
            pnl_growth = (self.pnl_history[-1] - self.pnl_history[-5]) / max(1, abs(self.pnl_history[-5]))
            print(f"PnL 5-step growth: {pnl_growth:.2%}")
        
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
            
            # FIX: Use the method names that match the implemented methods
            elif product == RESIN:
                try:
                    # Change from enhanced_resin_mm_strategy to resin_mm_strategy
                    result[product] = self.resin_mm_strategy(state)
                except Exception as e:
                    print(f"Error in resin_strategy: {type(e).__name__}: {str(e)}")
            
            elif product == SQUID_INK:
                try:
                    # Change from enhanced_squid_ink_mm_strategy to squid_ink_mm_strategy
                    result[product] = self.squid_ink_mm_strategy(state)
                except Exception as e:
                    print(f"Error in ink_strategy: {type(e).__name__}: {str(e)}")
            
            elif product == PICNIC_BASKET1:
                try:
                    # Change from enhanced_basket1_strategy to picnic_1_mm_strategy
                    result[product] = self.picnic_1_mm_strategy(state)
                except Exception as e:
                    print(f"Error in picnic_basket1_strategy: {type(e).__name__}: {str(e)}")
            
            elif product == PICNIC_BASKET2:
                try:
                    # Change from enhanced_basket2_strategy to picnic_2_mm_strategy
                    result[product] = self.picnic_2_mm_strategy(state)
                except Exception as e:
                    print(f"Error in picnic_basket2_strategy: {type(e).__name__}: {str(e)}")
        
        # Process volcanic rock vouchers separately since they're interdependent
        try:
            if VOLCANIC_ROCK in state.order_depths or any(product in state.order_depths for product in self.voucher_strikes):
                # Change from enhanced_volcanic_rock_vouchers_strategy to volcanic_rock_vouchers_strategy
                voucher_results = self.volcanic_rock_vouchers_strategy(state)
                result.update(voucher_results)
        except Exception as e:
            print(f"Error in volcanic_rock_vouchers_strategy: {type(e).__name__}: {str(e)}")
            
        # Process other products if needed
        try:
            if CROISSANTS in state.order_depths:
                # Change from enhanced_croissants_strategy to croissants_strategy
                result[CROISSANTS] = self.croissants_strategy(state)
        except Exception as e:
            print(f"Error in croissants_strategy: {type(e).__name__}: {str(e)}")
            
        try:
            if JAMS in state.order_depths:
                # Change from enhanced_jams_strategy to jams_strategy
                result[JAMS] = self.jams_strategy(state)
        except Exception as e:
            print(f"Error in jams_strategy: {type(e).__name__}: {str(e)}")
            
        try:
            if DJEMBES in state.order_depths:
                # Change from enhanced_djembes_strategy to djembes_strategy
                result[DJEMBES] = self.djembes_strategy(state)
        except Exception as e:
            print(f"Error in djembes_strategy: {type(e).__name__}: {str(e)}")
            
        try:
            if KELP in state.order_depths:
                # Change from enhanced_kelp_strategy to kelp_strategy
                result[KELP] = self.kelp_strategy(state)
        except Exception as e:
            print(f"Error in kelp_strategy: {type(e).__name__}: {str(e)}")
            
        # Apply profit-taking logic if we're highly profitable in a product
        for product in list(result.keys()):
            if product in self.product_pnl and self.get_position(product, state) != 0:
                product_profit = self.product_pnl.get(product, 0)
                position = self.get_position(product, state)
                mid_price = self.get_mid_price(product, state)
                
                # If we have a significant position and significant profit in the product
                position_value = abs(position * mid_price)
                if position_value > 0 and abs(product_profit / position_value) > 0.25:  # 25% return threshold
                    # Apply profit-taking logic
                    result[product] = self.apply_profit_taking(product, result[product], state)
        
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
            price, _, _ = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, self.voucher_volatility, q=0)
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
        """Update exponential moving average prices for all products"""
        for product in PRODUCTS:
            if product not in state.order_depths:
                continue
                
            mid_price = self.get_mid_price(product, state)
            if product in self.voucher_strikes:
                S = self.get_mid_price(VOLCANIC_ROCK, state)
                K = self.voucher_strikes[product]
                mid_price, _, _ = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, self.voucher_volatility, q=0)
                
            if product not in self.ema_prices or self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1 - self.ema_param) * self.ema_prices[product]
    
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
            if position <= -70:
                return 10  # Convert short to buy and free up capacity
            elif position >= 70:
                return 10  # Convert long to free up capacity
                
            # If we have a moderate position but want to increase profits in the market phase
            if self.market_phase == "aggressive_short" and position <= -50:
                return 10  # Convert some shorts to establish new ones
                
            # If we have a position opposite to our strategy
            if self.market_phase in ["aggressive_short", "opportunistic"] and position >= 10:
                return 10  # Convert long position (against our strategy)
        
        # For now, no conversions for other products
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
            self.voucher_volatility = max(np.std(log_returns) * math.sqrt(252), 0.1)
        return self.voucher_volatility
        
    def implied_volatility(self, S, K, T, r, option_price, q=0) -> float:
        """Calculate implied volatility using numerical methods"""
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
    
    # Market making strategies for various products from round3.py
    def resin_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for Rainforest Resin with aggressive parameters
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
    
    def squid_ink_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for Squid Ink
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
        if pos > position_limit * 0.4:  # If we're long more than 40% of capacity
            # Try to sell at market
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(abs(pos), abs(bid_volume))
                orders.append(Order(product, best_bid, -sell_size))
        
        elif pos < -position_limit * 0.4:  # If we're short more than 40% of capacity
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
        
    def picnic_1_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for Picnic Basket 1
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
        if pos > position_limit * 0.4:  # If we're long more than 40% of capacity
            # Try to sell at market
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(abs(pos), abs(bid_volume))
                orders.append(Order(product, best_bid, -sell_size))
        
        elif pos < -position_limit * 0.4:  # If we're short more than 40% of capacity
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
        
    def volcanic_rock_vouchers_strategy(self, state: TradingState) -> Dict[str, List[Order]]:
        """Market-making strategy for VOLCANIC_ROCK vouchers."""
        result = {}
        S = self.get_mid_price(VOLCANIC_ROCK, state) or DEFAULT_PRICES[VOLCANIC_ROCK]
        self.past_prices[VOLCANIC_ROCK].append(S)
        if len(self.past_prices[VOLCANIC_ROCK]) > 25: #50
            self.update_volatility(self.past_prices[VOLCANIC_ROCK][-25:])  

        total_delta = 0
        
        rock_pos = self.get_position(VOLCANIC_ROCK, state)
        rock_bid_volume = self.position_limit[VOLCANIC_ROCK] - rock_pos
        rock_ask_volume = -self.position_limit[VOLCANIC_ROCK] - rock_pos
        hedging_orders = []

        trade_size = 50 #200
 
        for product in self.voucher_strikes:
            K = self.voucher_strikes[product]
            pos = self.get_position(product, state)

            # Use market-based pricing
            sigma = self.voucher_volatility
            market_price = None
            best_bid = None
            best_ask = None
            if product in state.order_depths and state.order_depths[product].buy_orders and state.order_depths[product].sell_orders:
                market_price = self.get_mid_price(product, state) or DEFAULT_PRICES[product]
                best_bid = max(state.order_depths[product].buy_orders.keys())
                best_ask = min(state.order_depths[product].sell_orders.keys())
                sigma = self.implied_volatility(S, K, self.time_to_maturity, self.risk_free_rate, market_price)
                sigma = max(sigma, 1e-6) if sigma is not None else 1e-6
            price, delta, gamma = black_scholes_call(S, K, self.time_to_maturity, self.risk_free_rate, sigma, q=0)

            spread = 1.5 + min(0.3 * gamma, 0.4)
            inventory_adjust = 0.02 * pos # 0.2
            fair_bid = (price - spread) / 2 - inventory_adjust
            fair_ask = (price + spread) / 2 - inventory_adjust

            # Match order book if competitive
            bid_price = int(best_bid) if best_bid and fair_bid <= best_bid else max(int(fair_bid), 1)
            ask_price = int(best_ask) if best_ask and fair_ask >= best_ask else int(fair_ask)

            bid_volume = self.position_limit[product] - pos
            ask_volume = -self.position_limit[product] - pos
            
            orders = []
            if bid_volume > 0:
                orders.append(Order(product, bid_price, min(bid_volume, trade_size)))
            if ask_volume < 0:
                orders.append(Order(product, ask_price, max(ask_volume, -trade_size)))
            result[product] = orders

            total_delta += pos * delta

            # Diagnostics
            market_mid = market_price if market_price else price
            print(f"{product}: bid={bid_price}, ask={ask_price}, best_bid={best_bid or 'N/A'}, best_ask={best_ask or 'N/A'}, pos={pos}, fair={price:.2f}, market={market_mid:.2f}, sigma={sigma:.3f}")

        voucher_value = 0
        hedging_pnl = 0
        for product in self.voucher_strikes:
            value = self.get_value_on_product(product, state)
            voucher_value += value
            pos = self.get_position(product, state)
            print(f"{product}: pos={pos}, value={value:.2f}")

        for product in state.own_trades:
            if product in self.voucher_strikes or product == VOLCANIC_ROCK:
                for trade in state.own_trades[product]:
                    if trade.timestamp == state.timestamp - 100:
                        mid = self.get_mid_price(product, state)
                        profit = (mid - trade.price) * trade.quantity if trade.buyer == SUBMISSION else (trade.price - mid) * trade.quantity
                        if product == VOLCANIC_ROCK:
                            hedging_pnl += profit
                        print(f"Trade: {product}, price={trade.price}, qty={trade.quantity}, side={'buy' if trade.buyer == SUBMISSION else 'sell'}, profit={profit:.2f}")
        print(f"Cash: {self.cash}, Voucher Value: {voucher_value:.2f}, Hedging P&L: {hedging_pnl:.2f}, Total Delta: {total_delta:.2f}")

        # Delta hedging
        target_rock_pos = -total_delta
        rock_trade = int(target_rock_pos - rock_pos)
       
        best_ask = min(state.order_depths.get(VOLCANIC_ROCK, {}).sell_orders.keys(), default=S)
        best_bid = max(state.order_depths.get(VOLCANIC_ROCK, {}).buy_orders.keys(), default=S)
        spread = best_ask - best_bid
        if spread > 5:
            best_ask = best_bid = S
        rock_trade = max(min(rock_trade, 400), -400)  
        if rock_trade > 0 and rock_bid_volume >= rock_trade:
            hedging_orders.append(Order(VOLCANIC_ROCK, int(best_ask), rock_trade))
            print(f"Hedge: Buy {rock_trade} VOLCANIC_ROCK at {best_ask}, delta={total_delta:.2f}")
        elif rock_trade < 0 and rock_ask_volume <= rock_trade:
            hedging_orders.append(Order(VOLCANIC_ROCK, int(best_bid), rock_trade))
            print(f"Hedge: Sell {-rock_trade} VOLCANIC_ROCK at {best_bid}, delta={total_delta:.2f}")

        result[VOLCANIC_ROCK] = hedging_orders
        return result
        
    def picnic_2_mm_strategy(self, state: TradingState) -> List[Order]: 
        """
        Market making strategy for Picnic Basket 2
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
        if pos > position_limit * 0.15:  # If we're long more than 15% of capacity
            # Try to sell at market
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(abs(pos), abs(bid_volume))
                orders.append(Order(product, best_bid, -sell_size))
        
        elif pos < -position_limit * 0.15:  # If we're short more than 15% of capacity
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

    # Strategies for various products from round3.py
    def croissants_strategy(self, state: TradingState) -> List[Order]:
        product = CROISSANTS
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        
        # Initialize price history if needed
        if product not in self.new_history:
            self.new_history[product] = []
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
        
        # Initialize price history if needed
        if product not in self.new_history:
            self.new_history[product] = []
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
        
        # Initialize price history if needed
        if product not in self.new_history:
            self.new_history[product] = []
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

    def kelp_strategy(self, state: TradingState) -> List[Order]:
        mid_price = self.get_mid_price(KELP, state)
        self.kelp_mid_prices.append(mid_price)
        perc_diff = 0
        if self.kelp_prev_mid_price and self.kelp_prev_mid_price != 0:
            perc_diff = (mid_price - self.kelp_prev_mid_price) / self.kelp_prev_mid_price
        self.kelp_prev_mid_price = mid_price
        
        if len(self.kelp_mid_prices) >= 20:
            X = [(self.kelp_mid_prices[i] - self.kelp_mid_prices[i-1]) / self.kelp_mid_prices[i-1]
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

        # MAGNIFICENT_MACARONS strategy (from macarons trader)
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
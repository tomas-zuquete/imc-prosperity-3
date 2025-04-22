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
        # Position limits for all products
        self.position_limit = {
            MAGNIFICENT_MACARONS: 70,
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
        
        # Product-specific price tracking (only store latest values, not full history)
        self.ink_prev_mid_price = None
        self.kelp_prev_mid_price = None
        
        # Limit the size of price histories to prevent memory growth
        self.price_latest = {}  # Store only the latest few prices for each product
        self.short_price_window = 5  # Keep only 5 latest prices for trend calculation
        
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
                # Only restore essential state variables to minimize serialization overhead
                essential_keys = [
                    'round', 'cash', 'positions', 'ema_prices', 
                    'market_phase', 'phase_counter', 'last_mid', 
                    'ink_prev_mid_price', 'kelp_prev_mid_price',
                    'price_latest', 'voucher_volatility', 'time_to_maturity',
                    'ou_params', 'kelp_ou_params'
                ]
                for key in essential_keys:
                    if key in trader_state:
                        self.__dict__[key] = trader_state[key]
            except Exception as e:
                print(f"Error restoring state: {e}")
        
        # Update positions from current state
        for product, pos in state.position.items():
            self.positions[product] = pos
        
        # Increment round count and calculate pnl
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)
        
        # Minimal debug output to avoid overhead
        print(f"Round {self.round} - Cash: {self.cash} - PnL: {pnl}")
        
        # Initialize result container
        result = {}
        
        # Process each product with its respective strategy if it exists in order_depths
        for product in state.order_depths:
            if product == MAGNIFICENT_MACARONS:
                self.trades_this_round = 0
                # Modified: Handle macarons trading directly here rather than calling a separate method
                orders = []
                order_depth = state.order_depths[product]
                position = self.positions.get(product, 0)
                
                # Get best prices from order book
                best_bid = None
                best_bid_amount = None
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]

                best_ask = None
                best_ask_amount = None
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                
                # If market is one-sided, skip this product
                if not best_bid or not best_ask:
                    continue
                
                # Calculate mid price
                mid_price = (best_bid + best_ask) / 2
                
                # Update price history with fixed window
                if product not in self.price_latest:
                    self.price_latest[product] = []
                
                self.price_latest[product].append(mid_price)
                if len(self.price_latest[product]) > self.short_price_window:
                    self.price_latest[product] = self.price_latest[product][-self.short_price_window:]
                
                # Calculate price trend (negative = downtrend)
                price_trend = self.calculate_trend(product)
                
                # Basic strategy: maintain a short bias
                if position > 0:  # If we have a long position, reduce it
                    sell_amount = min(best_bid_amount, position)
                    if sell_amount > 0:
                        orders.append(Order(product, best_bid, -sell_amount))
                
                elif position > -40:  # If we're not at our preferred short position
                    sell_amount = min(best_bid_amount, 40 + position)
                    if sell_amount > 0:
                        orders.append(Order(product, best_bid, -sell_amount))
                
                elif position < -60:  # If we're too short, reduce position
                    buy_amount = min(-best_ask_amount, -position - 40)
                    if buy_amount > 0:
                        orders.append(Order(product, best_ask, buy_amount))
                
                self.last_mid = mid_price
                result[product] = orders
            
            elif product == RESIN:
                try:
                    result[product] = self.resin_mm_strategy(state)
                except Exception as e:
                    print(f"Error in resin_strategy: {type(e).__name__}")
            
            elif product == SQUID_INK:
                try:
                    result[product] = self.squid_ink_mm_strategy(state)
                except Exception as e:
                    print(f"Error in ink_strategy: {type(e).__name__}")
            
            # elif product == PICNIC_BASKET1:
            #     try:
            #         result[product] = self.picnic_1_mm_strategy(state)
            #     except Exception as e:
            #         print(f"Error in picnic_basket1_strategy: {type(e).__name__}")
            
            # elif product == PICNIC_BASKET2:
            #     try:
            #         result[product] = self.picnic_2_mm_strategy(state)
            #     except Exception as e:
            #         print(f"Error in picnic_basket2_strategy: {type(e).__name__}")
            
            # elif product == CROISSANTS:
            #     try:
            #         result[product] = self.croissants_strategy(state)
            #     except Exception as e:
            #         print(f"Error in croissants_strategy: {type(e).__name__}")
                    
            # elif product == JAMS:
            #     try:
            #         result[product] = self.jams_strategy(state)
            #     except Exception as e:
            #         print(f"Error in jams_strategy: {type(e).__name__}")
                    
            # elif product == DJEMBES:
            #     try:
            #         result[product] = self.djembes_strategy(state)
            #     except Exception as e:
            #         print(f"Error in djembes_strategy: {type(e).__name__}")
                    
            # elif product == KELP:
            #     try:
            #         result[product] = self.kelp_strategy(state)
            #     except Exception as e:
            #         print(f"Error in kelp_strategy: {type(e).__name__}")
        
        # Process volcanic rock vouchers separately since they're interdependent
        # try:
        #     if VOLCANIC_ROCK in state.order_depths or any(product in state.order_depths for product in self.voucher_strikes):
        #         voucher_results = self.volcanic_rock_vouchers_strategy(state)
        #         result.update(voucher_results)
        # except Exception as e:
        #     print(f"Error in volcanic_rock_vouchers_strategy: {type(e).__name__}")
        
        # Determine if conversion is needed
        conversion = 0
        for product in state.position:
            product_conversion = self.determine_conversion(product, state.position.get(product, 0))
            if product_conversion > 0:
                conversion = product_conversion
                break
        
        # Save state - only serialize essential data
        essential_state = {
            'round': self.round,
            'cash': self.cash,
            'positions': self.positions,
            'ema_prices': self.ema_prices,
            'market_phase': self.market_phase,
            'phase_counter': self.phase_counter,
            'last_mid': self.last_mid,
            'ink_prev_mid_price': self.ink_prev_mid_price,
            'kelp_prev_mid_price': self.kelp_prev_mid_price,
            'price_latest': self.price_latest,
            'voucher_volatility': self.voucher_volatility,
            'time_to_maturity': self.time_to_maturity,
            'ou_params': self.ou_params,
            'kelp_ou_params': self.kelp_ou_params
        }
        trader_data = jsonpickle.encode(essential_state)
        
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
            
            # Update price_latest with fixed window size
            if product not in self.price_latest:
                self.price_latest[product] = []
            
            self.price_latest[product].append(mid_price)
            if len(self.price_latest[product]) > self.short_price_window:
                self.price_latest[product] = self.price_latest[product][-self.short_price_window:]
    
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
        prices = self.price_latest.get(product, [])
        
        if len(prices) < 2:
            return 0  # Not enough data
            
        # Calculate short-term trend (using available prices)
        return (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        
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
        for _ in range(5):  # Reduced iterations for performance
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
    
    # Market making strategies for various products
    def resin_mm_strategy(self, state: TradingState) -> List[Order]: 
        """Market making strategy for Rainforest Resin with aggressive parameters"""
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
        
        # Update price history
        if product not in self.price_latest:
            self.price_latest[product] = []
        
        self.price_latest[product].append(mid)
        if len(self.price_latest[product]) > self.short_price_window:
            self.price_latest[product] = self.price_latest[product][-self.short_price_window:]
        
        # Calculate available capacity
        position_limit = self.position_limit[product]
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        orders = []
        
        # Market making approach - try to capture bid-ask spread
        if order_depth.buy_orders and order_depth.sell_orders:
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
        
        return orders
    
    def squid_ink_mm_strategy(self, state: TradingState) -> List[Order]: 
        """Market making strategy for Squid Ink"""
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
        
        # Update price history with fixed window size
        if product not in self.price_latest:
            self.price_latest[product] = []
        
        self.price_latest[product].append(mid)
        if len(self.price_latest[product]) > self.short_price_window:
            self.price_latest[product] = self.price_latest[product][-self.short_price_window:]
        
        # Calculate available capacity
        position_limit = self.position_limit[product]
        buy_capacity = position_limit - pos
        sell_capacity = position_limit + pos
        
        orders = []
        
        # Market making approach - try to capture bid-ask spread
        if order_depth.buy_orders and order_depth.sell_orders:
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
        
        return orders
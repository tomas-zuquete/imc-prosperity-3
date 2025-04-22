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
        print(f"Error in Black-Scholes: {e}")
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
                    'ou_params', 'kelp_ou_params', 'preferred_position'
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
            # if product == MAGNIFICENT_MACARONS:
            #     self.trades_this_round = 0
            #     orders = self.trade_macarons(
            #         product,
            #         state.order_depths[product],
            #         self.positions.get(product, 0)
            #     )
            #     result[product] = orders
            
            # elif product == RESIN:
            # if product == RESIN:
            #     try:
            #         result[product] = self.resin_mm_strategy(state)
            #     except Exception as e:
            #         print(f"Error in resin_strategy: {type(e).__name__}")
            
            # elif product == SQUID_INK:
            #     try:
            #         result[product] = self.squid_ink_mm_strategy(state)
            #     except Exception as e:
            #         print(f"Error in ink_strategy: {type(e).__name__}")
            
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
            if product == KELP:
                try:
                    result[product] = self.kelp_strategy(state)
                except Exception as e:
                    print(f"Error in kelp_strategy: {type(e).__name__}")
        
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
            'kelp_ou_params': self.kelp_ou_params,
            'preferred_position': self.preferred_position
        }
        trader_data = jsonpickle.encode(essential_state)
        
        return result, conversion, trader_data

    def trade_macarons(self, product, order_depth, position):
        """Optimized trading strategy for MAGNIFICENT_MACARONS"""
        orders = []
        
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
        
        # If market is one-sided, return empty orders
        if not best_bid or not best_ask:
            return orders
        
        # Calculate mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Update price history with fixed window
        if product not in self.price_latest:
            self.price_latest[product] = []
        
        self.price_latest[product].append(mid_price)
        if len(self.price_latest[product]) > self.short_price_window:
            self.price_latest[product] = self.price_latest[product][-self.short_price_window:]
        
        # Calculate price trend (negative value = downtrend)
        price_trend = self.calculate_trend(product)
        
        # Simple strategy for MAGNIFICENT_MACARONS
        # Focus on maintaining a short bias around -40 position
        
        # If we have a long position, exit immediately
        if position > 0:
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                return orders
        
        # If we don't have enough short position
        if position > -30:
            # Add more shorts
            sell_amount = min(best_bid_amount, 30 + position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                return orders
                
        # If we're too short, reduce position
        if position < -50:
            buy_amount = min(-best_ask_amount, -position - 40)
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                return orders
        
        # If we're near our target position, consider market making
        if abs(position - self.preferred_position) < 10:
            spread = best_ask - best_bid
            if spread >= 2:
                # Place orders on both sides with a slight edge
                if position > self.preferred_position - 10:
                    sell_price = best_ask - 1
                    sell_amount = min(5, self.position_limit.get(product, 70) + position)
                    if sell_amount > 0:
                        orders.append(Order(product, sell_price, -sell_amount))
                
                if position < self.preferred_position + 10:
                    buy_price = best_bid + 1
                    buy_amount = min(5, self.position_limit.get(product, 70) - position)
                    if buy_amount > 0:
                        orders.append(Order(product, buy_price, buy_amount))
        
        self.last_mid = mid_price
        return orders
        
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
    
    def kelp_strategy(self, state: TradingState) -> List[Order]:
        """Reversed trading approach for KELP - doing the opposite of the losing strategy"""
        product = KELP
        position = self.get_position(product, state)
        position_limit = self.position_limit[product]
        
        # Initialize orders list
        orders = []
        
        # Validate order depth exists
        if product not in state.order_depths:
            return orders
        
        order_depth = state.order_depths[product]
        
        # Get best bid and ask from the market
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        market_bid = max(order_depth.buy_orders.keys())
        market_ask = min(order_depth.sell_orders.keys())
        mid_price = (market_bid + market_ask) / 2
        
        # Update price history with fixed window size
        if product not in self.price_latest:
            self.price_latest[product] = []
        
        self.price_latest[product].append(mid_price)
        if len(self.price_latest[product]) > 20:
            self.price_latest[product] = self.price_latest[product][-20:]
        
        # Get the market volumes
        bid_volume = abs(order_depth.buy_orders[market_bid])
        ask_volume = abs(order_depth.sell_orders[market_ask])
        
        # Calculate volume imbalance (positive means more buyers than sellers)
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            volume_imbalance = (bid_volume - ask_volume) / total_volume
        else:
            volume_imbalance = 0
        
        # Calculate available capacity
        bid_capacity = position_limit - position  # How much more we can buy
        ask_capacity = position_limit + position  # How much more we can sell
        
        # Calculate price trends
        if len(self.price_latest[product]) >= 3:
            short_trend = self.price_latest[product][-1] - self.price_latest[product][-3]  # Very short-term trend
        else:
            short_trend = 0
        
        # REVERSED STRATEGY 1: Take the opposite side of volume imbalance
        if volume_imbalance > 0.3 and short_trend > 0:  # Strong buying pressure and uptrend
            if ask_capacity > 0:
                # Sell into the buying pressure
                sell_price = market_bid
                sell_size = min(30, ask_capacity)
                orders.append(Order(product, sell_price, -sell_size))
                print(f"KELP: COUNTER-TREND SELL {sell_size} @ {sell_price} (buying pressure: {volume_imbalance:.2f})")
                return orders
                
        elif volume_imbalance < -0.3 and short_trend < 0:  # Strong selling pressure and downtrend
            if bid_capacity > 0:
                # Buy into the selling pressure
                buy_price = market_ask
                buy_size = min(30, bid_capacity)
                orders.append(Order(product, buy_price, buy_size))
                print(f"KELP: COUNTER-TREND BUY {buy_size} @ {buy_price} (selling pressure: {volume_imbalance:.2f})")
                return orders
        
        # REVERSED STRATEGY 2: Exploit narrow spreads instead of wide ones
        spread = market_ask - market_bid
        if spread <= 2:  # Narrow spread
            # Place orders at the opposite side of your position
            if position < 0:  # Short, place sell orders
                if ask_capacity > 0:
                    orders.append(Order(product, market_bid, -min(25, ask_capacity)))
                    print(f"KELP: CONTRARIAN SELL {min(25, ask_capacity)} @ {market_bid}")
            elif position > 0:  # Long, place buy orders
                if bid_capacity > 0:
                    orders.append(Order(product, market_ask, min(25, bid_capacity)))
                    print(f"KELP: CONTRARIAN BUY {min(25, bid_capacity)} @ {market_ask}")
            else:  # No position, take both sides but opposite of vol imbalance
                if volume_imbalance > 0 and ask_capacity > 0:  # More buying, so we sell
                    orders.append(Order(product, market_bid, -min(25, ask_capacity)))
                    print(f"KELP: CONTRARIAN SELL {min(25, ask_capacity)} @ {market_bid}")
                elif volume_imbalance < 0 and bid_capacity > 0:  # More selling, so we buy
                    orders.append(Order(product, market_ask, min(25, bid_capacity)))
                    print(f"KELP: CONTRARIAN BUY {min(25, bid_capacity)} @ {market_ask}")
        else:  # Wide spread (≥3)
            # Stay out of the market when spread is wide to avoid getting caught
            pass
        
        # If we have a significant position, actively reduce it
        if abs(position) > 15:
            if position > 15 and ask_capacity > 0:  # Too long
                orders = [Order(product, market_bid, -min(40, ask_capacity))]
                print(f"KELP: POSITION REDUCTION SELL {min(40, ask_capacity)} @ {market_bid}")
            elif position < -15 and bid_capacity > 0:  # Too short
                orders = [Order(product, market_ask, min(40, bid_capacity))]
                print(f"KELP: POSITION REDUCTION BUY {min(40, bid_capacity)} @ {market_ask}")
        
        print(f"KELP Summary: position={position}, mid={mid_price:.1f}, spread={spread}, imbalance={volume_imbalance:.2f}, orders={[(o.price, o.quantity) for o in orders]}")
        
        return orders

    
    # def kelp_strategy(self, state: TradingState) -> List[Order]:
    #     """Market making strategy for KELP with mean reversion"""
    #     product = KELP
    #     mid_price = self.get_mid_price(product, state)
        
    #     # Store only a few recent prices
    #     if product not in self.price_latest:
    #         self.price_latest[product] = []
        
    #     self.price_latest[product].append(mid_price)
    #     if len(self.price_latest[product]) > 20:
    #         self.price_latest[product] = self.price_latest[product][-20:]
        
    #     perc_diff = 0
    #     if self.kelp_prev_mid_price and self.kelp_prev_mid_price != 0:
    #         perc_diff = (mid_price - self.kelp_prev_mid_price) / self.kelp_prev_mid_price
    #     self.kelp_prev_mid_price = mid_price
        
    #     if len(self.price_latest[product]) >= 20:
    #         try:
    #             X = [(self.price_latest[product][i] - self.price_latest[product][i-1]) / self.price_latest[product][i-1]
    #                 for i in range(1, len(self.price_latest[product])) if self.price_latest[product][i-1] != 0]
    #             if X:
    #                 mu, theta, sigma = self.fit_ou(X)
    #                 self.kelp_ou_params.update({"mu": mu, "theta": theta, "sigma": sigma})
    #         except Exception:
    #             pass
                
    #     mu, theta, sigma = self.kelp_ou_params["mu"], self.kelp_ou_params["theta"], self.kelp_ou_params["sigma"]
    #     position_kelp = self.get_position(product, state)
        
    #     # Avoid division by zero
    #     denom = (sigma / np.sqrt(2 * theta)) if sigma > 0 and theta > 0 else 1
    #     z_score = (perc_diff - mu) / denom if denom != 0 else 0
        
    #     orders = []
    #     max_trade_size = 50
    #     bid_volume = min(max_trade_size, self.position_limit[product] - position_kelp)
    #     ask_volume = min(max_trade_size, -self.position_limit[product] - position_kelp)
        
    #     if product in state.order_depths:
    #         order_depth = state.order_depths[product]
    #         best_ask = min(order_depth.sell_orders.keys(), default=int(mid_price + 1))
    #         best_bid = max(order_depth.buy_orders.keys(), default=int(mid_price - 1))
            
    #         if perc_diff != 0:
    #             if z_score >= 0.5 and ask_volume > 0:
    #                 size = min(max_trade_size, ask_volume, abs(order_depth.buy_orders.get(best_bid, 0)))
    #                 if size > 0:
    #                     orders.append(Order(product, best_bid, -size))
    #             elif z_score <= -0.5 and bid_volume > 0:
    #                 size = min(max_trade_size, bid_volume, abs(order_depth.sell_orders.get(best_ask, 0)))
    #                 if size > 0:
    #                     orders.append(Order(product, best_ask, size))
    #             else:
    #                 fair_price = mid_price * (1 + mu)
    #                 bid_price = int(fair_price - 1)
    #                 ask_price = int(fair_price + 1)
                    
    #                 if bid_volume > 0:
    #                     orders.append(Order(product, bid_price, bid_volume))
    #                 if ask_volume < 0:
    #                     orders.append(Order(product, ask_price, ask_volume))
        
    #     return orders
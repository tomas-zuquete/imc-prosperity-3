import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


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
        logger.print("Initializing Trader...")

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

        logger.print(f"Log round {self.round}")

        logger.print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    logger.print(trade)

        logger.print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            if product in state.position:
                logger.print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}")
        logger.print(f"\tPnL {pnl}")

        # Initialize the method output dict as an empty dict
        result = {}
        
        # # Other strategies you may want to activate
        # try:
        #     result[SQUID_INK] = self.ink_strategy(state)
        # except Exception as e:
        #     logger.print("Error occurred while executing ink_strategy:")
        #     logger.print(f"Exception Type: {type(e).__name__}")
        #     logger.print(f"Exception Message: {str(e)}")
        
        #############################################################################################################################

        # Trade each market-amking product individually
        try:
            result[RESIN] = self.resin_mm_strategy(state)
        except Exception as e:
            logger.print(f"Error occurred while executing resin_mm_strategy:")
            logger.print(f"Exception Type: {type(e).__name__}")
            logger.print(f"Exception Message: {str(e)}")
            
        # for MM_PRODUCT in MM_PRODUCTS:
        #     try:
        #         result[MM_PRODUCT] = self.resin_mm_strategy(MM_PRODUCT, state)
        #     except Exception as e:
        #         logger.print(f"Error occurred while executing resin_mm_strategy:")
        #         logger.print(f"Exception Type: {type(e).__name__}")
        #         logger.print(f"Exception Message: {str(e)}")
            
            
        #############################################################################################################################

        # # Trade each voucher individually
        # try:
        #     if VOLCANIC_ROCK_VOUCHER_9500 in state.order_depths:
        #         result[VOLCANIC_ROCK_VOUCHER_9500] = self.volcanic_rock_voucher_strategy(state, VOLCANIC_ROCK_VOUCHER_9500)
        # except Exception as e:
        #     logger.print(f"Error trading {VOLCANIC_ROCK_VOUCHER_9500}: {str(e)}")
        
        # try:
        #     if VOLCANIC_ROCK_VOUCHER_9750 in state.order_depths:
        #         result[VOLCANIC_ROCK_VOUCHER_9750] = self.volcanic_rock_voucher_strategy(state, VOLCANIC_ROCK_VOUCHER_9750)
        # except Exception as e:
        #     logger.print(f"Error trading {VOLCANIC_ROCK_VOUCHER_9750}: {str(e)}")
        
        # try:
        #     if VOLCANIC_ROCK_VOUCHER_10000 in state.order_depths:
        #         result[VOLCANIC_ROCK_VOUCHER_10000] = self.volcanic_rock_voucher_strategy(state, VOLCANIC_ROCK_VOUCHER_10000)
        # except Exception as e:
        #     logger.print(f"Error trading {VOLCANIC_ROCK_VOUCHER_10000}: {str(e)}")
        
        # try:
        #     if VOLCANIC_ROCK_VOUCHER_10250 in state.order_depths:
        #         result[VOLCANIC_ROCK_VOUCHER_10250] = self.volcanic_rock_voucher_strategy(state, VOLCANIC_ROCK_VOUCHER_10250)
        # except Exception as e:
        #     logger.print(f"Error trading {VOLCANIC_ROCK_VOUCHER_10250}: {str(e)}")
        
        # try:
        #     if VOLCANIC_ROCK_VOUCHER_10500 in state.order_depths:
        #         result[VOLCANIC_ROCK_VOUCHER_10500] = self.volcanic_rock_voucher_strategy(state, VOLCANIC_ROCK_VOUCHER_10500)
        # except Exception as e:
        #     logger.print(f"Error trading {VOLCANIC_ROCK_VOUCHER_10500}: {str(e)}")
        
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
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

    # # Volcanic rock strategies
    # def old_volcanic_rock_strategy(self, state: TradingState) -> List[Order]:
    #     """
    #     Market making strategy for volcanic rock
        
    #     Args:
    #         state: Current trading state
            
    #     Returns:
    #         List of orders for volcanic rock
    #     """
    #     product = VOLCANIC_ROCK
    #     pos = self.get_position(product, state)
        
    #     # Calculate fair value using mid price
    #     mid = self.get_mid_price(product, state)
        
    #     # Store price for volatility calculation
    #     if product not in self.new_history:
    #         self.new_history[product] = []
    #     self.new_history[product].append(mid)
        
    #     # Market making spread
    #     base_spread = 2
        
    #     # Adjust spread based on position
    #     position_factor = abs(pos) / (self.position_limit[product] * 0.5)
    #     spread_adjustment = int(position_factor * 5)
        
    #     # Calculate bid and ask prices
    #     if pos > 0:
    #         # We're long, so make selling more attractive
    #         bid_price = int(mid - (base_spread + spread_adjustment))
    #         ask_price = int(mid + base_spread)
    #     elif pos < 0:
    #         # We're short, so make buying more attractive
    #         bid_price = int(mid - base_spread)
    #         ask_price = int(mid + (base_spread + spread_adjustment))
    #     else:
    #         # Neutral position
    #         bid_price = int(mid - base_spread)
    #         ask_price = int(mid + base_spread)
        
    #     # Calculate order volumes
    #     position_limit = self.position_limit[product]
        
    #     # Scale order sizes based on proximity to position limit
    #     size_factor = 1.0 - (abs(pos) / position_limit)
    #     base_size = max(1, int(40 * size_factor))
        
    #     bid_volume = min(base_size, position_limit - pos)
    #     ask_volume = min(base_size, position_limit + pos)
        
    #     orders = []
        
    #     # Only place orders if we have capacity
    #     if bid_volume > 0:
    #         orders.append(Order(product, bid_price, bid_volume))
        
    #     if ask_volume > 0:
    #         orders.append(Order(product, ask_price, -ask_volume))
        
    #     logger.print(f"[VOLCANIC_ROCK] pos={pos}, mid={mid:.1f}, spread={base_spread+spread_adjustment}, bid={bid_price}x{bid_volume}, ask={ask_price}x{ask_volume}")
        
    #     return orders

    # Resin pnl - 1.91k
    def resin_mm_strategy(self, MM_PRODUCT, state: TradingState) -> List[Order]: 
        """
        Market making strategy for resin rock with more aggressive parameters
        
        Args:
            state: Current trading state
            
        Returns:
            List of orders for volcanic rock
        """
        product = MM_PRODUCT
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
        
        # logger.print debug info
        logger.print(f"[VOLCANIC_ROCK] pos={pos}, mid={mid:.1f}, orders={[(o.price, o.quantity) for o in orders]}")
        
        return orders


    def volcanic_rock_voucher_strategy(self, state: TradingState, product: str) -> List[Order]:
        """
        Trading strategy for volcanic rock vouchers based on option pricing theory
        
        Args:
            state: Current trading state
            product: Voucher product name
            
        Returns:
            List of orders for the voucher
        """
        # Get current position
        pos = self.get_position(product, state)
        
        # Get strike price for this voucher
        strike_price = int(product.split("_")[-1])
        
        # Get underlying price
        rock_price = self.get_mid_price(VOLCANIC_ROCK, state)
        
        # Get market price of voucher
        market_price = self.get_mid_price(product, state)
        
        # Calculate days to expiry based on current round
        days_to_expiry = 8 - self.round
        
        # Calculate time to expiry in years (assuming 365 days in a year)
        T = max(0.001, days_to_expiry / 365.0)
        
        # Estimate volatility based on volcanic rock price history
        if len(self.new_history.get(VOLCANIC_ROCK, [])) >= 20:
            prices = self.new_history[VOLCANIC_ROCK]
            returns = [math.log(prices[i]/prices[i-1]) for i in range(1, len(prices))]
            volatility = np.std(returns) * math.sqrt(252)  # Annualized
            self.volatility = max(0.1, min(volatility, 1.0))  # Bound between 10% and 100%
        else:
            volatility = self.volatility
        
        # Calculate theoretical price using Black-Scholes
        r = 0.0  # Risk-free rate
        try:
            theoretical_price = black_scholes_call_price(rock_price, strike_price, T, r, volatility)
        except:
            # Fallback to intrinsic value if calculation fails
            theoretical_price = max(0, rock_price - strike_price)
        
        # Price difference between theoretical and market price
        price_diff = market_price - theoretical_price
        
        # Calculate bid/ask spread based on proximity to expiry
        # Wider spreads as expiry approaches due to increased volatility risk
        base_spread = 2 + int(5 * (1 / max(1, days_to_expiry)))
        
        # Calculate delta for risk management
        try:
            delta = black_scholes_call_delta(rock_price, strike_price, T, r, volatility)
        except:
            # Fallback if calculation fails
            delta = 1.0 if rock_price > strike_price else 0.0
        
        # Adjust our fair value based on our position and delta
        position_adjustment = -pos * 0.5 * delta
        adjusted_fair_value = theoretical_price + position_adjustment
        
        # Set bid/ask prices around our adjusted fair value
        bid_price = max(1, int(adjusted_fair_value - base_spread))
        ask_price = max(bid_price + 1, int(adjusted_fair_value + base_spread))
        
        # Calculate order volumes
        position_limit = self.position_limit[product]
        
        # Scale order sizes based on delta (smaller sizes for low delta options)
        size_factor = max(0.2, delta) * (1.0 - (abs(pos) / position_limit))
        base_size = max(1, int(20 * size_factor))
        
        bid_volume = min(base_size, position_limit - pos)
        ask_volume = min(base_size, position_limit + pos)
        
        # Adjust volumes based on mispricing
        if price_diff > base_spread:  # Market price higher than theoretical
            # Opportunity to sell at a premium
            ask_volume = int(ask_volume * 1.5)
            bid_volume = max(1, int(bid_volume * 0.5))
        elif price_diff < -base_spread:  # Market price lower than theoretical
            # Opportunity to buy at a discount
            bid_volume = int(bid_volume * 1.5)
            ask_volume = max(1, int(ask_volume * 0.5))
        
        orders = []
        
        # Only place orders if we have capacity and they make sense
        if bid_volume > 0 and bid_price > 0:
            orders.append(Order(product, bid_price, bid_volume))
        
        if ask_volume > 0 and days_to_expiry > 0:  # Don't sell options on expiry day
            orders.append(Order(product, ask_price, -ask_volume))
        
        logger.print(f"[{product}] pos={pos}, theo={theoretical_price:.1f}, market={market_price:.1f}, " +
              f"delta={delta:.2f}, days_left={days_to_expiry}, bid={bid_price}x{bid_volume}, ask={ask_price}x{ask_volume}")
        
        return orders
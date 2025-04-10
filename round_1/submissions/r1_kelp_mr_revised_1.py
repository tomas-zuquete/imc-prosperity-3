import math
from typing import Dict, List, Any, Tuple
# import numpy as np # Not used
from datamodel import Order, TradingState, OrderDepth, UserId # Assuming these are in datamodel.py

# Constants
SUBMISSION = "SUBMISSION"
KELP = "KELP"
PRODUCTS = [KELP]
DEFAULT_PRICES = {KELP: 2030} # Fallback/initial estimate

class Trader:
    def __init__(self) -> None:
        print("Initializing Trader (v4 - Trend Following Logic)...")

        # --- Parameters based on the working example provided ---
        self.position_limit = {KELP: 50}
        self.ema_short_param = 2 / (5 + 1)
        self.ema_long_param = 2 / (20 + 1)
        self.ema_threshold = 0.0005 # From working example
        self.z_score_window = 20
        self.z_score_entry = 0.6  # From working example (used for strength check)
        self.z_score_exit = 0.2   # From working example (for profit taking)
        self.z_stop_loss = 1.5    # From working example
        self.risk_factor = 0.4    # From working example
        self.volatility_adjustment_factor = 5 # Kept from previous

        # --- State Variables ---
        self.round = 0
        self.cash = 0
        self.past_prices = {p: [] for p in PRODUCTS}
        self.ema_short = {p: None for p in PRODUCTS}
        self.ema_long = {p: None for p in PRODUCTS}
        self.rolling_mean = {p: None for p in PRODUCTS}
        self.rolling_std = {p: None for p in PRODUCTS}
        self.volatility = {p: 0.01 for p in PRODUCTS}
        self.last_trade_time = {KELP: -500} # Initialize to allow immediate first trade

    # --- Utility Methods (Unchanged from v3) ---
    def get_position(self, product, state: TradingState) -> int:
        return state.position.get(product, 0)

    def get_bid_ask_mid_price(self, product, state: TradingState) -> Tuple[int | None, int | None, float | None]:
        if product not in state.order_depths: return None, None, self.ema_short.get(product, DEFAULT_PRICES.get(product))
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        best_bid = max(market_bids.keys()) if market_bids else None
        best_ask = min(market_asks.keys()) if market_asks else None
        if best_bid is not None and best_ask is not None: return best_bid, best_ask, (best_bid + best_ask) / 2
        elif best_bid is not None: return best_bid, None, self.ema_short.get(product, best_bid + 0.5) # Estimate mid slightly above bid
        elif best_ask is not None: return None, best_ask, self.ema_short.get(product, best_ask - 0.5) # Estimate mid slightly below ask
        else: return None, None, self.ema_short.get(product, DEFAULT_PRICES.get(product))


    # --- Indicator Update Methods (Unchanged from v3) ---
    def update_price_history(self, product, mid_price):
        if mid_price is None: return
        self.past_prices[product].append(mid_price)
        if len(self.past_prices[product]) > 50: self.past_prices[product].pop(0)
        # Simple volatility calc (use previous robust one if preferred)
        if len(self.past_prices[product]) >= 5:
            prices = self.past_prices[product][-5:]
            returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices)) if prices[i-1] != 0]
            if returns:
                 variance = sum((r - (sum(returns) / len(returns))) ** 2 for r in returns) / len(returns) if len(returns) > 0 else 0
                 self.volatility[product] = max(math.sqrt(variance), 0.0001) # Min volatility floor


    def update_emas(self, product, mid_price):
        if mid_price is None: return
        if self.ema_short[product] is None: self.ema_short[product] = mid_price
        else: self.ema_short[product] = mid_price * self.ema_short_param + (1 - self.ema_short_param) * self.ema_short[product]
        if self.ema_long[product] is None: self.ema_long[product] = mid_price
        else: self.ema_long[product] = mid_price * self.ema_long_param + (1 - self.ema_long_param) * self.ema_long[product]

    def update_z_score_stats(self, product, mid_price):
        if mid_price is None or len(self.past_prices[product]) < 5: return
        window = min(len(self.past_prices[product]), self.z_score_window)
        prices = self.past_prices[product][-window:]
        self.rolling_mean[product] = sum(prices) / len(prices)
        variance = sum((p - self.rolling_mean[product]) ** 2 for p in prices) / len(prices)
        self.rolling_std[product] = max(math.sqrt(variance), 0.1) # Min std dev floor

    def calculate_z_score(self, product, mid_price):
        if self.rolling_mean.get(product) is None or self.rolling_std.get(product) is None or self.rolling_std[product] == 0 or mid_price is None:
            return 0
        return (mid_price - self.rolling_mean[product]) / self.rolling_std[product]

    # --- Signal & Sizing Logic (ADOPTED FROM WORKING EXAMPLE `r2.py`) ---
    def identify_signal(self, product, mid_price, state: TradingState) -> str:
        """Identifies signal based on EMA Crossover + Momentum Filter + Z-Score Strength."""
        if self.ema_short[product] is None or self.ema_long[product] is None or mid_price is None:
            return "NEUTRAL"

        # 1. EMA Crossover Signal
        ema_short = self.ema_short[product]
        ema_long = self.ema_long[product]
        diff_pct = (ema_short - ema_long) / ema_long if ema_long != 0 else 0
        base_signal = "NEUTRAL"
        if diff_pct > self.ema_threshold: # Short EMA above Long EMA -> BUY signal (Trend Following)
            base_signal = "BUY"
        elif diff_pct < -self.ema_threshold: # Short EMA below Long EMA -> SELL signal (Trend Following)
            base_signal = "SELL"

        # 2. Momentum Filter (Check recent price slope)
        if len(self.past_prices[product]) >= 5:
            prices_last_5 = self.past_prices[product][-5:]
            # Simplified slope check: current vs 5 periods ago
            price_slope = prices_last_5[-1] - prices_last_5[0] if len(prices_last_5) == 5 else 0

            if base_signal == "BUY" and price_slope < 0: # Don't buy if price is recently falling
                # print("Momentum Filter: BUY signal overridden by falling price.")
                base_signal = "NEUTRAL"
            if base_signal == "SELL" and price_slope > 0: # Don't sell if price is recently rising
                # print("Momentum Filter: SELL signal overridden by rising price.")
                base_signal = "NEUTRAL"

        # 3. Z-Score Check for Strength (Upgrade signal if Z-score confirms extremity)
        final_signal = base_signal
        z_score = self.calculate_z_score(product, mid_price)
        if base_signal == "BUY" and z_score > self.z_score_entry: # Price high relative to mean within uptrend? (Original uses z < -entry) Let's stick to r2's logic first.
             # R2 logic check: if BUY and z < -entry -> STRONG_BUY
             if z_score < -self.z_score_entry:
                  final_signal = "STRONG_BUY"
        elif base_signal == "SELL" and z_score < -self.z_score_entry: # Price low relative to mean within downtrend? (Original uses z > entry)
              # R2 logic check: if SELL and z > entry -> STRONG_SELL
             if z_score > self.z_score_entry:
                  final_signal = "STRONG_SELL"

        # Note: The Z-score check for strength in r2.py seems counter-intuitive for trend following
        # (e.g., requiring low Z for STRONG_BUY in an uptrend). Sticking to r2.py's apparent logic for now.
        # Consider revising this if needed. For now, let's use the base signal mostly.
        # Simplification: Let's just return base_signal after momentum filter for now.
        final_signal = base_signal # Override the strength check for simplicity first

        return final_signal # Return BUY, SELL, or NEUTRAL based on EMA + Momentum


    def calculate_position_size(self, product, signal, current_position, mid_price):
        """Calculates position delta based on r2.py logic (Z-score exits)."""
        position_limit = self.position_limit[product]
        z_score = self.calculate_z_score(product, mid_price) if mid_price is not None else 0

        # --- Explicit Z-Score based Exits (Stop Loss & Profit Taking) ---
        if abs(z_score) > self.z_stop_loss and current_position != 0:
            print(f"STOP LOSS triggered: Z={z_score:.2f}. Exiting position {current_position}")
            return -current_position # Force exit
        if -self.z_score_exit <= z_score <= self.z_score_exit and current_position != 0:
            # If Z-score is near mean, take profit / exit
            print(f"Profit Take / Exit triggered: Z={z_score:.2f}. Exiting position {current_position}")
            return -current_position # Force exit

        # --- Calculate Target Position Based on Signal (if not exited) ---
        target_position = current_position # Default: hold if NEUTRAL and not exited
        if signal == "BUY" or signal == "SELL": # Includes STRONG signals if implemented
            volatility = max(self.volatility.get(product, 0.01), 0.0001)
            volatility_factor = 1 / (1 + volatility * self.volatility_adjustment_factor)
            base_size = math.floor(position_limit * self.risk_factor * volatility_factor)

            # Adjust size based on signal strength (using r2.py approach)
            size_multiplier = 0.7 # Default for BUY/SELL
            # if signal == "STRONG_BUY" or signal == "STRONG_SELL": size_multiplier = 1.0 # Use full base size for strong

            if signal == "BUY": # or signal == "STRONG_BUY":
                target_position = math.floor(base_size * size_multiplier)
            elif signal == "SELL": # or signal == "STRONG_SELL":
                target_position = -math.floor(base_size * size_multiplier)

        elif signal == "NEUTRAL":
             target_position = 0 # Drift towards zero on NEUTRAL if not exited by Z-score


        # Clamp target_position to limits
        target_position = max(-position_limit, min(position_limit, target_position))
        position_delta = target_position - current_position

        # Avoid tiny adjustments
        if abs(position_delta) < 1: return 0

        # Final limit check on delta
        if current_position + position_delta > position_limit: position_delta = position_limit - current_position
        elif current_position + position_delta < -position_limit: position_delta = -position_limit - current_position

        return int(round(position_delta))


    # --- Strategy Execution (Using Passive Limit Order logic from v3) ---
    def create_kelp_orders(self, product: str, state: TradingState, position_delta: int, best_bid: int | None, best_ask: int | None) -> List[Order]:
        """Creates passive limit orders to achieve the position_delta."""
        orders: List[Order] = []
        if abs(position_delta) < 1: return []
        print(f"Attempting to place orders for delta: {position_delta}")

        if position_delta > 0:  # BUY
            price_to_use = best_bid # Place passive limit at best bid
            if price_to_use is None: price_to_use = int(self.ema_short.get(product, DEFAULT_PRICES[product]) - 2)
            orders.append(Order(product, price_to_use, position_delta))
            print(f"Placing BUY LIMIT order: {position_delta}@{price_to_use}")
        elif position_delta < 0:  # SELL
            price_to_use = best_ask # Place passive limit at best ask
            if price_to_use is None: price_to_use = int(self.ema_short.get(product, DEFAULT_PRICES[product]) + 2)
            orders.append(Order(product, price_to_use, position_delta))
            print(f"Placing SELL LIMIT order: {-position_delta}@{price_to_use}")
        return orders

    # --- Main Strategy Logic ---
    def kelp_strategy(self, state: TradingState) -> List[Order]:
        """Orchestrates the KELP strategy."""
        product = KELP

        # --- Cooldown Check ---
        time_since_last_trade = state.timestamp - self.last_trade_time[product]
        if time_since_last_trade < 300: # Cooldown period (e.g., 300 timestamps)
            # print(f"Cooldown active for KELP ({time_since_last_trade} < 300)")
            return []

        current_position = self.get_position(product, state)
        best_bid, best_ask, mid_price = self.get_bid_ask_mid_price(product, state)

        # Update indicators
        self.update_price_history(product, mid_price)
        self.update_emas(product, mid_price)
        self.update_z_score_stats(product, mid_price)

        # Get signal & desired position change
        signal = self.identify_signal(product, mid_price, state) # Pass state if needed by signal logic
        position_delta = self.calculate_position_size(product, signal, current_position, mid_price) # Pass mid_price

        # --- Logging ---
        if self.round % 5 == 0 or position_delta != 0 :
             z_score = self.calculate_z_score(product, mid_price)
             print(f"--- KELP --- Rd:{self.round} Pos:{current_position} Delta:{position_delta} Signal:{signal} Z:{z_score:.2f}")

        # --- Create Orders ---
        if abs(position_delta) >= 1:
            orders = self.create_kelp_orders(product, state, position_delta, best_bid, best_ask)
            if orders: # If orders were actually created
                 self.last_trade_time[product] = state.timestamp # Update last trade time
            return orders
        else:
            return []


    # --- PnL & Main Run Method (Unchanged PnL logic) ---
    def update_pnl(self, state: TradingState):
        self.cash = getattr(self, 'cash', 0) # Ensure cash exists
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp >= state.timestamp - 100:
                    if trade.buyer == SUBMISSION: self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION: self.cash += trade.quantity * trade.price
        position_value = 0
        for product in state.position:
            _, _, mid_price = self.get_bid_ask_mid_price(product, state)
            fallback_price = self.ema_short.get(product, DEFAULT_PRICES.get(product, 0)) if self.ema_short else DEFAULT_PRICES.get(product, 0)
            position_value += state.position[product] * (mid_price if mid_price is not None else fallback_price)
        return self.cash + position_value

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int | None, str | None]:
        self.round += 1
        final_orders: Dict[str, List[Order]] = {}

        # Initialize cash if first round
        if self.round == 1: self.cash = 0

        # Update indicators needed for ALL products
        for prod in PRODUCTS:
             _, _, current_mid = self.get_bid_ask_mid_price(prod, state)
             self.update_emas(prod, current_mid) # Keep EMAs updated for fallback prices

        # Execute KELP Strategy
        try:
            kelp_orders = self.kelp_strategy(state)
            if kelp_orders:
                final_orders[KELP] = kelp_orders
        except Exception as e:
            print(f"!!! ERROR in KELP strategy execution: {e} !!!")
            import traceback
            traceback.print_exc()

        # Log Trades and PnL periodically
        log_this_round = (self.round % 20 == 0 or bool(final_orders)) # Log every 20 rounds or if orders placed

        if log_this_round:
            print(f"--- Round {self.round} Log ---")
            print("OWN TRADES (Last Interval):")
            trade_occurred = False
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp >= state.timestamp - 100:
                        print(f"  {trade}")
                        trade_occurred = True
            if not trade_occurred: print("  (None)")

            current_pnl = self.update_pnl(state) # Update cash from trades then calc PnL
            print(f"CASH: {self.cash:.2f}")
            for product in PRODUCTS:
                 pos = self.get_position(product, state)
                 _, _, mid = self.get_bid_ask_mid_price(product, state)
                 mid_str = f'{mid:.1f}' if mid is not None else 'N/A'
                 fallback_price = self.ema_short.get(product, DEFAULT_PRICES.get(product, 0)) if self.ema_short else DEFAULT_PRICES.get(product, 0)
                 val = pos * (mid if mid is not None else fallback_price)
                 print(f"  {product}: Pos={pos}, Mid={mid_str}, Val={val:.0f}")
            print(f"Est. PNL: {current_pnl:.2f}")


        conversions = None
        traderData = None
        if not isinstance(final_orders, dict): final_orders = {}
        return final_orders, conversions, traderData
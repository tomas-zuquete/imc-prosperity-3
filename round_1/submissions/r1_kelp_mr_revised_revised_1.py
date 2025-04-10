import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState, OrderDepth, UserId

SUBMISSION = "SUBMISSION"
KELP = "KELP"

PRODUCTS = [KELP]
DEFAULT_PRICES = {KELP: 2030}

class Trader:
    def __init__(self) -> None:
        print("Initializing Optimized Trader...")

        self.position_limit = {KELP: 50}
        self.round = 0
        self.cash = 0

        self.past_prices = {product: [] for product in PRODUCTS}
        self.ema_short = {product: None for product in PRODUCTS}
        self.ema_long = {product: None for product in PRODUCTS}
        self.ema_short_param = 2 / (5 + 1)
        self.ema_long_param = 2 / (20 + 1)

        self.z_score_window = 20
        self.rolling_mean = {KELP: None}
        self.rolling_std = {KELP: None}

        self.peak_hours = [3, 10, 13]

        self.z_score_entry = 0.6
        self.z_score_exit = 0.2
        self.z_stop_loss = 1.5
        self.ema_threshold = 0.0005
        self.risk_factor = 0.4

        self.current_hour = {KELP: 0}
        self.volatility = {KELP: 0.01}
        self.initial_trades_done = False
        self.last_trade_time = {KELP: -10}

    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    def get_bid_ask_mid_price(self, product, state: TradingState):
        if product not in state.order_depths or not state.order_depths[product].buy_orders or not state.order_depths[product].sell_orders:
            return None, None, self.ema_short[product] or DEFAULT_PRICES[product]

        best_bid = max(state.order_depths[product].buy_orders.keys())
        best_ask = min(state.order_depths[product].sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        return best_bid, best_ask, mid_price

    def update_emas(self, product, mid_price):
        if mid_price is None:
            return
        if self.ema_short[product] is None:
            self.ema_short[product] = mid_price
        else:
            self.ema_short[product] = mid_price * self.ema_short_param + (1 - self.ema_short_param) * self.ema_short[product]

        if self.ema_long[product] is None:
            self.ema_long[product] = mid_price
        else:
            self.ema_long[product] = mid_price * self.ema_long_param + (1 - self.ema_long_param) * self.ema_long[product]

    def update_price_history(self, product, mid_price):
        if mid_price is None:
            return
        self.past_prices[product].append(mid_price)
        if len(self.past_prices[product]) > 50:
            self.past_prices[product] = self.past_prices[product][-50:]

        if len(self.past_prices[product]) >= 5:
            prices = self.past_prices[product][-5:]
            returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            self.volatility[product] = max(math.sqrt(variance), 0.001)

    def update_z_score(self, product, mid_price):
        if mid_price is None:
            return
        window = min(len(self.past_prices[product]), self.z_score_window)
        if window < 5:
            return
        prices = self.past_prices[product][-window:]
        self.rolling_mean[product] = sum(prices) / len(prices)
        variance = sum((p - self.rolling_mean[product]) ** 2 for p in prices) / len(prices)
        self.rolling_std[product] = max(math.sqrt(variance), 0.1)

    def calculate_z_score(self, product, mid_price):
        if self.rolling_mean[product] is None or self.rolling_std[product] is None or mid_price is None:
            return 0
        return (mid_price - self.rolling_mean[product]) / self.rolling_std[product]

    def update_hour(self, product, timestamp):
        hour = (timestamp % 86400) // 3600
        self.current_hour[product] = hour

    def identify_signal(self, product, mid_price):
        if len(self.past_prices[product]) < 10 and not self.initial_trades_done:
            return "INITIAL_BUY"
        if self.ema_short[product] is None or self.ema_long[product] is None:
            return "NEUTRAL"

        diff_pct = (self.ema_short[product] - self.ema_long[product]) / self.ema_long[product]
        z_score = self.calculate_z_score(product, mid_price)

        signal = "NEUTRAL"
        if diff_pct < -self.ema_threshold:
            signal = "BUY"
        elif diff_pct > self.ema_threshold:
            signal = "SELL"

        if len(self.past_prices[product]) >= 5:
            price_slope = self.past_prices[product][-1] - self.past_prices[product][-5]
            if signal == "BUY" and price_slope < 0:
                return "NEUTRAL"
            if signal == "SELL" and price_slope > 0:
                return "NEUTRAL"

        if signal == "BUY" and z_score < -self.z_score_entry:
            return "STRONG_BUY"
        elif signal == "SELL" and z_score > self.z_score_entry:
            return "STRONG_SELL"
        return signal

    def calculate_position_size(self, product, signal, current_position, mid_price):
        position_limit = self.position_limit[product]
        z_score = self.calculate_z_score(product, mid_price)

        if signal == "INITIAL_BUY":
            self.initial_trades_done = True
            return min(10, position_limit - current_position)

        if abs(z_score) > self.z_stop_loss:
            print(f"STOP LOSS triggered: Z-score = {z_score}")
            return -current_position

        if -self.z_score_exit <= z_score <= self.z_score_exit and current_position != 0:
            print("Taking profit near mean.")
            return -current_position

        volatility_factor = 1 / (1 + self.volatility[product] * 5)
        hour_factor = 1.2 if self.current_hour[product] in self.peak_hours else 1.0
        base_size = math.floor(position_limit * self.risk_factor * volatility_factor * hour_factor)

        if signal == "STRONG_BUY":
            target_position = base_size
        elif signal == "BUY":
            target_position = math.floor(base_size * 0.7)
        elif signal == "STRONG_SELL":
            target_position = -base_size
        elif signal == "SELL":
            target_position = -math.floor(base_size * 0.7)
        else:
            target_position = 0

        target_position = max(-position_limit, min(position_limit, target_position))
        return target_position - current_position

    def kelp_strategy(self, state: TradingState):
        product = KELP
        if state.timestamp - self.last_trade_time[product] < 500:
            print("Trade cooldown active.")
            return []

        current_position = self.get_position(product, state)
        best_bid, best_ask, mid_price = self.get_bid_ask_mid_price(product, state)
        if mid_price is None:
            return []

        self.update_price_history(product, mid_price)
        self.update_emas(product, mid_price)
        self.update_z_score(product, mid_price)
        self.update_hour(product, state.timestamp)

        signal = self.identify_signal(product, mid_price)
        position_delta = self.calculate_position_size(product, signal, current_position, mid_price)
        if position_delta == 0:
            return []

        self.last_trade_time[product] = state.timestamp
        orders = []

        if position_delta > 0:
            available_sells = sorted(state.order_depths[product].sell_orders.items())
            remaining_to_buy = position_delta
            for price, volume in available_sells:
                buy_quantity = min(remaining_to_buy, -volume)
                if buy_quantity > 0:
                    orders.append(Order(product, price, buy_quantity))
                    remaining_to_buy -= buy_quantity
                if remaining_to_buy <= 0:
                    break
            if remaining_to_buy > 0 and best_bid is not None:
                orders.append(Order(product, best_bid + 1, remaining_to_buy))

        elif position_delta < 0:
            available_buys = sorted(state.order_depths[product].buy_orders.items(), reverse=True)
            remaining_to_sell = -position_delta
            for price, volume in available_buys:
                sell_quantity = min(remaining_to_sell, volume)
                if sell_quantity > 0:
                    orders.append(Order(product, price, -sell_quantity))
                    remaining_to_sell -= sell_quantity
                if remaining_to_sell <= 0:
                    break
            if remaining_to_sell > 0 and best_ask is not None:
                orders.append(Order(product, best_ask - 1, -remaining_to_sell))

        return orders

    def update_pnl(self, state: TradingState):
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
                _, _, mid_price = self.get_bid_ask_mid_price(product, state)
                if mid_price is not None:
                    value += state.position[product] * mid_price
            return value

        update_cash()
        return self.cash + get_value_on_positions()

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Any, Any]:
        self.round += 1
        pnl = self.update_pnl(state)

        print(f"Round {self.round}, PnL: {pnl}")

        result = {}
        try:
            result[KELP] = self.kelp_strategy(state)
        except Exception as e:
            print(f"Error in KELP strategy: {e}")

        return result, 0, None

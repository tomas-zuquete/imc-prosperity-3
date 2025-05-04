from round_5.sub.datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import numpy as np
import jsonpickle

class Trader:

    def __init__(self):
        # Use dictionary to maintain rolling windows and position state
        self.rolling_window = {
            "KELP": {
                "prices": [],
                "window_size": 100
            }
        }

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = state.traderData

        for product in state.order_depths:
            if product != "KELP":
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Determine mid price
            best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None

            if best_ask is None or best_bid is None:
                result[product] = orders
                continue

            mid_price = (best_ask + best_bid) / 2

            # Update rolling price window
            window = self.rolling_window[product]["prices"]
            window.append(mid_price)
            if len(window) > self.rolling_window[product]["window_size"]:
                window.pop(0)

            if len(window) < 10:
                result[product] = orders
                continue

            # Calculate rolling mean and std dev
            rolling_mean = np.mean(window)
            rolling_std = np.std(window)
            z_score = (mid_price - rolling_mean) / rolling_std if rolling_std > 0 else 0

            position_limit = 100  # Assuming a default position limit
            position = state.position.get(product, 0)

            volume = 10  # can be dynamically adjusted later

            # Entry logic
            if z_score < -1 and position + volume <= position_limit:
                # BUY signal
                orders.append(Order(product, best_ask, volume))

            elif z_score > 1 and position - volume >= -position_limit:
                # SELL signal
                orders.append(Order(product, best_bid, -volume))

            # Exit logic (mean reversion to zero)
            elif -0.5 < z_score < 0 and position > 0:
                # Exit long
                orders.append(Order(product, best_bid, -position))

            elif 0 < z_score < 0.5 and position < 0:
                # Exit short
                orders.append(Order(product, best_ask, -position))

            result[product] = orders

        # Serialize rolling window using jsonpickle
        traderData = jsonpickle.encode(self.rolling_window)
        return result, conversions, traderData

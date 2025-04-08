from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import statistics

class Trader:
    def run(self, state: TradingState):
        # Initialize the result dict
        result = {}
        
        # Store state data between runs
        if state.traderData == "":
            trader_data = {}
        else:
            trader_data = json.loads(state.traderData)
        
        # Process each product
        for product in state.order_depths:
            # Get order depth for this product
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            position_limit = 20  # adjust this per product
            
            # Calculate fair value (mid price)
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                fair_value = (best_bid + best_ask) / 2
            else:
                # If we don't have both sides, use history or skip
                fair_value = trader_data.get(f"{product}_fair_value", None)
                if fair_value is None:
                    continue
            
            # Store the fair value for next iteration
            trader_data[f"{product}_fair_value"] = fair_value
            
            # Create orders list
            orders: List[Order] = []
            
            # Calculate remaining position capacity
            buy_capacity = position_limit - position
            sell_capacity = position_limit + position
            
            # Place buy orders (take advantage of prices below fair value)
            if len(order_depth.sell_orders) > 0:
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price < fair_value * 0.99:  # Buy if price is at least 1% below fair value
                        volume_to_buy = min(-ask_volume, buy_capacity)
                        if volume_to_buy > 0:
                            orders.append(Order(product, ask_price, volume_to_buy))
                            buy_capacity -= volume_to_buy
            
            # Place sell orders (take advantage of prices above fair value)
            if len(order_depth.buy_orders) > 0:
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > fair_value * 1.01:  # Sell if price is at least 1% above fair value
                        volume_to_sell = min(bid_volume, sell_capacity)
                        if volume_to_sell > 0:
                            orders.append(Order(product, bid_price, -volume_to_sell))
                            sell_capacity -= volume_to_sell
            
            # Only add non-empty orders to the result
            if orders:
                result[product] = orders
        
        # Return the result along with conversions and updated trader data
        return result, 0, json.dumps(trader_data)
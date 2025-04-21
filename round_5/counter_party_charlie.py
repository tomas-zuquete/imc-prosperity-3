from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle

class Trader:
    def run(self, state: TradingState):
        # Initialize or retrieve trader data
        if state.traderData == "":
            trader_data = {
                "charlie_buy_prices": [],
                "charlie_sell_prices": [],
                "last_mid_price": None,
                "position": 0,
                "trades_with_charlie": 0
            }
        else:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {
                    "charlie_buy_prices": [],
                    "charlie_sell_prices": [],
                    "last_mid_price": None,
                    "position": 0,
                    "trades_with_charlie": 0
                }
        
        # Update position from state
        if "PICNIC_BASKET1" in state.position:
            trader_data["position"] = state.position["PICNIC_BASKET1"]
        
        # Process recent trades to gather counterparty data
        if "PICNIC_BASKET1" in state.own_trades and state.own_trades["PICNIC_BASKET1"]:
            for trade in state.own_trades["PICNIC_BASKET1"]:
                counterparty = trade.counter_party if hasattr(trade, 'counter_party') else "Unknown"
                
                if counterparty == "Charlie":
                    trader_data["trades_with_charlie"] += 1
                    if trade.quantity > 0:  # We bought from Charlie
                        trader_data["charlie_sell_prices"].append(trade.price)
                    else:  # We sold to Charlie
                        trader_data["charlie_buy_prices"].append(trade.price)
        
        # Initialize result dictionary
        result = {}
        
        # Only generate orders for PICNIC_BASKET1
        target_product = "PICNIC_BASKET1"
        
        # Skip if the product is not in order depths
        if target_product not in state.order_depths:
            result[target_product] = []
            return result, 0, jsonpickle.encode(trader_data)
        
        # Get order depth for PICNIC_BASKET1
        order_depth = state.order_depths[target_product]
        
        # Skip if empty order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            result[target_product] = []
            return result, 0, jsonpickle.encode(trader_data)
        
        # Calculate fair value
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Store for next iteration
        trader_data["last_mid_price"] = mid_price
        
        # Position limit for PICNIC_BASKET1
        position_limit = 20
        current_position = trader_data["position"]
        
        # Calculate remaining capacity
        max_long_capacity = position_limit - current_position
        max_short_capacity = position_limit + current_position
        
        # Generate orders list
        orders = []
        
        # Charlie-specific thresholds based on analysis
        # Charlie buys at 58412.94 and sells at 58350.33 - extremely profitable!
        charlie_buy_threshold = 58400  # Price at which Charlie tends to buy
        charlie_sell_threshold = 58360  # Price at which Charlie tends to sell
        
        # Calculate average Charlie prices if we have data
        if trader_data["charlie_buy_prices"]:
            avg_charlie_buy = sum(trader_data["charlie_buy_prices"]) / len(trader_data["charlie_buy_prices"])
            charlie_buy_threshold = avg_charlie_buy
        
        if trader_data["charlie_sell_prices"]:
            avg_charlie_sell = sum(trader_data["charlie_sell_prices"]) / len(trader_data["charlie_sell_prices"])
            charlie_sell_threshold = avg_charlie_sell
        
        # Buy strategy - look for prices below Charlie's selling threshold
        if max_long_capacity > 0:  # Only buy if we have capacity
            sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sell_prices:
                if price <= charlie_sell_threshold * 1.0001:  # Just slightly above Charlie's selling price
                    # This is a good buying opportunity - Charlie will likely buy higher
                    volume = abs(order_depth.sell_orders[price])
                    
                    # Determine buy quantity based on how good the price is
                    if price < charlie_sell_threshold * 0.999:  # Very good price
                        buy_quantity = min(volume, max_long_capacity, 15)  # Be aggressive
                    else:  # Good price
                        buy_quantity = min(volume, max_long_capacity, 10)  # Standard buy
                    
                    if buy_quantity > 0:
                        orders.append(Order(target_product, price, buy_quantity))
                        max_long_capacity -= buy_quantity  # Update remaining capacity
        
        # Sell strategy - look for prices above Charlie's buying threshold
        if max_short_capacity > 0:  # Only sell if we have capacity
            buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in buy_prices:
                if price >= charlie_buy_threshold * 0.9999:  # Just slightly below Charlie's buying price
                    # This is a good selling opportunity - Charlie will likely sell lower
                    volume = order_depth.buy_orders[price]
                    
                    # Determine sell quantity based on how good the price is
                    if price > charlie_buy_threshold * 1.001:  # Very good price
                        sell_quantity = min(volume, max_short_capacity, 15)  # Be aggressive
                    else:  # Good price
                        sell_quantity = min(volume, max_short_capacity, 10)  # Standard sell
                    
                    if sell_quantity > 0:
                        orders.append(Order(target_product, price, -sell_quantity))
                        max_short_capacity -= sell_quantity  # Update remaining capacity
        
        # Add orders to result
        result[target_product] = orders
        
        # For all other products, return empty order lists
        for product in state.order_depths:
            if product != target_product:
                result[product] = []
        
        return result, 0, jsonpickle.encode(trader_data)
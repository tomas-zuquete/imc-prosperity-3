from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np
import jsonpickle

class Trader:
    def __init__(self):
        # Initialize position limits
        self.position_limits = {
            "KELP": 20,
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            "BERRIES": 250,
            "DIVING_GEAR": 50,
            "DOLPHIN_SIGHTINGS": 10,
            "STARFRUIT": 20,
            "SHELL_NECKLACE": 70,
            "SQUID_INK": 300,
            "RAINFOREST_RESIN": 60,
        }
        
        # Market making parameters
        self.target_position = 0  # We aim to stay neutral
        self.fair_values = {}     # Our estimate of each product's fair value
        self.position = {}        # Current positions
        self.position_history = {}  # Track position changes
        self.price_history = {}   # Track price history
        
    def run(self, state: TradingState):
        """
        Simple market making strategy that buys at bid and sells at ask
        """
        # Initialize the result dictionary
        result = {}
        
        # Load state from previous iteration if available
        if state.traderData:
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.fair_values = saved_state.get("fair_values", {})
                self.position_history = saved_state.get("position_history", {})
                self.price_history = saved_state.get("price_history", {})
            except Exception as e:
                print(f"Error loading state: {e}")
        
        # Update positions from state
        self.position = state.position
        
        # Process each product
        for product in state.order_depths.keys():
            # Skip products we're not interested in (focus on KELP only for now)
            if product != "RAINFOREST_RESIN":
                continue
            
            # Get order depth for this product
            order_depth = state.order_depths[product]
            
            # Initialize orders list
            orders: List[Order] = []
            
            # Initialize product history if needed
            if product not in self.position_history:
                self.position_history[product] = []
            
            if product not in self.price_history:
                self.price_history[product] = []
            
            # Current position
            current_position = self.position.get(product, 0)
            
            # Record position
            self.position_history[product].append(current_position)
            
            # Extract market data
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                # Get best bid and ask
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                
                # Record mid price
                mid_price = (best_bid + best_ask) / 2
                self.price_history[product].append(mid_price)
                
                # Keep price history limited
                if len(self.price_history[product]) > 100:
                    self.price_history[product] = self.price_history[product][-100:]
                
                # Calculate fair value using simple average of recent prices
                if len(self.price_history[product]) > 0:
                    self.fair_values[product] = np.mean(self.price_history[product][-10:])
                else:
                    self.fair_values[product] = mid_price
                
                # Calculate spread
                spread = best_ask - best_bid
                
                # Position limits
                position_limit = self.position_limits.get(product, 0)
                
                # Strategy: Pure market making - always try to buy at bid and sell at ask
                
                # Calculate position adjustment based on current position
                if current_position > self.target_position:
                    # We're net long, focus on selling
                    sell_size = min(position_limit + self.target_position, abs(current_position - self.target_position))
                    buy_size = max(0, position_limit - current_position)
                elif current_position < self.target_position:
                    # We're net short, focus on buying
                    buy_size = min(position_limit - self.target_position, abs(current_position - self.target_position))
                    sell_size = max(0, position_limit + current_position)
                else:
                    # We're neutral, equal focus
                    buy_size = position_limit
                    sell_size = position_limit
                
                # Adjust buy and sell sizes based on available liquidity
                if len(order_depth.buy_orders) > 0:
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    sell_size = min(sell_size, best_bid_volume)
                else:
                    sell_size = 0
                
                if len(order_depth.sell_orders) > 0:
                    best_ask_volume = abs(order_depth.sell_orders[best_ask])
                    buy_size = min(buy_size, best_ask_volume)
                else:
                    buy_size = 0
                
                # Execute orders
                if sell_size > 0:
                    # Sell at best bid
                    orders.append(Order(product, best_bid, -sell_size))
                    print(f"SELL {sell_size}x {product} @ {best_bid}")
                
                if buy_size > 0:
                    # Buy at best ask
                    orders.append(Order(product, best_ask, buy_size))
                    print(f"BUY {buy_size}x {product} @ {best_ask}")
            
            # Add orders to result
            result[product] = orders
        
        # Save state for next iteration
        traderData = jsonpickle.encode({
            "fair_values": self.fair_values,
            "position_history": self.position_history,
            "price_history": self.price_history
        })
        
        return result, 0, traderData
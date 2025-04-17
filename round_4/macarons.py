from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math
import json
import jsonpickle

class Trader:
    def __init__(self):
        self.product_positions = {}
        self.fair_value_history = {}
        self.mid_price_history = {}
        self.position_history = {}
        self.last_observations = {}

    def run(self, state: TradingState):
        # Initialize or restore the state
        if state.traderData != "":
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.product_positions = saved_state['product_positions']
                self.fair_value_history = saved_state['fair_value_history']
                self.mid_price_history = saved_state['mid_price_history']
                self.position_history = saved_state['position_history']
                self.last_observations = saved_state['last_observations']
            except Exception as e:
                print(f"Error restoring state: {e}")
                # Initialize if restoration fails
                self.product_positions = {}
                self.fair_value_history = {}
                self.mid_price_history = {}
                self.position_history = {}
                self.last_observations = {}

        # Record current positions
        for product, pos in state.position.items():
            self.product_positions[product] = pos
            
            # Initialize history for new products
            if product not in self.fair_value_history:
                self.fair_value_history[product] = []
            if product not in self.mid_price_history:
                self.mid_price_history[product] = []
            if product not in self.position_history:
                self.position_history[product] = []
            
            # Track position history
            self.position_history[product].append(pos)
        
        # Store the observations for use in fair value calculation
        if state.observations:
            if hasattr(state.observations, 'conversionObservations'):
                for product, obs in state.observations.conversionObservations.items():
                    self.last_observations[product] = obs
        
        # Orders to be placed on exchange matching engine
        result = {}
        conversions = {}
        
        # Process each product
        for product in state.order_depths:
            # For now, we'll focus specifically on MAGNIFICENT_MACARONS
            if product == "MAGNIFICENT_MACARONS":
                orders = self.trade_magnificent_macarons(
                    product, 
                    state.order_depths[product], 
                    state.position.get(product, 0),
                    state.observations
                )
                result[product] = orders
            else:
                # For other products, initialize empty order list
                result[product] = []
        
        # Save the state
        trader_data = {
            'product_positions': self.product_positions,
            'fair_value_history': self.fair_value_history,
            'mid_price_history': self.mid_price_history,
            'position_history': self.position_history,
            'last_observations': self.last_observations
        }
        
        traderData = jsonpickle.encode(trader_data)
        
        # For MAGNIFICENT_MACARONS, determine if conversion is needed
        # We'll set this as 0 by default since our analysis showed no profitable conversion opportunities
        conversion_count = 0
        
        return result, conversion_count, traderData
    
    def trade_magnificent_macarons(self, product, order_depth, position, observations):
        """
        Strategy for trading MAGNIFICENT_MACARONS based on observations and order book
        """
        orders: List[Order] = []
        
        # Get the observations for MAGNIFICENT_MACARONS
        obs = None
        if hasattr(observations, 'conversionObservations') and product in observations.conversionObservations:
            obs = observations.conversionObservations[product]
            self.last_observations[product] = obs
        elif product in self.last_observations:
            obs = self.last_observations[product]
            
        # Position limit for MAGNIFICENT_MACARONS is 75
        POSITION_LIMIT = 75
        
        # Calculate mid price from the order book
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price:
            self.mid_price_history[product].append(mid_price)
        
        # Calculate fair value based on observations and historical data
        fair_value = self.calculate_fair_value_macarons(order_depth, obs, mid_price)
        if fair_value:
            self.fair_value_history[product].append(fair_value)
        else:
            # If we can't calculate fair value, use mid price as fair value
            fair_value = mid_price
            
        # If we don't have a fair value, we can't trade
        if not fair_value:
            return orders
            
        # Get highest bid and lowest ask
        best_bid, best_bid_amount = self.get_best_bid(order_depth)
        best_ask, best_ask_amount = self.get_best_ask(order_depth)
        
        # If there are no orders on one side, we can't calculate a reasonable fair value
        if not best_bid or not best_ask:
            return orders
            
        # Calculate bid-ask spread
        spread = best_ask - best_bid
        
        # Trading logic based on fair value
        # We'll add a small threshold to reduce excessive trading
        threshold = 0.5
        
        # Sell if price is above fair value (plus threshold)
        if best_bid > fair_value + threshold and position > -POSITION_LIMIT:
            # Calculate how many we can sell
            sell_quantity = min(best_bid_amount, POSITION_LIMIT + position)
            orders.append(Order(product, best_bid, -sell_quantity))
            
            # Update position
            position -= sell_quantity
            
        # Buy if price is below fair value (minus threshold)
        if best_ask < fair_value - threshold and position < POSITION_LIMIT:
            # Calculate how many we can buy
            buy_quantity = min(-best_ask_amount, POSITION_LIMIT - position)
            orders.append(Order(product, best_ask, buy_quantity))
            
            # Update position
            position += buy_quantity
        
        # Update our position tracking
        self.product_positions[product] = position
        
        return orders
    
    def calculate_fair_value_macarons(self, order_depth, observations, mid_price):
        """
        Calculate fair value for MAGNIFICENT_MACARONS based on observations and market data
        """
        if not observations:
            # If no observations, use mid price if available
            return mid_price
            
        # Based on our correlation analysis, we found these weights work well
        # bidPrice and askPrice had the strongest correlations (0.999)
        # followed by sunlightIndex (0.921), sugarPrice (0.856), and exportTariff (0.719)
        weights = {
            'bidPrice': 0.35,
            'askPrice': 0.35,
            'sunlightIndex': 0.15,
            'sugarPrice': 0.10,
            'exportTariff': 0.05
        }
        
        # Calculate weighted fair value
        fair_value = 0
        weight_sum = 0
        
        if hasattr(observations, 'bidPrice') and hasattr(observations, 'askPrice'):
            fair_value += observations.bidPrice * weights['bidPrice']
            fair_value += observations.askPrice * weights['askPrice']
            weight_sum += weights['bidPrice'] + weights['askPrice']
            
        if hasattr(observations, 'sunlightIndex'):
            # We found sunlightIndex has a strong correlation with price (0.921)
            # Each point in sunlightIndex correlates to ~10-11 points in price
            fair_value += observations.sunlightIndex * 10.5 * weights['sunlightIndex']
            weight_sum += weights['sunlightIndex']
            
        if hasattr(observations, 'sugarPrice'):
            # sugarPrice has a correlation of 0.856 with price
            # Each point in sugarPrice correlates to ~3-3.5 points in price
            fair_value += observations.sugarPrice * 3.2 * weights['sugarPrice']
            weight_sum += weights['sugarPrice']
            
        if hasattr(observations, 'exportTariff'):
            # exportTariff has a correlation of 0.719 with price
            # Each point in exportTariff correlates to ~7-8 points in price
            fair_value += observations.exportTariff * 7.5 * weights['exportTariff']
            weight_sum += weights['exportTariff']
        
        # If we couldn't calculate any components of fair value, return mid price
        if weight_sum == 0:
            return mid_price
            
        # Normalize the fair value by dividing by the sum of weights used
        fair_value /= weight_sum
        
        # If we have historical fair values, smooth it with a moving average
        if 'MAGNIFICENT_MACARONS' in self.fair_value_history and len(self.fair_value_history['MAGNIFICENT_MACARONS']) > 0:
            # Calculate a weighted average of current and historical fair values
            # More weight to current calculation, some weight to history for stability
            historical_avg = sum(self.fair_value_history['MAGNIFICENT_MACARONS'][-5:]) / min(5, len(self.fair_value_history['MAGNIFICENT_MACARONS']))
            fair_value = 0.7 * fair_value + 0.3 * historical_avg
        
        return fair_value
    
    def calculate_mid_price(self, order_depth):
        """Calculate the mid price from the order book"""
        best_bid, _ = self.get_best_bid(order_depth)
        best_ask, _ = self.get_best_ask(order_depth)
        
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        elif best_bid:
            return best_bid
        elif best_ask:
            return best_ask
        else:
            return None
    
    def get_best_bid(self, order_depth):
        """Get the highest bid price and quantity"""
        if not order_depth.buy_orders:
            return None, None
            
        best_bid = max(order_depth.buy_orders.keys())
        return best_bid, order_depth.buy_orders[best_bid]
    
    def get_best_ask(self, order_depth):
        """Get the lowest ask price and quantity"""
        if not order_depth.sell_orders:
            return None, None
            
        best_ask = min(order_depth.sell_orders.keys())
        return best_ask, order_depth.sell_orders[best_ask]
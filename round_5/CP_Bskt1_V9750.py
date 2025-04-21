from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle

class Trader:
    def run(self, state: TradingState):
        # Initialize or retrieve trader data
        if state.traderData == "":
            trader_data = {
                # Product 1: PICNIC_BASKET1
                "PICNIC_BASKET1": {
                    "charlie_buy_prices": [],
                    "charlie_sell_prices": [],
                    "last_mid_price": None,
                    "position": 0,
                    "market_volatility": 0,
                    "recent_prices": [],
                    "last_trades": [],
                    "average_position_price": 0
                },
                # Product 2: VOLCANIC_ROCK_VOUCHER_9750
                "VOLCANIC_ROCK_VOUCHER_9750": {
                    "penelope_sell_prices": [],
                    "pablo_buy_prices": [],
                    "last_mid_price": None,
                    "position": 0,
                    "market_volatility": 0,
                    "recent_prices": [],
                    "last_trades": [],
                    "average_position_price": 0
                },
                "iteration": 0
            }
        else:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {
                    # Product 1: PICNIC_BASKET1
                    "PICNIC_BASKET1": {
                        "charlie_buy_prices": [],
                        "charlie_sell_prices": [],
                        "last_mid_price": None,
                        "position": 0,
                        "market_volatility": 0,
                        "recent_prices": [],
                        "last_trades": [],
                        "average_position_price": 0
                    },
                    # Product 2: VOLCANIC_ROCK_VOUCHER_9750
                    "VOLCANIC_ROCK_VOUCHER_9750": {
                        "penelope_sell_prices": [],
                        "pablo_buy_prices": [],
                        "last_mid_price": None,
                        "position": 0,
                        "market_volatility": 0,
                        "recent_prices": [],
                        "last_trades": [],
                        "average_position_price": 0
                    },
                    "iteration": 0
                }
        
        trader_data["iteration"] += 1
        
        # Update positions from state
        if "PICNIC_BASKET1" in state.position:
            trader_data["PICNIC_BASKET1"]["position"] = state.position["PICNIC_BASKET1"]
        
        if "VOLCANIC_ROCK_VOUCHER_9750" in state.position:
            trader_data["VOLCANIC_ROCK_VOUCHER_9750"]["position"] = state.position["VOLCANIC_ROCK_VOUCHER_9750"]
        
        # Process recent trades to gather counterparty data
        self.update_trade_data(state, trader_data)
        
        # Initialize result dictionary
        result = {}
        
        # Generate orders for PICNIC_BASKET1
        # picnic_orders = self.generate_picnic_basket_orders(state, trader_data["PICNIC_BASKET1"])
        # if picnic_orders:
        #     result["PICNIC_BASKET1"] = picnic_orders
        
        # Generate orders for VOLCANIC_ROCK_VOUCHER_9750
        voucher_orders = self.generate_voucher_orders(state, trader_data["VOLCANIC_ROCK_VOUCHER_9750"])
        if voucher_orders:
            result["VOLCANIC_ROCK_VOUCHER_9750"] = voucher_orders
        
        # For all other products, return empty order lists
        for product in state.order_depths:
            if product not in result:
                result[product] = []
        
        return result, 0, jsonpickle.encode(trader_data)
    
    def update_trade_data(self, state, trader_data):
        """Update trader data with recent trade information"""
        
        # Process PICNIC_BASKET1 trades
        if "PICNIC_BASKET1" in state.own_trades and state.own_trades["PICNIC_BASKET1"]:
            for trade in state.own_trades["PICNIC_BASKET1"]:
                counterparty = trade.counter_party if hasattr(trade, 'counter_party') else "Unknown"
                
                if counterparty == "Charlie":
                    if trade.quantity > 0:  # We bought from Charlie
                        trader_data["PICNIC_BASKET1"]["charlie_sell_prices"].append(trade.price)
                    else:  # We sold to Charlie
                        trader_data["PICNIC_BASKET1"]["charlie_buy_prices"].append(trade.price)
                
                # Track last 10 trades
                trader_data["PICNIC_BASKET1"]["last_trades"].append({
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "counterparty": counterparty
                })
                
                if len(trader_data["PICNIC_BASKET1"]["last_trades"]) > 10:
                    trader_data["PICNIC_BASKET1"]["last_trades"] = trader_data["PICNIC_BASKET1"]["last_trades"][-10:]
        
        # Process VOLCANIC_ROCK_VOUCHER_9750 trades
        if "VOLCANIC_ROCK_VOUCHER_9750" in state.own_trades and state.own_trades["VOLCANIC_ROCK_VOUCHER_9750"]:
            for trade in state.own_trades["VOLCANIC_ROCK_VOUCHER_9750"]:
                counterparty = trade.counter_party if hasattr(trade, 'counter_party') else "Unknown"
                
                if counterparty == "Penelope" and trade.quantity > 0:  # We bought from Penelope
                    trader_data["VOLCANIC_ROCK_VOUCHER_9750"]["penelope_sell_prices"].append(trade.price)
                elif counterparty == "Pablo" and trade.quantity < 0:  # We sold to Pablo
                    trader_data["VOLCANIC_ROCK_VOUCHER_9750"]["pablo_buy_prices"].append(trade.price)
                
                # Track last 10 trades
                trader_data["VOLCANIC_ROCK_VOUCHER_9750"]["last_trades"].append({
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "counterparty": counterparty
                })
                
                if len(trader_data["VOLCANIC_ROCK_VOUCHER_9750"]["last_trades"]) > 10:
                    trader_data["VOLCANIC_ROCK_VOUCHER_9750"]["last_trades"] = trader_data["VOLCANIC_ROCK_VOUCHER_9750"]["last_trades"][-10:]
    
    def generate_picnic_basket_orders(self, state, product_data):
        """Generate orders for PICNIC_BASKET1 focusing on Charlie"""
        target_product = "PICNIC_BASKET1"
        
        # Skip if the product is not in order depths
        if target_product not in state.order_depths:
            return []
        
        # Get order depth for PICNIC_BASKET1
        order_depth = state.order_depths[target_product]
        
        # Skip if empty order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        
        # Calculate fair value and market metrics
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Store current mid price and track recent prices
        product_data["recent_prices"].append(mid_price)
        if len(product_data["recent_prices"]) > 20:
            product_data["recent_prices"] = product_data["recent_prices"][-20:]
        
        # Calculate market volatility as standard deviation of recent prices
        if len(product_data["recent_prices"]) >= 5:
            mean_price = sum(product_data["recent_prices"]) / len(product_data["recent_prices"])
            squared_diffs = [(price - mean_price) ** 2 for price in product_data["recent_prices"]]
            product_data["market_volatility"] = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
        
        # Store for next iteration
        product_data["last_mid_price"] = mid_price
        
        # Position limit for PICNIC_BASKET1
        position_limit = 20
        current_position = product_data["position"]
        
        # Detect market trend
        market_trend = self.detect_market_trend(product_data["recent_prices"])
        
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
        if product_data["charlie_buy_prices"]:
            recent_prices = product_data["charlie_buy_prices"][-10:] if len(product_data["charlie_buy_prices"]) > 10 else product_data["charlie_buy_prices"]
            avg_charlie_buy = sum(recent_prices) / len(recent_prices)
            charlie_buy_threshold = avg_charlie_buy
        
        if product_data["charlie_sell_prices"]:
            recent_prices = product_data["charlie_sell_prices"][-10:] if len(product_data["charlie_sell_prices"]) > 10 else product_data["charlie_sell_prices"]
            avg_charlie_sell = sum(recent_prices) / len(recent_prices)
            charlie_sell_threshold = avg_charlie_sell
        
        # Calculate risk parameters
        position_scale, stop_loss_triggered, avoid_new_positions = self.calculate_risk_parameters(
            product_data, current_position, position_limit, mid_price)
        
        # Recalculate remaining capacity with position limits
        effective_position_limit = position_limit
        if product_data["market_volatility"] > 100:  # High volatility
            effective_position_limit = int(position_limit * 0.7)  # Reduce position limit
            
        max_long_capacity = min(max_long_capacity, effective_position_limit - current_position)
        max_short_capacity = min(max_short_capacity, effective_position_limit + current_position)
        
        # Buy orders
        if max_long_capacity > 0 and not (avoid_new_positions and current_position >= 0):
            sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sell_prices:
                # Skip if stop loss is triggered and we'd increase position
                if stop_loss_triggered and current_position > 0:
                    continue
                    
                # Standard buy logic - look for prices below Charlie's selling threshold
                if price <= charlie_sell_threshold * 1.0001:
                    volume = abs(order_depth.sell_orders[price])
                    
                    # Determine buy quantity based on how good the price is
                    if price < charlie_sell_threshold * 0.998:  # Very good price
                        buy_quantity = min(volume, max_long_capacity, 12)
                    elif price < charlie_sell_threshold * 0.999:  # Good price
                        buy_quantity = min(volume, max_long_capacity, 8)
                    else:  # Acceptable price
                        buy_quantity = min(volume, max_long_capacity, 5)
                    
                    # Apply position scaling
                    buy_quantity = int(buy_quantity * position_scale)
                    
                    # Adjust based on market trend
                    if market_trend == "UP":
                        buy_quantity = int(buy_quantity * 1.2)  # More aggressive in uptrend
                    elif market_trend == "DOWN":
                        buy_quantity = int(buy_quantity * 0.8)  # Less aggressive in downtrend
                    
                    # Make sure quantity is at least 1
                    buy_quantity = max(1, buy_quantity)
                    
                    if buy_quantity > 0:
                        orders.append(Order(target_product, price, buy_quantity))
                        max_long_capacity -= buy_quantity
        
        # Sell orders
        if max_short_capacity > 0 and not (avoid_new_positions and current_position <= 0):
            buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in buy_prices:
                # Skip if stop loss is triggered and we'd increase position
                if stop_loss_triggered and current_position < 0:
                    continue
                
                # Standard sell logic - look for prices above Charlie's buying threshold
                if price >= charlie_buy_threshold * 0.9999:
                    volume = order_depth.buy_orders[price]
                    
                    # Determine sell quantity based on how good the price is
                    if price > charlie_buy_threshold * 1.002:  # Very good price
                        sell_quantity = min(volume, max_short_capacity, 12)
                    elif price > charlie_buy_threshold * 1.001:  # Good price
                        sell_quantity = min(volume, max_short_capacity, 8)
                    else:  # Acceptable price
                        sell_quantity = min(volume, max_short_capacity, 5)
                    
                    # Apply position scaling
                    sell_quantity = int(sell_quantity * position_scale)
                    
                    # Adjust based on market trend
                    if market_trend == "DOWN":
                        sell_quantity = int(sell_quantity * 1.2)  # More aggressive in downtrend
                    elif market_trend == "UP":
                        sell_quantity = int(sell_quantity * 0.8)  # Less aggressive in uptrend
                    
                    # Make sure quantity is at least 1
                    sell_quantity = max(1, sell_quantity)
                    
                    if sell_quantity > 0:
                        orders.append(Order(target_product, price, -sell_quantity))
                        max_short_capacity -= sell_quantity
        
        # Add stop loss / profit taking orders
        stop_loss_orders = self.generate_risk_management_orders(target_product, order_depth, product_data, current_position, stop_loss_triggered)
        orders.extend(stop_loss_orders)
        
        return orders
    
    def generate_voucher_orders(self, state, product_data):
        """Generate orders for VOLCANIC_ROCK_VOUCHER_9750 - SIMPLIFIED to focus on Penelope and Pablo only"""
        target_product = "VOLCANIC_ROCK_VOUCHER_9750"
        
        # Skip if the product is not in order depths
        if target_product not in state.order_depths:
            return []
        
        # Get order depth
        order_depth = state.order_depths[target_product]
        
        # Skip if empty order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        
        # Calculate fair value and market metrics
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Store current mid price and track recent prices
        product_data["recent_prices"].append(mid_price)
        if len(product_data["recent_prices"]) > 70:
            product_data["recent_prices"] = product_data["recent_prices"][-20:]
        
        # Calculate market volatility as standard deviation of recent prices
        if len(product_data["recent_prices"]) >= 5:
            mean_price = sum(product_data["recent_prices"]) / len(product_data["recent_prices"])
            squared_diffs = [(price - mean_price) ** 2 for price in product_data["recent_prices"]]
            product_data["market_volatility"] = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
        
        # Store for next iteration
        product_data["last_mid_price"] = mid_price
        
        # Position limit for vouchers
        position_limit = 50
        current_position = product_data["position"]
        
        # Detect market trend
        market_trend = self.detect_market_trend(product_data["recent_prices"])
        
        # Calculate risk parameters
        position_scale, stop_loss_triggered, avoid_new_positions = self.calculate_risk_parameters(
            product_data, current_position, position_limit, mid_price)
        
        # Calculate remaining capacity
        effective_position_limit = position_limit
        if product_data["market_volatility"] > 50:  # High volatility for vouchers
            effective_position_limit = int(position_limit * 0.7)  # Reduce position limit
            
        max_long_capacity = min(position_limit - current_position, 
                               effective_position_limit - current_position)
        max_short_capacity = min(position_limit + current_position,
                                effective_position_limit + current_position)
        
        # Generate orders list
        orders = []
        
        # Voucher trading thresholds based on analysis
        # Buy from Penelope (319.51)
        # Sell to Pablo (359.21)
        penelope_sell_threshold = 325  # Default
        pablo_buy_threshold = 355      # Default
        
        # Calculate average prices if we have data
        if product_data["penelope_sell_prices"]:
            recent_prices = product_data["penelope_sell_prices"][-10:] if len(product_data["penelope_sell_prices"]) > 10 else product_data["penelope_sell_prices"]
            penelope_sell_threshold = sum(recent_prices) / len(recent_prices)
        
        if product_data["pablo_buy_prices"]:
            recent_prices = product_data["pablo_buy_prices"][-10:] if len(product_data["pablo_buy_prices"]) > 10 else product_data["pablo_buy_prices"]
            pablo_buy_threshold = sum(recent_prices) / len(recent_prices)
        
        # Buy orders - looking for prices at or below Penelope's selling price
        if max_long_capacity > 0 and not (avoid_new_positions and current_position >= 0):
            sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sell_prices:
                # Skip if stop loss is triggered and we'd increase position
                if stop_loss_triggered and current_position > 0:
                    continue
                    
                # Buy logic - look for prices at or below Penelope's selling threshold
                if price <= penelope_sell_threshold * 1.01:
                    volume = abs(order_depth.sell_orders[price])
                    
                    # Determine buy quantity based on how good the price is
                    if price < penelope_sell_threshold * 0.97:  # Very good price
                        buy_quantity = min(volume, max_long_capacity, 15)
                    elif price < penelope_sell_threshold * 0.99:  # Good price
                        buy_quantity = min(volume, max_long_capacity, 10)
                    else:  # Acceptable price
                        buy_quantity = min(volume, max_long_capacity, 5)
                    
                    # Apply position scaling
                    buy_quantity = int(buy_quantity * position_scale)
                    
                    # Adjust based on market trend
                    if market_trend == "UP":
                        buy_quantity = int(buy_quantity * 1.2)  # More aggressive in uptrend
                    elif market_trend == "DOWN":
                        buy_quantity = int(buy_quantity * 0.8)  # Less aggressive in downtrend
                    
                    # Make sure quantity is at least 1
                    buy_quantity = max(1, buy_quantity)
                    
                    if buy_quantity > 0:
                        orders.append(Order(target_product, price, buy_quantity))
                        max_long_capacity -= buy_quantity
        
        # Sell orders - looking for prices at or above Pablo's buying price
        if max_short_capacity > 0 and not (avoid_new_positions and current_position <= 0):
            buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in buy_prices:
                # Skip if stop loss is triggered and we'd increase position
                if stop_loss_triggered and current_position < 0:
                    continue
                
                # Sell logic - look for prices at or above Pablo's buying threshold
                if price >= pablo_buy_threshold * 0.99:
                    volume = order_depth.buy_orders[price]
                    
                    # Determine sell quantity based on how good the price is
                    if price > pablo_buy_threshold * 1.03:  # Very good price
                        sell_quantity = min(volume, max_short_capacity, 15)
                    elif price > pablo_buy_threshold * 1.01:  # Good price
                        sell_quantity = min(volume, max_short_capacity, 10)
                    else:  # Acceptable price
                        sell_quantity = min(volume, max_short_capacity, 5)
                    
                    # Apply position scaling
                    sell_quantity = int(sell_quantity * position_scale)
                    
                    # Adjust based on market trend
                    if market_trend == "DOWN":
                        sell_quantity = int(sell_quantity * 1.2)  # More aggressive in downtrend
                    elif market_trend == "UP":
                        sell_quantity = int(sell_quantity * 0.8)  # Less aggressive in uptrend
                    
                    # Make sure quantity is at least 1
                    sell_quantity = max(1, sell_quantity)
                    
                    if sell_quantity > 0:
                        orders.append(Order(target_product, price, -sell_quantity))
                        max_short_capacity -= sell_quantity
        
        # Add stop loss / profit taking orders
        stop_loss_orders = self.generate_risk_management_orders(target_product, order_depth, product_data, current_position, stop_loss_triggered)
        orders.extend(stop_loss_orders)
        
        return orders
    
    def detect_market_trend(self, recent_prices):
        """Detect market trend based on recent prices"""
        if len(recent_prices) < 10:
            return "NEUTRAL"
            
        early_prices = recent_prices[:5]
        late_prices = recent_prices[-5:]
        avg_early = sum(early_prices) / len(early_prices)
        avg_late = sum(late_prices) / len(late_prices)
        
        if avg_late > avg_early * 1.001:
            return "UP"
        elif avg_late < avg_early * 0.999:
            return "DOWN"
        return "NEUTRAL"
    
    def calculate_risk_parameters(self, product_data, current_position, position_limit, mid_price):
        """Calculate risk management parameters"""
        # 1. Position scaling based on current position
        position_scale = 1.0
        if abs(current_position) > position_limit * 0.7:
            position_scale = 0.7  # Scale down orders as we approach limits
        elif abs(current_position) > position_limit * 0.5:
            position_scale = 0.85
        
        # 2. Stop loss detection
        stop_loss_triggered = False
        average_position_price = product_data.get("average_position_price", 0)
        
        if average_position_price > 0:
            if current_position > 0 and mid_price < average_position_price * 0.97:
                stop_loss_triggered = True
            elif current_position < 0 and mid_price > average_position_price * 1.03:
                stop_loss_triggered = True
        
        # 3. Detect large recent price moves
        recent_large_move = False
        if len(product_data["recent_prices"]) >= 3:
            price_changes = [abs(product_data["recent_prices"][i] - product_data["recent_prices"][i-1]) 
                            for i in range(1, len(product_data["recent_prices"]))]
            if price_changes:
                avg_change = sum(price_changes) / len(price_changes)
                recent_changes = price_changes[-3:]
                if any(change > avg_change * 2 for change in recent_changes):
                    recent_large_move = True
        
        # 4. Avoid new positions during extreme conditions
        avoid_new_positions = recent_large_move
        if product_data.get("market_volatility", 0) > 0 and product_data["recent_prices"]:
            avg_price = sum(product_data["recent_prices"]) / len(product_data["recent_prices"])
            relative_volatility = product_data["market_volatility"] / avg_price
            
            if relative_volatility > 0.1:  # High relative volatility
                avoid_new_positions = True
        
        return position_scale, stop_loss_triggered, avoid_new_positions
    
    def generate_risk_management_orders(self, product, order_depth, product_data, current_position, stop_loss_triggered):
        """Generate orders for risk management (stop loss and profit taking)"""
        orders = []
        
        # Skip if no position
        if current_position == 0:
            return orders
            
        # Calculate average position price
        average_position_price = product_data.get("average_position_price", 0)
        if product_data["last_trades"]:
            position_trades = [t for t in product_data["last_trades"] 
                             if (t["quantity"] > 0 and current_position > 0) or 
                               (t["quantity"] < 0 and current_position < 0)]
            if position_trades:
                prices = [t["price"] for t in position_trades]
                average_position_price = sum(prices) / len(prices)
                product_data["average_position_price"] = average_position_price
        
        # If stop loss is triggered, reduce position
        if stop_loss_triggered:
            if current_position > 0 and order_depth.buy_orders:
                # Sell some position at market (best bid)
                best_price = max(order_depth.buy_orders.keys())
                volume = order_depth.buy_orders[best_price]
                # Sell up to 50% of our position
                sell_quantity = min(volume, current_position // 2)
                if sell_quantity > 0:
                    orders.append(Order(product, best_price, -sell_quantity))
            
            elif current_position < 0 and order_depth.sell_orders:
                # Buy some position at market (best ask)
                best_price = min(order_depth.sell_orders.keys())
                volume = abs(order_depth.sell_orders[best_price])
                # Buy up to 50% of our position
                buy_quantity = min(volume, abs(current_position) // 2)
                if buy_quantity > 0:
                    orders.append(Order(product, best_price, buy_quantity))
        
        # If position is highly profitable, consider reducing it
        elif average_position_price > 0:
            # Get mid price
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Profit thresholds depend on product
                profit_threshold = 0.01  # 1% default
                if product == "VOLCANIC_ROCK_VOUCHER_9750":
                    profit_threshold = 0.03  # 3% for vouchers (more volatile)
                
                if current_position > 5 and mid_price > average_position_price * (1 + profit_threshold):
                    # Take profit on some position
                    if order_depth.buy_orders:
                        best_price = max(order_depth.buy_orders.keys())
                        volume = order_depth.buy_orders[best_price]
                        # Sell up to 30% of our position
                        sell_quantity = min(volume, current_position // 3)
                        if sell_quantity > 0:
                            orders.append(Order(product, best_price, -sell_quantity))
                
                elif current_position < -5 and mid_price < average_position_price * (1 - profit_threshold):
                    # Take profit on some position
                    if order_depth.sell_orders:
                        best_price = min(order_depth.sell_orders.keys())
                        volume = abs(order_depth.sell_orders[best_price])
                        # Buy up to 30% of our position
                        buy_quantity = min(volume, abs(current_position) // 3)
                        if buy_quantity > 0:
                            orders.append(Order(product, best_price, buy_quantity))
        
        return orders
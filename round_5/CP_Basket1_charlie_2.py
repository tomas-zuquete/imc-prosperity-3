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
                "trades_with_charlie": 0,
                "market_volatility": 0,
                "recent_prices": [],
                "last_trades": [],
                "profit_tracking": [],
                "iteration": 0
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
                    "trades_with_charlie": 0,
                    "market_volatility": 0,
                    "recent_prices": [],
                    "last_trades": [],
                    "profit_tracking": [],
                    "iteration": 0
                }
        
        trader_data["iteration"] += 1
        
        # Update position from state
        if "PICNIC_BASKET1" in state.position:
            trader_data["position"] = state.position["PICNIC_BASKET1"]
        
        # Process recent trades to gather counterparty data
        if "PICNIC_BASKET1" in state.own_trades and state.own_trades["PICNIC_BASKET1"]:
            trade_prices = []
            for trade in state.own_trades["PICNIC_BASKET1"]:
                counterparty = trade.counter_party if hasattr(trade, 'counter_party') else "Unknown"
                trade_prices.append(trade.price)
                
                if counterparty == "Charlie":
                    trader_data["trades_with_charlie"] += 1
                    if trade.quantity > 0:  # We bought from Charlie
                        trader_data["charlie_sell_prices"].append(trade.price)
                    else:  # We sold to Charlie
                        trader_data["charlie_buy_prices"].append(trade.price)
                
                # Save last 10 trades to detect trends
                trader_data["last_trades"].append({
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "counterparty": counterparty
                })
                if len(trader_data["last_trades"]) > 10:
                    trader_data["last_trades"] = trader_data["last_trades"][-10:]
        
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
        spread = best_ask - best_bid
        
        # Store current mid price and track recent prices
        trader_data["recent_prices"].append(mid_price)
        if len(trader_data["recent_prices"]) > 20:
            trader_data["recent_prices"] = trader_data["recent_prices"][-20:]
        
        # Calculate market volatility as standard deviation of recent prices
        if len(trader_data["recent_prices"]) >= 5:
            mean_price = sum(trader_data["recent_prices"]) / len(trader_data["recent_prices"])
            squared_diffs = [(price - mean_price) ** 2 for price in trader_data["recent_prices"]]
            trader_data["market_volatility"] = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
        
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
            # Use the most recent 10 prices to be more adaptive
            recent_prices = trader_data["charlie_buy_prices"][-10:]
            avg_charlie_buy = sum(recent_prices) / len(recent_prices)
            charlie_buy_threshold = avg_charlie_buy
        
        if trader_data["charlie_sell_prices"]:
            # Use the most recent 10 prices to be more adaptive
            recent_prices = trader_data["charlie_sell_prices"][-10:]
            avg_charlie_sell = sum(recent_prices) / len(recent_prices)
            charlie_sell_threshold = avg_charlie_sell
        
        # Detect market trend based on recent prices
        market_trend = "NEUTRAL"
        if len(trader_data["recent_prices"]) >= 10:
            early_prices = trader_data["recent_prices"][:5]
            late_prices = trader_data["recent_prices"][-5:]
            avg_early = sum(early_prices) / len(early_prices)
            avg_late = sum(late_prices) / len(late_prices)
            
            if avg_late > avg_early * 1.001:
                market_trend = "UP"
            elif avg_late < avg_early * 0.999:
                market_trend = "DOWN"
        
        # Detect if there was a recent large price move
        recent_large_move = False
        if len(trader_data["recent_prices"]) >= 3:
            price_changes = [abs(trader_data["recent_prices"][i] - trader_data["recent_prices"][i-1]) 
                            for i in range(1, len(trader_data["recent_prices"]))]
            avg_change = sum(price_changes) / len(price_changes)
            recent_changes = price_changes[-3:]
            if any(change > avg_change * 2 for change in recent_changes):
                recent_large_move = True
        
        # ========== IMPROVED RISK MANAGEMENT ==========
        
        # 1. Adjust position limits based on volatility
        effective_position_limit = position_limit
        if trader_data["market_volatility"] > 100:  # High volatility
            effective_position_limit = int(position_limit * 0.7)  # Reduce position limit
        
        # Recalculate remaining capacity with adjusted limits
        max_long_capacity = min(max_long_capacity, effective_position_limit - current_position)
        max_short_capacity = min(max_short_capacity, effective_position_limit + current_position)
        
        # 2. Implement position scaling based on current position
        # Be more conservative as we approach position limits
        position_scale = 1.0
        if abs(current_position) > effective_position_limit * 0.7:
            position_scale = 0.7  # Scale down orders as we approach limits
        elif abs(current_position) > effective_position_limit * 0.5:
            position_scale = 0.85
        
        # 3. Implement profit-taking thresholds
        # If position is strongly profitable, be more willing to reduce it
        average_position_price = 0
        if current_position != 0 and trader_data["last_trades"]:
            position_trades = [t for t in trader_data["last_trades"] 
                              if (t["quantity"] > 0 and current_position > 0) or 
                                 (t["quantity"] < 0 and current_position < 0)]
            if position_trades:
                prices = [t["price"] for t in position_trades]
                average_position_price = sum(prices) / len(prices)
        
        # 4. Implement stop-loss protection
        stop_loss_triggered = False
        if current_position > 0 and average_position_price > 0 and mid_price < average_position_price * 0.997:
            stop_loss_triggered = True
        elif current_position < 0 and average_position_price > 0 and mid_price > average_position_price * 1.003:
            stop_loss_triggered = True
        
        # 5. Avoid trading during extreme volatility or after large price moves
        avoid_new_positions = trader_data["market_volatility"] > 150 or recent_large_move
        
        # ========== BUY STRATEGY ==========
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
        
        # ========== SELL STRATEGY ==========
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
        
        # ========== STOP LOSS / PROFIT TAKING LOGIC ==========
        
        # If stop loss is triggered, reduce position
        if stop_loss_triggered:
            if current_position > 0 and order_depth.buy_orders:
                # Sell some position at market (best bid)
                best_price = max(order_depth.buy_orders.keys())
                volume = order_depth.buy_orders[best_price]
                # Sell up to 50% of our position
                sell_quantity = min(volume, current_position // 2)
                if sell_quantity > 0:
                    orders.append(Order(target_product, best_price, -sell_quantity))
            
            elif current_position < 0 and order_depth.sell_orders:
                # Buy some position at market (best ask)
                best_price = min(order_depth.sell_orders.keys())
                volume = abs(order_depth.sell_orders[best_price])
                # Buy up to 50% of our position
                buy_quantity = min(volume, abs(current_position) // 2)
                if buy_quantity > 0:
                    orders.append(Order(target_product, best_price, buy_quantity))
        
        # If position is highly profitable, consider reducing it
        elif average_position_price > 0:
            if current_position > 5 and mid_price > average_position_price * 1.005:
                # Take profit on some position
                if order_depth.buy_orders:
                    best_price = max(order_depth.buy_orders.keys())
                    volume = order_depth.buy_orders[best_price]
                    # Sell up to 30% of our position
                    sell_quantity = min(volume, current_position // 3)
                    if sell_quantity > 0:
                        orders.append(Order(target_product, best_price, -sell_quantity))
            
            elif current_position < -5 and mid_price < average_position_price * 0.995:
                # Take profit on some position
                if order_depth.sell_orders:
                    best_price = min(order_depth.sell_orders.keys())
                    volume = abs(order_depth.sell_orders[best_price])
                    # Buy up to 30% of our position
                    buy_quantity = min(volume, abs(current_position) // 3)
                    if buy_quantity > 0:
                        orders.append(Order(target_product, best_price, buy_quantity))
        
        # Add orders to result
        result[target_product] = orders
        
        # For all other products, return empty order lists
        for product in state.order_depths:
            if product != target_product:
                result[product] = []
        
        return result, 0, jsonpickle.encode(trader_data)
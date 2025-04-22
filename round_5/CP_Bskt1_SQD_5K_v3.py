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
                    "average_position_price": 0,
                    "cumulative_pnl": 0,
                },
                # Product 2: SQUID_INK with multiple counterparties
                "SQUID_INK": {
                    # Price history for each counterparty
                    "Gary_buy_prices": [],  # When Gary buys from us
                    "Gary_sell_prices": [], # When Gary sells to us
                    "Caesar_buy_prices": [],  # When Caesar buys from us
                    "Caesar_sell_prices": [], # When Caesar sells to us
                    "Camilla_buy_prices": [],  # When Camilla buys from us
                    "Camilla_sell_prices": [], # When Camilla sells to us
                    "Paris_buy_prices": [],  # When Paris buys from us
                    "Paris_sell_prices": [], # When Paris sells to us
                    "Gina_buy_prices": [],  # When Gina buys from us
                    "Gina_sell_prices": [], # When Gina sells to us
                    "Pablo_buy_prices": [],  # When Pablo buys from us
                    "Pablo_sell_prices": [], # When Pablo sells to us
                    
                    # Average trading prices for each counterparty
                    "counterparty_data": {
                        "Gary": {"avg_buy_price": 2040, "avg_sell_price": 1940, "trades": 0, "profitable_trades": 0},
                        "Caesar": {"avg_buy_price": 2020, "avg_sell_price": 1960, "trades": 0, "profitable_trades": 0},
                        "Camilla": {"avg_buy_price": 2030, "avg_sell_price": 1975, "trades": 0, "profitable_trades": 0},
                        "Paris": {"avg_buy_price": 2035, "avg_sell_price": 1940, "trades": 0, "profitable_trades": 0},
                        "Charlie": {"avg_buy_price": 2035, "avg_sell_price": 1940, "trades": 0, "profitable_trades": 0},
                        "Gina": {"avg_buy_price": 2020, "avg_sell_price": 1940, "trades": 0, "profitable_trades": 0},
                        "Pablo": {"avg_buy_price": 2020, "avg_sell_price": 1940, "trades": 0, "profitable_trades": 0}
                    },
                    
                    # Best counterparties lists
                    "best_buyers": ["Gary","Paris", "Charlie"],
                    "best_sellers": ["Caesar", "Charlie", "Paris"],
                    
                    # Last mid price and other trading data
                    "last_mid_price": None,
                    "position": 0,
                    "market_volatility": 0,
                    "recent_prices": [],
                    "last_trades": [],
                    "average_position_price": 0,
                    "cumulative_pnl": 0,
                },
                "iteration": 0,
                "total_pnl": 0
            }
        else:
            try:
                trader_data = jsonpickle.decode(state.traderData)
                # Make sure we have all fields for SQUID_INK
                if "SQUID_INK" in trader_data:
                    # Initialize counterparty price arrays if missing
                    for party in ["Gary", "Caesar", "Camilla", "Paris", "Gina", "Pablo"]:
                        if f"{party}_buy_prices" not in trader_data["SQUID_INK"]:
                            trader_data["SQUID_INK"][f"{party}_buy_prices"] = []
                        if f"{party}_sell_prices" not in trader_data["SQUID_INK"]:
                            trader_data["SQUID_INK"][f"{party}_sell_prices"] = []
                    
                    # Initialize counterparty_data if missing
                    if "counterparty_data" not in trader_data["SQUID_INK"]:
                        trader_data["SQUID_INK"]["counterparty_data"] = {
                            "Gary": {"avg_buy_price": 2030, "avg_sell_price": 1980, "trades": 0, "profitable_trades": 0},
                            "Caesar": {"avg_buy_price": 2000, "avg_sell_price": 1970, "trades": 0, "profitable_trades": 0},
                            "Camilla": {"avg_buy_price": 2020, "avg_sell_price": 1975, "trades": 0, "profitable_trades": 0},
                            "Paris": {"avg_buy_price": 2010, "avg_sell_price": 1960, "trades": 0, "profitable_trades": 0},
                            "Gina": {"avg_buy_price": 2000, "avg_sell_price": 1945, "trades": 0, "profitable_trades": 0},
                            "Pablo": {"avg_buy_price": 2020, "avg_sell_price": 1970, "trades": 0, "profitable_trades": 0}
                        }
                    
                    # Initialize best buyers/sellers lists if missing
                    if "best_buyers" not in trader_data["SQUID_INK"]:
                        trader_data["SQUID_INK"]["best_buyers"] = ["Gary", "Camilla", "Pablo"]
                    if "best_sellers" not in trader_data["SQUID_INK"]:
                        trader_data["SQUID_INK"]["best_sellers"] = ["Caesar", "Paris", "Gina"]
            except:
                # Reset to default if deserialization fails
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
                        "average_position_price": 0,
                        "cumulative_pnl": 0,
                    },
                    # Product 2: SQUID_INK with multiple counterparties
                    "SQUID_INK": {
                        # Price history for each counterparty
                        "Gary_buy_prices": [],
                        "Gary_sell_prices": [],
                        "Caesar_buy_prices": [],
                        "Caesar_sell_prices": [],
                        "Camilla_buy_prices": [],
                        "Camilla_sell_prices": [],
                        "Paris_buy_prices": [],
                        "Paris_sell_prices": [],
                        "Gina_buy_prices": [],
                        "Gina_sell_prices": [],
                        "Pablo_buy_prices": [],
                        "Pablo_sell_prices": [],
                        
                        # Average trading prices for each counterparty
                        "counterparty_data": {
                            "Gary": {"avg_buy_price": 2030, "avg_sell_price": 1980, "trades": 0, "profitable_trades": 0},
                            "Caesar": {"avg_buy_price": 2000, "avg_sell_price": 1970, "trades": 0, "profitable_trades": 0},
                            "Camilla": {"avg_buy_price": 2020, "avg_sell_price": 1975, "trades": 0, "profitable_trades": 0},
                            "Paris": {"avg_buy_price": 2040, "avg_sell_price": 1950, "trades": 0, "profitable_trades": 0},
                            "Gina": {"avg_buy_price": 2000, "avg_sell_price": 1945, "trades": 0, "profitable_trades": 0},
                            "Pablo": {"avg_buy_price": 2020, "avg_sell_price": 1970, "trades": 0, "profitable_trades": 0}
                        },
                        
                        # Best counterparties lists
                        "best_buyers": ["Gary", "Camilla", "Pablo"],
                        "best_sellers": ["Caesar", "Paris", "Gina"],
                        
                        "last_mid_price": None,
                        "position": 0,
                        "market_volatility": 0,
                        "recent_prices": [],
                        "last_trades": [],
                        "average_position_price": 0,
                        "cumulative_pnl": 0,
                    },
                    "iteration": 0,
                    "total_pnl": 0
                }
        
        trader_data["iteration"] += 1
        
        # Update positions from state
        if "PICNIC_BASKET1" in state.position:
            trader_data["PICNIC_BASKET1"]["position"] = state.position["PICNIC_BASKET1"]
        
        if "SQUID_INK" in state.position:
            trader_data["SQUID_INK"]["position"] = state.position["SQUID_INK"]
        
        # Process recent trades to gather counterparty data
        self.update_trade_data(state, trader_data)
        
        # Update best counterparties based on trade history
        self.update_best_counterparties(trader_data["SQUID_INK"])
        
        # Initialize result dictionary
        result = {}
        
        # Generate orders for PICNIC_BASKET1
        picnic_orders = self.generate_picnic_basket_orders(state, trader_data["PICNIC_BASKET1"])
        if picnic_orders:
            result["PICNIC_BASKET1"] = picnic_orders
        
        # Generate orders for SQUID_INK with multiple counterparties
        squid_orders = self.generate_multi_counterparty_squid_orders(state, trader_data["SQUID_INK"])
        if squid_orders:
            result["SQUID_INK"] = squid_orders
        
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
                # In Round 5, counter_party is now available directly
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
        
        # Process SQUID_INK trades with multiple counterparties
        if "SQUID_INK" in state.own_trades and state.own_trades["SQUID_INK"]:
            for trade in state.own_trades["SQUID_INK"]:
                # Get counterparty from trade object - Handle both upper and lowercase
                counterparty = trade.counter_party if hasattr(trade, 'counter_party') else "Unknown"
                
                # Standardize capitalization for all known counterparties
                for name in ["Gary", "Caesar", "Camilla", "Paris", "Gina", "Pablo"]:
                    if counterparty.lower() == name.lower():
                        counterparty = name
                        break
                
                # Track trade with specific counterparty
                if counterparty in trader_data["SQUID_INK"]["counterparty_data"]:
                    trader_data["SQUID_INK"]["counterparty_data"][counterparty]["trades"] += 1
                    
                    if trade.quantity > 0:  # We bought from them
                        # Track their selling price
                        trader_data["SQUID_INK"][f"{counterparty}_sell_prices"].append(trade.price)
                        
                        # Update their average selling price
                        recent_prices = trader_data["SQUID_INK"][f"{counterparty}_sell_prices"][-10:] if len(trader_data["SQUID_INK"][f"{counterparty}_sell_prices"]) > 10 else trader_data["SQUID_INK"][f"{counterparty}_sell_prices"]
                        if recent_prices:
                            trader_data["SQUID_INK"]["counterparty_data"][counterparty]["avg_sell_price"] = sum(recent_prices) / len(recent_prices)
                    
                    else:  # We sold to them
                        # Track their buying price
                        trader_data["SQUID_INK"][f"{counterparty}_buy_prices"].append(trade.price)
                        
                        # Update their average buying price
                        recent_prices = trader_data["SQUID_INK"][f"{counterparty}_buy_prices"][-10:] if len(trader_data["SQUID_INK"][f"{counterparty}_buy_prices"]) > 10 else trader_data["SQUID_INK"][f"{counterparty}_buy_prices"]
                        if recent_prices:
                            trader_data["SQUID_INK"]["counterparty_data"][counterparty]["avg_buy_price"] = sum(recent_prices) / len(recent_prices)
                
                # Track all trades for market analysis
                trader_data["SQUID_INK"]["last_trades"].append({
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "counterparty": counterparty,
                    "timestamp": state.timestamp if hasattr(state, 'timestamp') else 0
                })
                
                if len(trader_data["SQUID_INK"]["last_trades"]) > 20:
                    trader_data["SQUID_INK"]["last_trades"] = trader_data["SQUID_INK"]["last_trades"][-20:]
    
    def update_best_counterparties(self, product_data):
        """Update lists of best buyers and sellers based on trade history"""
        counterparty_data = product_data["counterparty_data"]
        
        # Only update if we have enough trade data
        total_trades = sum(data["trades"] for data in counterparty_data.values())
        if total_trades < 5:  # Need at least 5 trades to have reliable data
            return
        
        # Find best buyers (highest buying prices)
        buyers = [(name, data["avg_buy_price"]) for name, data in counterparty_data.items() 
                 if data["trades"] > 0 and data["avg_buy_price"] > 0]
        buyers.sort(key=lambda x: x[1], reverse=True)  # Sort by highest buy price
        
        # Find best sellers (lowest selling prices)
        sellers = [(name, data["avg_sell_price"]) for name, data in counterparty_data.items() 
                  if data["trades"] > 0 and data["avg_sell_price"] > 0]
        sellers.sort(key=lambda x: x[1])  # Sort by lowest sell price
        
        # Update best counterparties if we have enough data
        if len(buyers) >= 3:
            product_data["best_buyers"] = [name for name, _ in buyers[:3]]
        
        if len(sellers) >= 3:
            product_data["best_sellers"] = [name for name, _ in sellers[:3]]
    
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

    def generate_multi_counterparty_squid_orders(self, state, product_data):
        """Generate orders for SQUID_INK using multiple counterparties"""
        target_product = "SQUID_INK"
        
        # Skip if the product is not in order depths
        if target_product not in state.order_depths:
            return []
        
        # Get order depth for SQUID_INK
        order_depth = state.order_depths[target_product]
        
        # Skip if empty order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        
        # Calculate market metrics
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Store current mid price and track recent prices
        product_data["recent_prices"].append(mid_price)
        if len(product_data["recent_prices"]) > 20:
            product_data["recent_prices"] = product_data["recent_prices"][-20:]
        
        # Calculate market volatility
        if len(product_data["recent_prices"]) >= 5:
            mean_price = sum(product_data["recent_prices"]) / len(product_data["recent_prices"])
            squared_diffs = [(price - mean_price) ** 2 for price in product_data["recent_prices"]]
            product_data["market_volatility"] = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
        
        # Store for next iteration
        product_data["last_mid_price"] = mid_price
        
        # Position limit for SQUID_INK
        position_limit = 60
        current_position = product_data["position"]
        
        # Get market trend
        market_trend = self.detect_market_trend(product_data["recent_prices"])
        
        # Calculate risk parameters
        position_scale, stop_loss_triggered, avoid_new_positions = self.calculate_risk_parameters_squid(
            product_data, current_position, position_limit, mid_price)
        
        # Calculate remaining capacity
        max_long_capacity = position_limit - current_position
        max_short_capacity = position_limit + current_position
        
        # Adjust position limits based on volatility
        if product_data["market_volatility"] > mid_price * 0.03:  # More than 3% volatility
            max_long_capacity = int(max_long_capacity * 0.7)
            max_short_capacity = int(max_short_capacity * 0.7)
        
        # Generate orders list
        orders = []
        
        # ---- LEVEL 1: GROUP-LEVEL TRADING ----
        # Find the best sellers and buyers across all counterparties
        counterparty_data = product_data["counterparty_data"]
        best_sellers = product_data["best_sellers"]
        best_buyers = product_data["best_buyers"]
        
        # Find lowest sell price among best sellers
        lowest_sell_price = float('inf')
        for seller in best_sellers:
            if seller in counterparty_data:
                lowest_sell_price = min(lowest_sell_price, counterparty_data[seller]["avg_sell_price"])
                
        # Find highest buy price among best buyers
        highest_buy_price = 0
        for buyer in best_buyers:
            if buyer in counterparty_data:
                highest_buy_price = max(highest_buy_price, counterparty_data[buyer]["avg_buy_price"])
        
        # Use reasonable defaults if we don't have enough data
        if lowest_sell_price == float('inf'):
            lowest_sell_price = mid_price * 0.99  # 1% below mid price
        
        if highest_buy_price == 0:
            highest_buy_price = mid_price * 1.01  # 1% above mid price
        
        # Group-level buy orders - buy from best sellers at/below their typical selling price
        if max_long_capacity > 0 and not (avoid_new_positions and current_position >= 0):
            sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sell_prices:
                # Skip if stop loss triggered
                if stop_loss_triggered and current_position > 0:
                    continue
                
                # Look for prices at or below our best sellers' threshold
                if price <= lowest_sell_price * 1.005:  # Allow up to 0.5% higher
                    volume = abs(order_depth.sell_orders[price])
                    
                    # Determine quantity based on price quality
                    if price < lowest_sell_price * 0.99:  # Excellent price (1% below threshold)
                        buy_quantity = min(volume, max_long_capacity, 15)
                    elif price < lowest_sell_price * 0.995:  # Very good price (0.5% below)
                        buy_quantity = min(volume, max_long_capacity, 12)
                    else:  # Good price (at or slightly above threshold)
                        buy_quantity = min(volume, max_long_capacity, 8)
                    
                    # Apply position scaling
                    buy_quantity = int(buy_quantity * position_scale)
                    
                    # Adjust for market trend
                    if market_trend == "UP":
                        buy_quantity = int(buy_quantity * 1.2)  # More aggressive in uptrend
                    elif market_trend == "DOWN":
                        buy_quantity = int(buy_quantity * 0.8)  # Less aggressive in downtrend
                    
                    # Ensure minimum quantity
                    buy_quantity = max(1, buy_quantity)
                    
                    if buy_quantity > 0:
                        orders.append(Order(target_product, price, buy_quantity))
                        max_long_capacity -= buy_quantity
                        
                        # Don't place too many buy orders
                        if len(orders) >= 2:
                            break
        
        # Group-level sell orders - sell to best buyers at/above their typical buying price
        if max_short_capacity > 0 and not (avoid_new_positions and current_position <= 0):
            buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in buy_prices:
                # Skip if stop loss triggered
                if stop_loss_triggered and current_position < 0:
                    continue
                
                # Look for prices at or above our best buyers' threshold
                if price >= highest_buy_price * 0.995:  # Allow up to 0.5% lower
                    volume = order_depth.buy_orders[price]
                    
                    # Determine quantity based on price quality
                    if price > highest_buy_price * 1.01:  # Excellent price (1% above threshold)
                        sell_quantity = min(volume, max_short_capacity, 15)
                    elif price > highest_buy_price * 1.005:  # Very good price (0.5% above)
                        sell_quantity = min(volume, max_short_capacity, 12)
                    else:  # Good price (at or slightly below threshold)
                        sell_quantity = min(volume, max_short_capacity, 8)
                    
                    # Apply position scaling
                    sell_quantity = int(sell_quantity * position_scale)
                    
                    # Adjust for market trend
                    if market_trend == "DOWN":
                        sell_quantity = int(sell_quantity * 1.2)  # More aggressive in downtrend
                    elif market_trend == "UP":
                        sell_quantity = int(sell_quantity * 0.8)  # Less aggressive in uptrend
                    
                    # Ensure minimum quantity
                    sell_quantity = max(1, sell_quantity)
                    
                    if sell_quantity > 0:
                        orders.append(Order(target_product, price, -sell_quantity))
                        max_short_capacity -= sell_quantity
                        
                        # Don't place too many sell orders
                        if len(orders) >= 4:  # Allow up to 2 buys + 2 sells
                            break
        
        # ---- LEVEL 2: PROFITABLE PAIR TRADING ----
        # Calculate expected profit for various trade pairs
        profitable_pairs = []
        
        for seller in best_sellers:
            seller_price = counterparty_data[seller]["avg_sell_price"]
            
            for buyer in best_buyers:
                buyer_price = counterparty_data[buyer]["avg_buy_price"]
                
                # Calculate potential profit
                profit = buyer_price - seller_price
                
                # Only include profitable pairs
                if profit > 5:  # Minimum 5-point spread to be considered profitable
                    profitable_pairs.append((seller, buyer, profit))
        
        # Sort pairs by profitability (highest first)
        profitable_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # If we still have capacity, place targeted orders for the most profitable pairs
        if (max_long_capacity > 0 or max_short_capacity > 0) and len(orders) < 6:  # Limit to 6 total orders
            # Target the top 3 most profitable pairs
            for seller, buyer, profit in profitable_pairs[:3]:
                seller_price = counterparty_data[seller]["avg_sell_price"]
                buyer_price = counterparty_data[buyer]["avg_buy_price"]
                
                # If profit is substantial, place more aggressive orders
                is_high_profit = profit > 10
                
                # Buy from seller
                if max_long_capacity > 0:
                    sell_prices = sorted(order_depth.sell_orders.keys())
                    
                    for price in sell_prices:
                        # Adjust price threshold based on profitability
                        price_threshold = seller_price * (1.01 if is_high_profit else 1.005)
                        
                        if price <= price_threshold:
                            volume = abs(order_depth.sell_orders[price])
                            # More aggressive quantity for high-profit pairs
                            buy_quantity = min(volume, max_long_capacity, 12 if is_high_profit else 8)
                            
                            if buy_quantity > 0:
                                orders.append(Order(target_product, price, buy_quantity))
                                max_long_capacity -= buy_quantity
                                break  # One targeted buy per pair
                
                # Sell to buyer
                if max_short_capacity > 0:
                    buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
                    
                    for price in buy_prices:
                        # Adjust price threshold based on profitability
                        price_threshold = buyer_price * (0.99 if is_high_profit else 0.995)
                        
                        if price >= price_threshold:
                            volume = order_depth.buy_orders[price]
                            # More aggressive quantity for high-profit pairs
                            sell_quantity = min(volume, max_short_capacity, 12 if is_high_profit else 8)
                            
                            if sell_quantity > 0:
                                orders.append(Order(target_product, price, -sell_quantity))
                                max_short_capacity -= sell_quantity
                                break  # One targeted sell per pair
        
        # ---- LEVEL 3: INDIVIDUAL COUNTERPARTY ORDERS ----
        # If we still have capacity and not too many orders, place orders for individual counterparties
        if (max_long_capacity > 0 or max_short_capacity > 0) and len(orders) < 8:
            # Get the best individual counterparties
            best_individual_seller = best_sellers[0] if best_sellers else None
            best_individual_buyer = best_buyers[0] if best_buyers else None
            
            # If we have identified the best individual counterparties
            if best_individual_seller and best_individual_buyer:
                individual_seller_price = counterparty_data[best_individual_seller]["avg_sell_price"]
                individual_buyer_price = counterparty_data[best_individual_buyer]["avg_buy_price"]
                
                # Buy specifically from best seller
                if max_long_capacity > 0:
                    sell_prices = sorted(order_depth.sell_orders.keys())
                    
                    for price in sell_prices:
                        if price <= individual_seller_price * 1.003:  # Very targeted threshold
                            volume = abs(order_depth.sell_orders[price])
                            buy_quantity = min(volume, max_long_capacity, 5)  # Conservative quantity
                            
                            if buy_quantity > 0:
                                orders.append(Order(target_product, price, buy_quantity))
                                max_long_capacity -= buy_quantity
                                break
                
                # Sell specifically to best buyer
                if max_short_capacity > 0:
                    buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
                    
                    for price in buy_prices:
                        if price >= individual_buyer_price * 0.997:  # Very targeted threshold
                            volume = order_depth.buy_orders[price]
                            sell_quantity = min(volume, max_short_capacity, 5)  # Conservative quantity
                            
                            if sell_quantity > 0:
                                orders.append(Order(target_product, price, -sell_quantity))
                                max_short_capacity -= sell_quantity
                                break
        
        # Add risk management orders
        risk_orders = self.generate_risk_management_orders_squid(
            target_product, order_depth, product_data, current_position, stop_loss_triggered
        )
        orders.extend(risk_orders)
        
        # Position rebalancing if too skewed
        if current_position > position_limit * 0.8 and order_depth.buy_orders:
            # Reduce long position
            best_price = max(order_depth.buy_orders.keys())
            volume = order_depth.buy_orders[best_price]
            sell_quantity = min(volume, int(current_position * 0.3))
            if sell_quantity > 0:
                orders.append(Order(target_product, best_price, -sell_quantity))
        
        elif current_position < -position_limit * 0.8 and order_depth.sell_orders:
            # Reduce short position
            best_price = min(order_depth.sell_orders.keys())
            volume = abs(order_depth.sell_orders[best_price])
            buy_quantity = min(volume, int(abs(current_position) * 0.3))
            if buy_quantity > 0:
                orders.append(Order(target_product, best_price, buy_quantity))
        
        # If no orders were generated, use fallback strategy
        if not orders:
            return self.generate_fallback_orders(
                target_product, order_depth, current_position, max_long_capacity, max_short_capacity
            )
        
        return orders
    
    def generate_fallback_orders(self, product, order_depth, current_position, max_long_capacity, max_short_capacity):
        """Generate fallback orders when the main strategy isn't viable"""
        orders = []
        
        # Basic market data
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # If spread is decent, do market making
        if spread > mid_price * 0.005:  # Spread is more than 0.5% of price
            # Place limit orders inside the spread
            if max_long_capacity > 0:
                limit_buy_price = int(best_bid + spread * 0.25)
                limit_buy_quantity = min(5, max_long_capacity)
                if limit_buy_quantity > 0:
                    orders.append(Order(product, limit_buy_price, limit_buy_quantity))
            
            if max_short_capacity > 0:
                limit_sell_price = int(best_ask - spread * 0.25)
                limit_sell_quantity = min(5, max_short_capacity)
                if limit_sell_quantity > 0:
                    orders.append(Order(product, limit_sell_price, -limit_sell_quantity))
        
        # If position is skewed, try to reduce it
        elif abs(current_position) > 20:
            if current_position > 20 and order_depth.buy_orders:
                # Reduce long position
                best_price = max(order_depth.buy_orders.keys())
                volume = order_depth.buy_orders[best_price]
                sell_quantity = min(volume, int(abs(current_position) * 0.3))
                if sell_quantity > 0:
                    orders.append(Order(product, best_price, -sell_quantity))
            
            elif current_position < -20 and order_depth.sell_orders:
                # Reduce short position
                best_price = min(order_depth.sell_orders.keys())
                volume = abs(order_depth.sell_orders[best_price])
                buy_quantity = min(volume, int(abs(current_position) * 0.3))
                if buy_quantity > 0:
                    orders.append(Order(product, best_price, buy_quantity))
        
        return orders

    def calculate_risk_parameters(self, product_data, current_position, position_limit, mid_price):
        """Calculate risk management parameters for PICNIC_BASKET1"""
        # 1. Position scaling based on current position
        position_scale = 1.0
        if abs(current_position) > position_limit * 0.7:
            position_scale = 0.6  # More conservative scaling
        elif abs(current_position) > position_limit * 0.5:
            position_scale = 0.8
        
        # 2. Stop loss detection
        stop_loss_triggered = False
        average_position_price = product_data.get("average_position_price", 0)
        
        if average_position_price > 0:
            if current_position > 0 and mid_price < average_position_price * 0.98:  # 2% loss
                stop_loss_triggered = True
            elif current_position < 0 and mid_price > average_position_price * 1.02:  # 2% loss
                stop_loss_triggered = True
        
        # 3. Avoid new positions during extreme conditions
        avoid_new_positions = False
        if product_data.get("market_volatility", 0) > 0 and product_data["recent_prices"]:
            avg_price = sum(product_data["recent_prices"]) / len(product_data["recent_prices"])
            relative_volatility = product_data["market_volatility"] / avg_price
            
            if relative_volatility > 0.05:  # 5% volatility is quite high
                avoid_new_positions = True
        
        return position_scale, stop_loss_triggered, avoid_new_positions
    
    def calculate_risk_parameters_squid(self, product_data, current_position, position_limit, mid_price):
        """Calculate risk management parameters for SQUID_INK with tighter stops"""
        # 1. Position scaling based on current position - more conservative for SQUID_INK
        position_scale = 1.0
        if abs(current_position) > position_limit * 0.7:
            position_scale = 0.5  # More conservative scaling than PICNIC_BASKET1
        elif abs(current_position) > position_limit * 0.5:
            position_scale = 0.7  # More conservative scaling than PICNIC_BASKET1
        
        # 2. Stop loss detection - tighter at 2% instead of 3%
        stop_loss_triggered = False
        average_position_price = product_data.get("average_position_price", 0)
        
        if average_position_price > 0:
            if current_position > 0 and mid_price < average_position_price * 0.98:  # 2% loss
                stop_loss_triggered = True
            elif current_position < 0 and mid_price > average_position_price * 1.02:  # 2% loss
                stop_loss_triggered = True
        
        # 3. Avoid new positions during extreme conditions - lower threshold at 5%
        avoid_new_positions = False
        if product_data.get("market_volatility", 0) > 0 and product_data["recent_prices"]:
            avg_price = sum(product_data["recent_prices"]) / len(product_data["recent_prices"])
            relative_volatility = product_data["market_volatility"] / avg_price
            
            if relative_volatility > 0.05:  # 5% volatility threshold as per requirements
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
                # Sell up to 60% of our position
                sell_quantity = min(volume, int(current_position * 0.6))
                if sell_quantity > 0:
                    orders.append(Order(product, best_price, -sell_quantity))
            
            elif current_position < 0 and order_depth.sell_orders:
                # Buy some position at market (best ask)
                best_price = min(order_depth.sell_orders.keys())
                volume = abs(order_depth.sell_orders[best_price])
                # Buy up to 60% of our position
                buy_quantity = min(volume, int(abs(current_position) * 0.6))
                if buy_quantity > 0:
                    orders.append(Order(product, best_price, buy_quantity))
        
        # If position is profitable, consider reducing it
        elif average_position_price > 0:
            # Get mid price
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Profit thresholds
                profit_threshold = 0.015  # 1.5% 
                
                if current_position > 5 and mid_price > average_position_price * (1 + profit_threshold):
                    # Take profit on some position
                    if order_depth.buy_orders:
                        best_price = max(order_depth.buy_orders.keys())
                        volume = order_depth.buy_orders[best_price]
                        # Sell up to 40% of our position
                        sell_quantity = min(volume, int(current_position * 0.4))
                        if sell_quantity > 0:
                            orders.append(Order(product, best_price, -sell_quantity))
                
                elif current_position < -5 and mid_price < average_position_price * (1 - profit_threshold):
                    # Take profit on some position
                    if order_depth.sell_orders:
                        best_price = min(order_depth.sell_orders.keys())
                        volume = abs(order_depth.sell_orders[best_price])
                        # Buy up to 40% of our position
                        buy_quantity = min(volume, int(abs(current_position) * 0.4))
                        if buy_quantity > 0:
                            orders.append(Order(product, best_price, buy_quantity))
        
        return orders
    
    def generate_risk_management_orders_squid(self, product, order_depth, product_data, current_position, stop_loss_triggered):
        """Generate orders for risk management (stop loss and profit taking) specifically for SQUID_INK"""
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
        
        # If stop loss is triggered, reduce position more aggressively for SQUID_INK
        if stop_loss_triggered:
            if current_position > 0 and order_depth.buy_orders:
                # Sell some position at market (best bid)
                best_price = max(order_depth.buy_orders.keys())
                volume = order_depth.buy_orders[best_price]
                # Sell up to 70% of our position (more aggressive than PICNIC_BASKET1)
                sell_quantity = min(volume, int(current_position * 0.7))
                if sell_quantity > 0:
                    orders.append(Order(product, best_price, -sell_quantity))
            
            elif current_position < 0 and order_depth.sell_orders:
                # Buy some position at market (best ask)
                best_price = min(order_depth.sell_orders.keys())
                volume = abs(order_depth.sell_orders[best_price])
                # Buy up to 70% of our position (more aggressive than PICNIC_BASKET1)
                buy_quantity = min(volume, int(abs(current_position) * 0.7))
                if buy_quantity > 0:
                    orders.append(Order(product, best_price, buy_quantity))
        
        # If position is profitable, take profits earlier
        elif average_position_price > 0:
            # Get mid price
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Lower profit threshold for SQUID_INK since we're exploiting counterparty differences
                profit_threshold = 0.01  # 1% (more aggressive than PICNIC_BASKET1's 1.5%)
                
                if current_position > 5 and mid_price > average_position_price * (1 + profit_threshold):
                    # Take profit on some position
                    if order_depth.buy_orders:
                        best_price = max(order_depth.buy_orders.keys())
                        volume = order_depth.buy_orders[best_price]
                        # Sell up to 50% of our position (more than PICNIC_BASKET1's 40%)
                        sell_quantity = min(volume, int(current_position * 0.5))
                        if sell_quantity > 0:
                            orders.append(Order(product, best_price, -sell_quantity))
                
                elif current_position < -5 and mid_price < average_position_price * (1 - profit_threshold):
                    # Take profit on some position
                    if order_depth.sell_orders:
                        best_price = min(order_depth.sell_orders.keys())
                        volume = abs(order_depth.sell_orders[best_price])
                        # Buy up to 50% of our position (more than PICNIC_BASKET1's 40%)
                        buy_quantity = min(volume, int(abs(current_position) * 0.5))
                        if buy_quantity > 0:
                            orders.append(Order(product, best_price, buy_quantity))
        
        return orders
    
    def detect_market_trend(self, recent_prices):
        """Detect market trend based on recent prices"""
        if len(recent_prices) < 10:
            return "NEUTRAL"
            
        early_prices = recent_prices[:5]
        late_prices = recent_prices[-5:]
        avg_early = sum(early_prices) / len(early_prices)
        avg_late = sum(late_prices) / len(late_prices)
        
        # Calculate price volatility in the recent window
        if len(recent_prices) >= 10:
            recent_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                            for i in range(1, len(recent_prices))]
            avg_change = sum(recent_changes) / len(recent_changes)
            
            # If market is too volatile, indicate that
            if avg_change > 0.01:  # 1% average change between ticks is volatile
                return "VOLATILE"
        
        if avg_late > avg_early * 1.005:  # 0.5% increase
            return "UP"
        elif avg_late < avg_early * 0.995:  # 0.5% decrease
            return "DOWN"
        return "NEUTRAL"
from round_5.datamodel import OrderDepth, UserId, TradingState, Order
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
        self.market_volatility = {}
        self.trade_count = {}
        self.last_conversion_time = {}
        self.consecutive_no_trade = {}
        self.price_trend = {}
        self.pnl_history = {}
        self.last_fair_values = {}
        self.market_direction = {}
        self.best_pnl = 0
        self.position_lock = False

    def run(self, state: TradingState):
        # Initialize or restore the state
        if state.traderData != "":
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.product_positions = saved_state.get('product_positions', {})
                self.fair_value_history = saved_state.get('fair_value_history', {})
                self.mid_price_history = saved_state.get('mid_price_history', {})
                self.position_history = saved_state.get('position_history', {})
                self.last_observations = saved_state.get('last_observations', {})
                self.market_volatility = saved_state.get('market_volatility', {})
                self.trade_count = saved_state.get('trade_count', {})
                self.last_conversion_time = saved_state.get('last_conversion_time', {})
                self.consecutive_no_trade = saved_state.get('consecutive_no_trade', {})
                self.price_trend = saved_state.get('price_trend', {})
                self.pnl_history = saved_state.get('pnl_history', {})
                self.last_fair_values = saved_state.get('last_fair_values', {})
                self.market_direction = saved_state.get('market_direction', {})
                self.best_pnl = saved_state.get('best_pnl', 0)
                self.position_lock = saved_state.get('position_lock', False)
            except Exception as e:
                print(f"Error restoring state: {e}")
                # Initialize if restoration fails
                self.product_positions = {}
                self.fair_value_history = {}
                self.mid_price_history = {}
                self.position_history = {}
                self.last_observations = {}
                self.market_volatility = {}
                self.trade_count = {}
                self.last_conversion_time = {}
                self.consecutive_no_trade = {}
                self.price_trend = {}
                self.pnl_history = {}
                self.last_fair_values = {}
                self.market_direction = {}
                self.best_pnl = 0
                self.position_lock = False

        # Record current positions and initialize new products
        for product, pos in state.position.items():
            self.product_positions[product] = pos
            
            # Initialize history and tracking for new products
            for dict_name in ['fair_value_history', 'mid_price_history', 'position_history', 'price_trend']:
                dict_obj = getattr(self, dict_name)
                if product not in dict_obj:
                    dict_obj[product] = []
            
            # Initialize scalar tracking variables
            for scalar_name in ['market_volatility', 'trade_count', 'consecutive_no_trade', 'market_direction']:
                scalar_obj = getattr(self, scalar_name)
                if product not in scalar_obj:
                    scalar_obj[product] = 0
            
            # Initialize other tracking structures
            if product not in self.last_conversion_time:
                self.last_conversion_time[product] = 0
                
            if product not in self.pnl_history:
                self.pnl_history[product] = []
            
            if product not in self.last_fair_values:
                self.last_fair_values[product] = []
            
            # Track position history
            self.position_history[product].append(pos)
        
        # Track profit and loss over time
        current_pnl = 0
        
        # Orders to be placed on exchange matching engine
        result = {}
        
        # Process each product
        for product in state.order_depths:
            # For now, we'll focus specifically on MAGNIFICENT_MACARONS
            if product == "MAGNIFICENT_MACARONS":
                orders = self.trade_magnificent_macarons(
                    product, 
                    state.order_depths[product], 
                    state.position.get(product, 0),
                    state.observations,
                    current_pnl
                )
                result[product] = orders
            else:
                # For other products, initialize empty order list
                result[product] = []
        
        # Store the observations for use in fair value calculation
        if state.observations:
            if hasattr(state.observations, 'conversionObservations'):
                for product, obs in state.observations.conversionObservations.items():
                    self.last_observations[product] = obs
        
        # Determine if conversion is needed for MAGNIFICENT_MACARONS
        conversion_count = self.determine_conversion_need(
            "MAGNIFICENT_MACARONS", 
            state.position.get("MAGNIFICENT_MACARONS", 0),
            state.timestamp,
            current_pnl
        )
        
        # Save the state
        trader_data = {
            'product_positions': self.product_positions,
            'fair_value_history': self.fair_value_history,
            'mid_price_history': self.mid_price_history,
            'position_history': self.position_history,
            'last_observations': self.last_observations,
            'market_volatility': self.market_volatility,
            'trade_count': self.trade_count,
            'last_conversion_time': self.last_conversion_time,
            'consecutive_no_trade': self.consecutive_no_trade,
            'price_trend': self.price_trend,
            'pnl_history': self.pnl_history,
            'last_fair_values': self.last_fair_values,
            'market_direction': self.market_direction,
            'best_pnl': self.best_pnl,
            'position_lock': self.position_lock
        }
        
        traderData = jsonpickle.encode(trader_data)
        
        return result, conversion_count, traderData
    
    def trade_magnificent_macarons(self, product, order_depth, position, observations, current_pnl):
        """
        Enhanced strategy for trading MAGNIFICENT_MACARONS to maintain upward profit trend
        """
        # Initialize return list of orders
        orders = []
        
        # Initialize the product in history dictionaries if not already present
        if product not in self.mid_price_history:
            self.mid_price_history[product] = []
        if product not in self.fair_value_history:
            self.fair_value_history[product] = []
        if product not in self.position_history:
            self.position_history[product] = []
        if product not in self.consecutive_no_trade:
            self.consecutive_no_trade[product] = 0
        if product not in self.price_trend:
            self.price_trend[product] = []
        if product not in self.last_fair_values:
            self.last_fair_values[product] = []
        
        # Get the observations for MAGNIFICENT_MACARONS
        obs = None
        if hasattr(observations, 'conversionObservations') and product in observations.conversionObservations:
            obs = observations.conversionObservations[product]
            self.last_observations[product] = obs
        elif product in self.last_observations:
            obs = self.last_observations[product]
            
        # Position limit for MAGNIFICENT_MACARONS is 75
        POSITION_LIMIT = 75
        CONVERSION_LIMIT = 10
        
        # Calculate mid price from the order book
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price:
            self.mid_price_history[product].append(mid_price)
            
            # Track price trend for this product
            self.price_trend[product].append(mid_price)
            if len(self.price_trend[product]) > 20:
                self.price_trend[product].pop(0)  # Keep only the most recent 20 prices
                
            # Calculate market direction and volatility
            self.calculate_market_dynamics(product)
        
        # Calculate fair value based on observations, historical data, and PnL trend
        fair_value = self.calculate_fair_value_macarons(product, order_depth, obs, mid_price, current_pnl)
        if fair_value:
            self.fair_value_history[product].append(fair_value)
            self.last_fair_values[product].append(fair_value)
            if len(self.last_fair_values[product]) > 10:
                self.last_fair_values[product].pop(0)
        else:
            # If we can't calculate fair value, use mid price as fair value
            fair_value = mid_price
            if mid_price:
                self.fair_value_history[product].append(mid_price)
                self.last_fair_values[product].append(mid_price)
                if len(self.last_fair_values[product]) > 10:
                    self.last_fair_values[product].pop(0)
            
        # If we don't have a fair value, we can't trade
        if not fair_value:
            self.consecutive_no_trade[product] += 1
            return orders
            
        # Get highest bid and lowest ask
        best_bid, best_bid_amount = self.get_best_bid(order_depth)
        best_ask, best_ask_amount = self.get_best_ask(order_depth)
        
        # If there are no orders on one side, we can't calculate a reasonable fair value
        if not best_bid or not best_ask:
            self.consecutive_no_trade[product] += 1
            return orders
            
        # Calculate bid-ask spread
        spread = best_ask - best_bid
        
        # Dynamic threshold based on market volatility, position, and PnL trend
        position_factor = abs(position) / POSITION_LIMIT  # 0 to 1
        volatility_factor = self.market_volatility.get(product, 0.5)  # Default if no volatility data
        
        # PnL trend factor - more aggressive when PnL is declining
        pnl_trend_factor = 0
        if len(self.pnl_history.get(product, [])) >= 3:
            recent_pnl = self.pnl_history[product][-3:]
            if recent_pnl[-1] < recent_pnl[0]:  # PnL is declining
                pnl_trend_factor = 0.3  # More aggressive to reverse the trend
        
        # Base threshold plus adjustments
        base_threshold = 0.3  # Reduced base threshold to be more aggressive in trading
        position_adjustment = position_factor * 0.5  # More conservative near limits
        volatility_adjustment = volatility_factor * 0.3  # More aggressive in volatile markets
        
        # Threshold decreases with consecutive no-trades to ensure we continue trading
        no_trade_factor = min(self.consecutive_no_trade.get(product, 0) * 0.1, 0.5)
        
        # Adjust threshold based on market direction
        direction_factor = 0
        if self.market_direction.get(product, 0) != 0:
            # In strongly trending markets, adjust threshold to trade with the trend
            direction_factor = 0.2 * self.market_direction.get(product, 0)
        
        threshold = max(0.1, base_threshold + position_adjustment - volatility_adjustment - 
                         no_trade_factor - pnl_trend_factor + direction_factor)
        
        # Track if we made any trades this iteration
        made_trade = False
        
        # Determine if we should lock position when PnL is at peak
        # If we're within 95% of our best PnL and would otherwise lose money, be more conservative
        if current_pnl > 0 and self.best_pnl > 0 and (current_pnl / self.best_pnl) >= 0.97:
            # We're close to our peak PnL, be more conservative
            if len(self.pnl_history.get(product, [])) >= 3:
                recent_pnl = self.pnl_history[product][-3:]
                if recent_pnl[-1] < recent_pnl[-2] < recent_pnl[-3]:  # Clear downtrend
                    # If PnL is consistently declining, lock in profits
                    self.position_lock = True
                    print(f"POSITION LOCK ENGAGED: Current PnL: {current_pnl}, Best PnL: {self.best_pnl}")
        
        # If PnL starts increasing again, release the lock
        if self.position_lock and len(self.pnl_history.get(product, [])) >= 2:
            if self.pnl_history[product][-1] > self.pnl_history[product][-2]:
                self.position_lock = False
                print("POSITION LOCK RELEASED: PnL improving")
        
        # If position is locked, focus on reducing position rather than new trades
        if self.position_lock and abs(position) > 0:
            # Reduce position if we can get a decent price
            if position > 0 and best_bid > mid_price:  # We're long, look to sell
                sell_quantity = min(best_bid_amount, position)
                orders.append(Order(product, best_bid, -sell_quantity))
                position -= sell_quantity
                made_trade = True
                print(f"POSITION LOCK SELL {sell_quantity} @ {best_bid}")
            elif position < 0 and best_ask < mid_price:  # We're short, look to buy
                buy_quantity = min(-best_ask_amount, -position)
                orders.append(Order(product, best_ask, buy_quantity))
                position += buy_quantity
                made_trade = True
                print(f"POSITION LOCK BUY {buy_quantity} @ {best_ask}")
            
            # If we made trades to reduce position, return early
            if made_trade:
                self.consecutive_no_trade[product] = 0
                self.trade_count[product] = self.trade_count.get(product, 0) + 1
                self.product_positions[product] = position
                return orders
        
        # NORMAL TRADING LOGIC (when not position locked)
        
        # 1. DIRECTIONAL TRADING: Based on fair value
        
        # SELLING LOGIC - modified to be more profitable
        if best_bid > fair_value + threshold:
            if position > -POSITION_LIMIT:  # Can still sell more
                # Adjust quantity based on confidence and position
                confidence = min(1.0, (best_bid - fair_value) / spread)
                sell_quantity = min(best_bid_amount, POSITION_LIMIT + position)
                # Scale sell quantity by confidence when not near position limit
                if position > -POSITION_LIMIT * 0.7:
                    sell_quantity = max(1, int(sell_quantity * max(0.2, confidence)))
                
                orders.append(Order(product, best_bid, -sell_quantity))
                position -= sell_quantity
                made_trade = True
                print(f"SELL {sell_quantity} @ {best_bid} (Fair value: {fair_value}, Confidence: {confidence:.2f})")
        
        # BUYING LOGIC - modified to be more profitable
        if best_ask < fair_value - threshold:
            if position < POSITION_LIMIT:  # Can still buy more
                # Adjust quantity based on confidence and position
                confidence = min(1.0, (fair_value - best_ask) / spread)
                buy_quantity = min(-best_ask_amount, POSITION_LIMIT - position)
                # Scale buy quantity by confidence when not near position limit
                if position < POSITION_LIMIT * 0.7:
                    buy_quantity = max(1, int(buy_quantity * max(0.2, confidence)))
                
                orders.append(Order(product, best_ask, buy_quantity))
                position += buy_quantity
                made_trade = True
                print(f"BUY {buy_quantity} @ {best_ask} (Fair value: {fair_value}, Confidence: {confidence:.2f})")
        
        # 2. POSITION REBALANCING: If position getting close to limits
        # This helps prevent getting stuck at position limits
        
        position_rebalance_threshold = 0.75 * POSITION_LIMIT  # 75% of position limit
        
        if position >= position_rebalance_threshold and not made_trade:
            # If we're very long, look to sell some to bring position back down
            if best_bid > mid_price:  # Only sell above mid price
                # Calculate how much to sell based on how close we are to limit
                position_pct = position / POSITION_LIMIT
                sell_quantity = min(best_bid_amount, max(3, int(position * (position_pct * 0.3))))
                orders.append(Order(product, best_bid, -sell_quantity))
                position -= sell_quantity
                made_trade = True
                print(f"REBALANCE SELL {sell_quantity} @ {best_bid} (Position: {position}, {position_pct*100:.1f}% of limit)")
        
        elif position <= -position_rebalance_threshold and not made_trade:
            # If we're very short, look to buy some to bring position back up
            if best_ask < mid_price:  # Only buy below mid price
                # Calculate how much to buy based on how close we are to limit
                position_pct = abs(position) / POSITION_LIMIT
                buy_quantity = min(-best_ask_amount, max(3, int(abs(position) * (position_pct * 0.3))))
                orders.append(Order(product, best_ask, buy_quantity))
                position += buy_quantity
                made_trade = True
                print(f"REBALANCE BUY {buy_quantity} @ {best_ask} (Position: {position}, {position_pct*100:.1f}% of limit)")
        
        # 3. MARKET MAKING: If we haven't made trades yet and spread is sufficient
        # Only do this if we're not too close to position limits and spread is profitable
        
        min_profitable_spread = 1.5  # Reduced min spread requirement for market making
        
        if not made_trade and spread > min_profitable_spread:
            # We have room in both directions
            if abs(position) < (POSITION_LIMIT * 0.6):  # Not too close to limits
                # Place orders on both sides with a slight edge
                bid_price = best_bid + 1  # Outbid the current best bid
                ask_price = best_ask - 1  # Undercut the current best ask
                
                # Calculate quantities based on current position and market direction
                # If long, bid less and ask more; if short, bid more and ask less
                position_bias = position / POSITION_LIMIT  # -1 to 1
                direction_bias = self.market_direction.get(product, 0) * 0.3  # Favor trading with trend
                
                # Combine position bias and direction bias
                combined_bias = position_bias - direction_bias
                
                base_quantity = 5  # Base quantity for market making
                buy_quantity = max(1, int(base_quantity * (1 - combined_bias)))
                sell_quantity = max(1, int(base_quantity * (1 + combined_bias)))
                
                # Ensure we don't exceed position limits
                buy_quantity = min(buy_quantity, POSITION_LIMIT - position)
                sell_quantity = min(sell_quantity, POSITION_LIMIT + position)
                
                if buy_quantity > 0:
                    orders.append(Order(product, bid_price, buy_quantity))
                    print(f"MM BUY {buy_quantity} @ {bid_price}")
                    
                if sell_quantity > 0:
                    orders.append(Order(product, ask_price, -sell_quantity))
                    print(f"MM SELL {sell_quantity} @ {ask_price}")
                    
                made_trade = True
        
        # 4. TREND FOLLOWING: If we're still not trading and have had multiple no-trades
        # This helps break out of situations where we're not trading at all
        
        if not made_trade and self.consecutive_no_trade.get(product, 0) >= 2:
            # Look at recent price trend
            if len(self.price_trend.get(product, [])) >= 3:
                recent_prices = self.price_trend[product][-3:]
                
                # Check if we've identified a clear market direction
                market_dir = self.market_direction.get(product, 0)
                
                if market_dir > 0.5:  # Strong uptrend
                    # Buy on uptrend if we have room
                    if position < POSITION_LIMIT * 0.85:
                        buy_quantity = min(-best_ask_amount, max(2, (POSITION_LIMIT - position) // 8))
                        orders.append(Order(product, best_ask, buy_quantity))
                        position += buy_quantity
                        made_trade = True
                        print(f"TREND BUY {buy_quantity} @ {best_ask} (Strong Uptrend)")
                        
                elif market_dir < -0.5:  # Strong downtrend
                    # Sell on downtrend if we have room
                    if position > -POSITION_LIMIT * 0.85:
                        sell_quantity = min(best_bid_amount, max(2, (POSITION_LIMIT + position) // 8))
                        orders.append(Order(product, best_bid, -sell_quantity))
                        position -= sell_quantity
                        made_trade = True
                        print(f"TREND SELL {sell_quantity} @ {best_bid} (Strong Downtrend)")
        
        # 5. AGGRESSIVE PNL RECOVERY: If PnL has been declining for multiple periods
        if not made_trade and len(self.pnl_history.get(product, [])) >= 4:
            recent_pnl = self.pnl_history[product][-4:]
            if recent_pnl[-1] < recent_pnl[-2] < recent_pnl[-3]:  # Clear downtrend in PnL
                print("ACTIVATING PNL RECOVERY MODE")
                
                # If our fair value has been changing, use that to determine direction
                fair_value_direction = 0
                if len(self.last_fair_values.get(product, [])) >= 3:
                    recent_fv = self.last_fair_values[product][-3:]
                    if recent_fv[-1] > recent_fv[0]:
                        fair_value_direction = 1
                    elif recent_fv[-1] < recent_fv[0]:
                        fair_value_direction = -1
                
                # If we have a clear direction, trade with it more aggressively
                if fair_value_direction > 0 and position < POSITION_LIMIT * 0.9:
                    # Market likely to go up, buy more
                    buy_quantity = min(-best_ask_amount, max(3, (POSITION_LIMIT - position) // 5))
                    orders.append(Order(product, best_ask, buy_quantity))
                    position += buy_quantity
                    made_trade = True
                    print(f"RECOVERY BUY {buy_quantity} @ {best_ask} (FV Trend: Up)")
                    
                elif fair_value_direction < 0 and position > -POSITION_LIMIT * 0.9:
                    # Market likely to go down, sell more
                    sell_quantity = min(best_bid_amount, max(3, (POSITION_LIMIT + position) // 5))
                    orders.append(Order(product, best_bid, -sell_quantity))
                    position -= sell_quantity
                    made_trade = True
                    print(f"RECOVERY SELL {sell_quantity} @ {best_bid} (FV Trend: Down)")
        
        # Update trading metrics and position tracking
        if made_trade:
            self.consecutive_no_trade[product] = 0
            self.trade_count[product] = self.trade_count.get(product, 0) + 1
        else:
            self.consecutive_no_trade[product] += 1
            
        # Update our position tracking
        self.product_positions[product] = position
        
        return orders
        if product not in self.position_history:
            self.position_history[product] = []
        if product not in self.consecutive_no_trade:
            self.consecutive_no_trade[product] = 0
        
        # Get the observations for MAGNIFICENT_MACARONS
        obs = None
        if hasattr(observations, 'conversionObservations') and product in observations.conversionObservations:
            obs = observations.conversionObservations[product]
            self.last_observations[product] = obs
        elif product in self.last_observations:
            obs = self.last_observations[product]
            
        # Position limit for MAGNIFICENT_MACARONS is 75
        POSITION_LIMIT = 75
        CONVERSION_LIMIT = 10
        
        # Calculate mid price from the order book
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price:
            self.mid_price_history[product].append(mid_price)
            # Calculate market volatility using recent price movements
            if len(self.mid_price_history[product]) > 1:
                recent_prices = self.mid_price_history[product][-10:]
                if len(recent_prices) > 1:
                    price_changes = [abs(recent_prices[i] - recent_prices[i-1]) for i in range(1, len(recent_prices))]
                    self.market_volatility[product] = sum(price_changes) / len(price_changes)
                else:
                    self.market_volatility[product] = 0.5  # Default value
            else:
                self.market_volatility[product] = 0.5  # Default value
        
        # Calculate fair value based on observations and historical data
        fair_value = self.calculate_fair_value_macarons(product, order_depth, obs, mid_price)
        if fair_value:
            self.fair_value_history[product].append(fair_value)
        else:
            # If we can't calculate fair value, use mid price as fair value
            fair_value = mid_price
            if mid_price:
                self.fair_value_history[product].append(mid_price)
            
        # If we don't have a fair value, we can't trade
        if not fair_value:
            self.consecutive_no_trade[product] += 1
            return orders
            
        # Get highest bid and lowest ask
        best_bid, best_bid_amount = self.get_best_bid(order_depth)
        best_ask, best_ask_amount = self.get_best_ask(order_depth)
        
        # If there are no orders on one side, we can't calculate a reasonable fair value
        if not best_bid or not best_ask:
            self.consecutive_no_trade[product] += 1
            return orders
            
        # Calculate bid-ask spread
        spread = best_ask - best_bid
        
        # Dynamic threshold based on market volatility and position
        # Higher threshold when close to position limits
        position_factor = abs(position) / POSITION_LIMIT  # 0 to 1
        volatility_factor = self.market_volatility.get(product, 0.5)  # Default if no volatility data
        
        # Base threshold plus adjustments
        base_threshold = 0.5
        position_adjustment = position_factor * 0.5  # More conservative near limits
        volatility_adjustment = volatility_factor * 0.5  # More aggressive in volatile markets
        
        # Threshold decreases with consecutive no-trades to ensure we continue trading
        no_trade_factor = min(self.consecutive_no_trade.get(product, 0) * 0.1, 0.5)
        
        threshold = max(0.1, base_threshold + position_adjustment - volatility_adjustment - no_trade_factor)
        
        # Track if we made any trades this iteration
        made_trade = False
        
        # TRADING LOGIC
        
        # 1. DIRECTIONAL TRADING: Based on fair value
        
        # SELLING LOGIC
        if best_bid > fair_value + threshold:
            if position > -POSITION_LIMIT:  # Can still sell more
                # Calculate how many we can sell
                sell_quantity = min(best_bid_amount, POSITION_LIMIT + position)
                orders.append(Order(product, best_bid, -sell_quantity))
                position -= sell_quantity
                made_trade = True
                print(f"SELL {sell_quantity} @ {best_bid} (Fair value: {fair_value})")
        
        # BUYING LOGIC
        if best_ask < fair_value - threshold:
            if position < POSITION_LIMIT:  # Can still buy more
                # Calculate how many we can buy
                buy_quantity = min(-best_ask_amount, POSITION_LIMIT - position)
                orders.append(Order(product, best_ask, buy_quantity))
                position += buy_quantity
                made_trade = True
                print(f"BUY {buy_quantity} @ {best_ask} (Fair value: {fair_value})")
        
        # 2. POSITION REBALANCING: If position getting close to limits
        # This helps prevent getting stuck at position limits
        
        position_rebalance_threshold = 0.8 * POSITION_LIMIT  # 80% of position limit
        
        if position >= position_rebalance_threshold and not made_trade:
            # If we're very long, look to sell some to bring position back down
            if best_bid > mid_price + (spread * 0.3):  # Still get decent price
                sell_quantity = min(best_bid_amount, max(5, position // 5))  # Sell at least 5 or 20% of position
                orders.append(Order(product, best_bid, -sell_quantity))
                position -= sell_quantity
                made_trade = True
                print(f"REBALANCE SELL {sell_quantity} @ {best_bid} (Position: {position})")
        
        elif position <= -position_rebalance_threshold and not made_trade:
            # If we're very short, look to buy some to bring position back up
            if best_ask < mid_price - (spread * 0.3):  # Still get decent price
                buy_quantity = min(-best_ask_amount, max(5, abs(position) // 5))  # Buy at least 5 or 20% of position
                orders.append(Order(product, best_ask, buy_quantity))
                position += buy_quantity
                made_trade = True
                print(f"REBALANCE BUY {buy_quantity} @ {best_ask} (Position: {position})")
        
        # 3. MARKET MAKING: If we haven't made trades yet and spread is sufficient
        # Only do this if we're not too close to position limits and spread is profitable
        
        min_profitable_spread = 2.0  # Minimum spread for market making to be profitable
        
        if not made_trade and spread > min_profitable_spread:
            # We have room in both directions
            if abs(position) < (POSITION_LIMIT * 0.7):  # Not too close to limits
                # Place orders on both sides with a slight edge
                bid_price = best_bid + 1  # Outbid the current best bid
                ask_price = best_ask - 1  # Undercut the current best ask
                
                # Calculate quantities based on current position
                # If long, bid less and ask more; if short, bid more and ask less
                position_bias = position / POSITION_LIMIT  # -1 to 1
                
                base_quantity = 5  # Base quantity for market making
                buy_quantity = max(1, int(base_quantity * (1 - position_bias)))
                sell_quantity = max(1, int(base_quantity * (1 + position_bias)))
                
                # Ensure we don't exceed position limits
                buy_quantity = min(buy_quantity, POSITION_LIMIT - position)
                sell_quantity = min(sell_quantity, POSITION_LIMIT + position)
                
                if buy_quantity > 0:
                    orders.append(Order(product, bid_price, buy_quantity))
                    print(f"MM BUY {buy_quantity} @ {bid_price}")
                    
                if sell_quantity > 0:
                    orders.append(Order(product, ask_price, -sell_quantity))
                    print(f"MM SELL {sell_quantity} @ {ask_price}")
                    
                made_trade = True
        
        # 4. TREND FOLLOWING: If we're still not trading and have had multiple no-trades
        # This helps break out of situations where we're not trading at all
        
        if not made_trade and self.consecutive_no_trade.get(product, 0) >= 3:
            # Look at recent price trend
            if len(self.mid_price_history[product]) >= 3:
                recent_prices = self.mid_price_history[product][-3:]
                if recent_prices[-1] > recent_prices[0]:  # Uptrend
                    # Buy on uptrend if we have room
                    if position < POSITION_LIMIT * 0.9:
                        buy_quantity = min(-best_ask_amount, max(2, (POSITION_LIMIT - position) // 10))
                        orders.append(Order(product, best_ask, buy_quantity))
                        position += buy_quantity
                        made_trade = True
                        print(f"TREND BUY {buy_quantity} @ {best_ask} (Uptrend)")
                        
                elif recent_prices[-1] < recent_prices[0]:  # Downtrend
                    # Sell on downtrend if we have room
                    if position > -POSITION_LIMIT * 0.9:
                        sell_quantity = min(best_bid_amount, max(2, (POSITION_LIMIT + position) // 10))
                        orders.append(Order(product, best_bid, -sell_quantity))
                        position -= sell_quantity
                        made_trade = True
                        print(f"TREND SELL {sell_quantity} @ {best_bid} (Downtrend)")
        
        # Update trading metrics and position tracking
        if made_trade:
            self.consecutive_no_trade[product] = 0
            self.trade_count[product] = self.trade_count.get(product, 0) + 1
        else:
            self.consecutive_no_trade[product] += 1
            
        # Update our position tracking
        self.product_positions[product] = position
        
        return orders
    
    def calculate_fair_value_macarons(self, product, order_depth, observations, mid_price):
        """
        Enhanced fair value calculation for MAGNIFICENT_MACARONS based on observations,
        market data, and position-aware adjustments
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
            
        if hasattr(observations, 'importTariff'):
            # Include import tariff if available (even though correlation was weaker)
            fair_value += observations.importTariff * (-2.0) * 0.03  # Negative correlation
            weight_sum += 0.03
            
        if hasattr(observations, 'transportFees'):
            # Include transport fees if available
            fair_value += observations.transportFees * (-1.0) * 0.02  # Negative impact on price
            weight_sum += 0.02
        
        # If we couldn't calculate any components of fair value, return mid price
        if weight_sum == 0:
            return mid_price
            
        # Normalize the fair value by dividing by the sum of weights used
        fair_value /= weight_sum
        
        # Apply historical smoothing for stability
        if product in self.fair_value_history and len(self.fair_value_history[product]) > 0:
            # Calculate a weighted average of current and historical fair values
            # More weight to current calculation, some weight to history for stability
            historical_avg = sum(self.fair_value_history[product][-5:]) / min(5, len(self.fair_value_history[product]))
            fair_value = 0.7 * fair_value + 0.3 * historical_avg
            
        # Apply position-based fair value adjustment
        # This helps prevent getting stuck at position limits by making the algorithm
        # more eager to trade in the direction that reduces extreme positions
        position = self.product_positions.get(product, 0)
        position_limit = 75  # For MAGNIFICENT_MACARONS
        
        if abs(position) > position_limit * 0.7:  # If we're using >70% of our position limit
            # Calculate position bias - ranges from -1 to 1
            position_bias = position / position_limit
            
            # Adjust fair value based on position bias
            # If very long, decrease fair value to encourage selling
            # If very short, increase fair value to encourage buying
            adjustment_factor = -position_bias * 0.5 * (abs(position) / position_limit)
            max_adjustment = self.market_volatility.get(product, 0.5)  # Limit adjustment based on volatility
            
            adjustment = max(-max_adjustment, min(max_adjustment, adjustment_factor))
            fair_value += adjustment
            
        # If we've had several iterations without trading, make fair value more aggressive
        no_trade_count = self.consecutive_no_trade.get(product, 0)
        if no_trade_count >= 4:
            # Gradually increase aggressiveness if we're not trading
            # If long, push fair value down; if short, push fair value up
            no_trade_adjustment = -position_bias * 0.1 * (no_trade_count - 3)
            fair_value += no_trade_adjustment
        
        return fair_value
        
    def calculate_market_dynamics(self, product):
        """
        Calculate market direction and volatility based on price trends
        """
        if product not in self.price_trend or len(self.price_trend[product]) < 6:
            return
            
        # Calculate short-term and medium-term trends
        short_term = self.price_trend[product][-3:]
        medium_term = self.price_trend[product][-6:]
        
        # Calculate direction: 1 (uptrend), -1 (downtrend), 0 (sideways)
        short_direction = 1 if short_term[-1] > short_term[0] else -1 if short_term[-1] < short_term[0] else 0
        medium_direction = 1 if medium_term[-1] > medium_term[0] else -1 if medium_term[-1] < medium_term[0] else 0
        
        # Combined direction with more weight to recent movements
        combined_direction = (short_direction * 0.7) + (medium_direction * 0.3)
        self.market_direction[product] = combined_direction
        
        # Calculate volatility as average price movement
        price_changes = [abs(self.price_trend[product][i] - self.price_trend[product][i-1]) 
                         for i in range(1, len(self.price_trend[product]))]
        if price_changes:
            volatility = sum(price_changes) / len(price_changes)
            # Normalize volatility to a 0-1 scale where 1 is high volatility
            norm_volatility = min(1.0, volatility / 5.0)  # Assuming average movement of 5 is high
            self.market_volatility[product] = norm_volatility
        else:
            self.market_volatility[product] = 0.2  # Default value

    def calculate_fair_value_macarons(self, product, order_depth, observations, mid_price, current_pnl):
        """
        Enhanced fair value calculation with PnL trend awareness
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
            
        if hasattr(observations, 'importTariff'):
            # Include import tariff if available (even though correlation was weaker)
            fair_value += observations.importTariff * (-2.0) * 0.03  # Negative correlation
            weight_sum += 0.03
            
        if hasattr(observations, 'transportFees'):
            # Include transport fees if available
            fair_value += observations.transportFees * (-1.0) * 0.02  # Negative impact on price
            weight_sum += 0.02
        
        # If we couldn't calculate any components of fair value, return mid price
        if weight_sum == 0:
            return mid_price
            
        # Normalize the fair value by dividing by the sum of weights used
        fair_value /= weight_sum
        
        # Apply historical smoothing for stability
        if product in self.fair_value_history and len(self.fair_value_history[product]) > 0:
            # Calculate a weighted average of current and historical fair values
            # More weight to current calculation, some weight to history for stability
            historical_avg = sum(self.fair_value_history[product][-5:]) / min(5, len(self.fair_value_history[product]))
            fair_value = 0.7 * fair_value + 0.3 * historical_avg
            
        # Apply position-based fair value adjustment
        position = self.product_positions.get(product, 0)
        position_limit = 75  # For MAGNIFICENT_MACARONS
        
        if abs(position) > position_limit * 0.7:  # If we're using >70% of our position limit
            # Calculate position bias - ranges from -1 to 1
            position_bias = position / position_limit
            
            # Adjust fair value based on position bias
            # If very long, decrease fair value to encourage selling
            # If very short, increase fair value to encourage buying
            adjustment_factor = -position_bias * 0.5 * (abs(position) / position_limit)
            max_adjustment = self.market_volatility.get(product, 0.5)  # Limit adjustment based on volatility
            
            adjustment = max(-max_adjustment, min(max_adjustment, adjustment_factor))
            fair_value += adjustment
            
        # If PnL is declining, adjust fair value to encourage more aggressive trading
        if len(self.pnl_history.get(product, [])) >= 3:
            recent_pnl = self.pnl_history[product][-3:]
            if recent_pnl[-1] < recent_pnl[-2]:  # PnL is declining
                # If market is trending, adjust fair value to trade more with the trend
                market_dir = self.market_direction.get(product, 0)
                if abs(market_dir) > 0.3:  # Clear trend
                    # Adjust fair value in the direction of the trend to be more aggressive
                    trend_adjustment = market_dir * 0.5  # Up to 0.5 point adjustment
                    fair_value += trend_adjustment
                    print(f"PNL TREND ADJUSTMENT: {trend_adjustment:.2f} (Market dir: {market_dir:.2f})")
            
        # If we've had several iterations without trading, make fair value more aggressive
        no_trade_count = self.consecutive_no_trade.get(product, 0)
        if no_trade_count >= 3:
            # Gradually increase aggressiveness if we're not trading
            no_trade_adjustment = 0.1 * (no_trade_count - 2)  # Start small and increase
            
            # Adjust based on market direction to trade with the trend
            market_dir = self.market_direction.get(product, 0)
            if abs(market_dir) > 0.3:
                fair_value += market_dir * no_trade_adjustment
                print(f"NO TRADE ADJUSTMENT: {market_dir * no_trade_adjustment:.2f}")
        
        return fair_value
        
    def determine_conversion_need(self, product, position, timestamp, current_pnl):
        """
        Determine if we need to convert MAGNIFICENT_MACARONS based on position and profitability
        """
        if product != "MAGNIFICENT_MACARONS":
            return 0
            
        # Conversion limit for MAGNIFICENT_MACARONS is 10
        CONVERSION_LIMIT = 10
        
        # Only consider conversion if:
        # 1. We're near position limits
        # 2. It's been at least 3 iterations since last conversion (reduced from 5)
        # 3. We have enough position to convert
        # 4. PnL is declining, and we need to free up position capacity
        
        # Check if we're close to position limit
        position_limit = 75
        position_pct = abs(position) / position_limit
        
        # Time since last conversion
        time_since_last = timestamp - self.last_conversion_time.get(product, 0)
        
        # Check PnL trend
        pnl_declining = False
        if len(self.pnl_history.get(product, [])) >= 3:
            recent_pnl = self.pnl_history[product][-3:]
            if recent_pnl[-1] < recent_pnl[-2]:
                pnl_declining = True
        
        # If we're using at least 65% of our position limit (reduced from 70%)
        if position_pct >= 0.65 and time_since_last >= 3:
            # Check if we have enough position for conversion
            if abs(position) >= CONVERSION_LIMIT:
                # Update last conversion time
                self.last_conversion_time[product] = timestamp
                
                # If we're long, convert to sell
                if position > 0:
                    print(f"CONVERTING {CONVERSION_LIMIT} UNITS (Long Position: {position})")
                    return CONVERSION_LIMIT
                # If we're short, convert to buy
                else:
                    print(f"CONVERTING {CONVERSION_LIMIT} UNITS (Short Position: {position})")
                    return CONVERSION_LIMIT
        
        # If PnL is declining significantly and we have position to convert
        if pnl_declining and abs(position) >= CONVERSION_LIMIT and time_since_last >= 3:
            self.last_conversion_time[product] = timestamp
            direction = 1 if position > 0 else -1
            print(f"PNL-BASED CONVERSION: {CONVERSION_LIMIT} units (PnL trend: declining)")
            return CONVERSION_LIMIT * direction
            
        # No conversion needed
        return 0
    
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
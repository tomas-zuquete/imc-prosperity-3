from round_5.sub.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle

class Trader:
    def __init__(self):
        # Core tracking
        self.positions = {}
        self.position_history = []
        self.price_history = {}
        self.pnl_history = []
        self.last_pnl = 0
        self.best_pnl = 0
        
        # Observation tracking
        self.observation_history = {
            'sugarPrice': [],
            'sunlightIndex': [],
            'exportTariff': [],
            'importTariff': [],
            'transportFees': []
        }
        
        # Factor weights based on correlation analysis
        self.factor_weights = {
            'sunlightIndex': 10.5,  # Strong positive correlation
            'sugarPrice': 3.2,      # Moderate positive correlation 
            'exportTariff': 7.5,    # Strong positive correlation
            'importTariff': -2.0,   # Weak negative correlation
            'transportFees': -1.0   # Weak negative correlation
        }
        
        # Position limits and tracking
        self.max_position = 70  # Slightly below the 75 limit for safety
        self.preferred_position = -40  # Maintain a net short bias in bear market
        
        # Market tracking
        self.market_phase = "discovery"  # Start in discovery phase
        self.phase_counter = 0
        self.trades_this_round = 0
        self.last_mid = 0
        
        # Fair value tracking
        self.fair_values = {}
        
    def run(self, state: TradingState):
        """Main entry point for the trading algorithm"""
        # Initialize or restore state
        if state.traderData:
            try:
                trader_state = jsonpickle.decode(state.traderData)
                self.__dict__.update(trader_state)
            except Exception as e:
                print(f"Error restoring state: {e}")
        
        # Update positions from current state
        for product, pos in state.position.items():
            self.positions[product] = pos
        
        # Process observations if available
        if hasattr(state, 'observations') and state.observations:
            self.store_observations(state.observations)
        
        # Initialize result container
        result = {}
        
        # Process MAGNIFICENT_MACARONS if available
        if "MAGNIFICENT_MACARONS" in state.order_depths:
            # Reset trade counter for this round
            self.trades_this_round = 0
            
            # Generate orders
            orders = self.trade_macarons(
                "MAGNIFICENT_MACARONS",
                state.order_depths["MAGNIFICENT_MACARONS"],
                self.positions.get("MAGNIFICENT_MACARONS", 0)
            )
            
            result["MAGNIFICENT_MACARONS"] = orders
        
        # Determine if conversion is needed
        conversion = self.determine_conversion(
            "MAGNIFICENT_MACARONS", 
            self.positions.get("MAGNIFICENT_MACARONS", 0)
        )
        
        # Save state
        trader_data = jsonpickle.encode(self.__dict__)
        
        return result, conversion, trader_data
        
    def store_observations(self, observations):
        """Store observation values for future use"""
        if hasattr(observations, 'conversionObservations'):
            for product, obs in observations.conversionObservations.items():
                if product == "MAGNIFICENT_MACARONS":
                    # Store each observation type if available
                    if hasattr(obs, 'sugarPrice'):
                        self.observation_history['sugarPrice'].append(obs.sugarPrice)
                    if hasattr(obs, 'sunlightIndex'):
                        self.observation_history['sunlightIndex'].append(obs.sunlightIndex)
                    if hasattr(obs, 'exportTariff'):
                        self.observation_history['exportTariff'].append(obs.exportTariff)
                    if hasattr(obs, 'importTariff'):
                        self.observation_history['importTariff'].append(obs.importTariff)
                    if hasattr(obs, 'transportFees'):
                        self.observation_history['transportFees'].append(obs.transportFees)
                    
                    # Keep history at reasonable size
                    for key in self.observation_history:
                        if len(self.observation_history[key]) > 30:
                            self.observation_history[key] = self.observation_history[key][-30:]
    
    def trade_macarons(self, product, order_depth, position):
        """Main trading strategy for MAGNIFICENT_MACARONS"""
        # Initialize orders list
        orders = []
        
        # Get best prices from order book
        best_bid, best_bid_amount = self.get_best_bid(order_depth)
        best_ask, best_ask_amount = self.get_best_ask(order_depth)
        
        # If market is one-sided, return empty orders
        if not best_bid or not best_ask:
            return orders
        
        # Calculate mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Initialize price history if needed
        if product not in self.price_history:
            self.price_history[product] = []
            
        # Initialize fair value history if needed
        if product not in self.fair_values:
            self.fair_values[product] = []
        
        # Record price
        self.price_history[product].append(mid_price)
        
        # Keep only the most recent 20 prices
        if len(self.price_history[product]) > 20:
            self.price_history[product] = self.price_history[product][-20:]
        
        # Calculate fair value based on observation factors
        fair_value = self.calculate_fair_value(product, mid_price)
        
        # Store fair value history
        self.fair_values[product].append(fair_value)
        if len(self.fair_values[product]) > 20:
            self.fair_values[product] = self.fair_values[product][-20:]
        
        # Calculate fair value gap - the main trading signal
        fair_value_gap = fair_value - mid_price
        
        # Calculate price trend (negative value = downtrend)
        price_trend = self.calculate_trend(product)
        
        # Record position
        self.position_history.append(position)
        
        # Keep only most recent 10 positions
        if len(self.position_history) > 10:
            self.position_history = self.position_history[-10:]
        
        # Update market phase based on observations and trends
        self.update_market_phase(price_trend, position, fair_value_gap)
        
        # CORE TRADING STRATEGY
        if self.market_phase == "discovery":
            # In discovery phase, take small positions to test the market
            orders = self.execute_discovery_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, fair_value_gap)
        
        elif self.market_phase == "aggressive_short":
            # In aggressive shorting phase, build significant short position
            orders = self.execute_aggressive_short_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, fair_value_gap)
        
        elif self.market_phase == "defensive":
            # In defensive phase, reduce exposure and protect capital
            orders = self.execute_defensive_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, fair_value_gap)
        
        elif self.market_phase == "opportunistic":
            # In opportunistic phase, look for quick profits in both directions
            orders = self.execute_opportunistic_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, price_trend, fair_value_gap)
        
        # Safety mechanism - verify we're not exceeding position limits
        self.trades_this_round += len(orders)
        self.last_mid = mid_price
        
        return orders
    
    def calculate_fair_value(self, product, mid_price):
        """Calculate fair value based on observation factors"""
        # If we don't have observation history, use mid price
        if all(len(history) == 0 for history in self.observation_history.values()):
            return mid_price
        
        # Calculate weighted contribution from each factor
        weighted_sum = 0
        total_weight = 0
        
        for factor, weight in self.factor_weights.items():
            # Skip factors without history
            if not self.observation_history[factor]:
                continue
                
            # Get latest value
            value = self.observation_history[factor][-1]
            
            # Add weighted contribution
            weighted_sum += value * weight
            total_weight += abs(weight)
        
        # If we couldn't calculate from factors, return mid price
        if total_weight == 0:
            return mid_price
            
        # Calculate normalized fair value
        fair_value = weighted_sum / total_weight
        
        # Smooth with historical values if available
        if product in self.fair_values and self.fair_values[product]:
            recent_values = self.fair_values[product][-5:]
            historical_avg = sum(recent_values) / len(recent_values)
            # Blend new and historical (70/30)
            fair_value = (fair_value * 0.7) + (historical_avg * 0.3)
        
        return fair_value
    
    def execute_discovery_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, fair_value_gap):
        """Initial discovery phase - take small positions to test market direction based on fair value"""
        orders = []
        
        # If fair value suggests significant underpricing (positive gap), consider buying
        if fair_value_gap > 2 and position < 10:
            # Buy at ask with small size
            buy_amount = min(-best_ask_amount, 10)
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                print(f"DISCOVERY: Buying {buy_amount} @ {best_ask} (Fair value gap: {fair_value_gap:.2f})")
                return orders
                
        # If fair value suggests significant overpricing (negative gap), establish short
        elif fair_value_gap < -2 and position > -10:
            # Sell at bid to establish short position
            sell_amount = min(best_bid_amount, 10 + position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DISCOVERY: Selling {sell_amount} @ {best_bid} (Fair value gap: {fair_value_gap:.2f})")
                return orders
        
        # If no clear signal from fair value, reduce any existing position
        if position > 0:
            # Sell at best bid to reduce long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DISCOVERY: Selling {sell_amount} @ {best_bid} to reduce long position")
        elif position < 0:
            # Buy at best ask to reduce short position
            buy_amount = min(-best_ask_amount, -position)
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                print(f"DISCOVERY: Buying {buy_amount} @ {best_ask} to reduce short position")
        
        return orders
    
    def execute_aggressive_short_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, fair_value_gap):
        """Aggressively build short position based on fair value signals"""
        orders = []
        
        # If we have a long position, exit immediately regardless of fair value
        if position > 0:
            # Sell at best bid to exit long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"AGGRESSIVE SHORT: Selling {sell_amount} @ {best_bid} to exit long position")
                return orders  # Exit after handling long position
        
        # Calculate target short position based on fair value gap
        # More aggressive short for larger negative gaps
        gap_factor = min(1.0, abs(fair_value_gap) / 5.0)  # Scale between 0-1 based on gap size
        target_short = int(20 + (gap_factor * 30))  # Between 20-50 based on gap
        target_short = min(target_short, self.max_position)  # Cap at max position
        
        # If we have room to add more short position
        if position > -target_short:
            # Calculate how much to sell
            sell_amount = min(best_bid_amount, target_short + position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"AGGRESSIVE SHORT: Selling {sell_amount} @ {best_bid} (Current: {position}, Target: {-target_short}, Gap: {fair_value_gap:.2f})")
        
        return orders
    
    def execute_defensive_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, fair_value_gap):
        """Defensive strategy - reduce exposure and protect capital based on fair value"""
        orders = []
        
        # Calculate target position based on fair value gap
        # For defensive mode, we aim for a more moderate position
        gap_magnitude = abs(fair_value_gap)
        target_position = 0  # Default to neutral
        
        if fair_value_gap < -1:  # Market overvalued
            # Target short position scaled by gap size
            target_position = -int(min(40, gap_magnitude * 10))
        elif fair_value_gap > 1:  # Market undervalued
            # Target small long position scaled by gap size
            target_position = int(min(20, gap_magnitude * 5))
        
        # Determine action based on current vs target position
        position_gap = position - target_position
        
        if position_gap > 10:  # We're too long
            # Sell to move toward target
            sell_amount = min(best_bid_amount, position_gap)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DEFENSIVE: Selling {sell_amount} @ {best_bid} (Current: {position}, Target: {target_position})")
        
        elif position_gap < -10:  # We're too short
            # Buy to move toward target
            buy_amount = min(-best_ask_amount, -position_gap)
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                print(f"DEFENSIVE: Buying {buy_amount} @ {best_ask} (Current: {position}, Target: {target_position})")
        
        return orders
    
    def execute_opportunistic_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, price_trend, fair_value_gap):
        """Opportunistic strategy - seek quick profits while responding to fair value signals"""
        orders = []
        
        # Calculate min/max position bounds
        min_position = max(-self.max_position, -60)  # Don't exceed -60 short
        max_position = min(self.max_position, 20)    # Don't exceed 20 long
        
        # If fair value shows clear opportunity, act on it
        if abs(fair_value_gap) >= 2:
            if fair_value_gap > 0:  # Market underpriced - BUY
                # Only buy if not over our max long position
                if position < max_position:
                    # Size trade based on gap magnitude
                    gap_factor = min(1.0, fair_value_gap / 5.0)
                    buy_size = int(5 + (gap_factor * 10))  # 5-15 units
                    buy_size = min(buy_size, max_position - position)
                    
                    if buy_size > 0:
                        buy_amount = min(-best_ask_amount, buy_size)
                        if buy_amount > 0:
                            orders.append(Order(product, best_ask, buy_amount))
                            print(f"OPPORTUNISTIC: Buying {buy_amount} @ {best_ask} (Gap: {fair_value_gap:.2f})")
                
            else:  # Market overpriced - SELL
                # Only sell if not over our max short position
                if position > min_position:
                    # Size trade based on gap magnitude
                    gap_factor = min(1.0, abs(fair_value_gap) / 5.0)
                    sell_size = int(5 + (gap_factor * 15))  # 5-20 units
                    sell_size = min(sell_size, position - min_position)
                    
                    if sell_size > 0:
                        sell_amount = min(best_bid_amount, sell_size)
                        if sell_amount > 0:
                            orders.append(Order(product, best_bid, -sell_amount))
                            print(f"OPPORTUNISTIC: Selling {sell_amount} @ {best_bid} (Gap: {fair_value_gap:.2f})")
                            
            return orders  # If we acted on fair value gap, exit
        
        # If no clear fair value signal, try market making if we have a reasonable position
        if min_position + 15 <= position <= max_position - 15:
            # Calculate the spread
            spread = best_ask - best_bid
            
            # Only market make if spread is decent
            if spread >= 2:
                # Place orders on both sides with a slight edge
                sell_price = best_ask - 1  # Undercut the ask
                buy_price = best_bid + 1   # Improve the bid
                
                # Calculate sizes based on current position bias
                position_bias = position / self.max_position  # -1 to 1
                
                # Favor selling when long, buying when short
                buy_amount = min(5, max_position - position)
                sell_amount = min(5, position - min_position)
                
                # Execute orders if sizes are valid
                if sell_amount > 0:
                    orders.append(Order(product, sell_price, -sell_amount))
                    print(f"OPPORTUNISTIC MM: Selling {sell_amount} @ {sell_price}")
                
                if buy_amount > 0:
                    orders.append(Order(product, buy_price, buy_amount))
                    print(f"OPPORTUNISTIC MM: Buying {buy_amount} @ {buy_price}")
        
        return orders
    
    def update_market_phase(self, price_trend, position, fair_value_gap):
        """Update market phase based on price trends, position, and fair value signals"""
        # Increment phase counter
        self.phase_counter += 1
        
        # Consider fair value gap in phase decisions
        # Strong negative gap indicates overvaluation (good for shorting)
        # Strong positive gap indicates undervaluation (good for buying)
        
        # If we're just starting or have only been in discovery for a short time
        if self.market_phase == "discovery" and self.phase_counter < 5:
            return  # Stay in discovery phase
            
        # Transition from discovery phase
        if self.market_phase == "discovery" and self.phase_counter >= 5:
            if fair_value_gap < -2 or price_trend < -0.001:  # Clear overvaluation or downtrend
                self.market_phase = "aggressive_short"
                self.phase_counter = 0
                print(f"PHASE CHANGE: Discovery -> Aggressive Short (Gap: {fair_value_gap:.2f}, Trend: {price_trend:.4f})")
            else:
                self.market_phase = "opportunistic"
                self.phase_counter = 0
                print(f"PHASE CHANGE: Discovery -> Opportunistic (Gap: {fair_value_gap:.2f})")
            return
            
        # Transitions from aggressive short phase
        if self.market_phase == "aggressive_short":
            if position <= -50:  # If we've built up a significant short position
                self.market_phase = "defensive"
                self.phase_counter = 0
                print(f"PHASE CHANGE: Aggressive Short -> Defensive (position: {position})")
            elif fair_value_gap > 1 or price_trend > 0.002:  # If fair value or trend reverses
                self.market_phase = "defensive"
                self.phase_counter = 0
                print(f"PHASE CHANGE: Aggressive Short -> Defensive (Gap: {fair_value_gap:.2f}, Trend: {price_trend:.4f})")
            return
            
        # Transitions from defensive phase
        if self.market_phase == "defensive" and self.phase_counter >= 3:
            self.market_phase = "opportunistic"
            self.phase_counter = 0
            print(f"PHASE CHANGE: Defensive -> Opportunistic (defense period complete)")
            return
            
        # Transitions from opportunistic phase
        if self.market_phase == "opportunistic":
            if fair_value_gap < -3 or price_trend < -0.002:  # Strong overvaluation or downtrend
                self.market_phase = "aggressive_short"
                self.phase_counter = 0
                print(f"PHASE CHANGE: Opportunistic -> Aggressive Short (Gap: {fair_value_gap:.2f}, Trend: {price_trend:.4f})")
            elif abs(position) > 50:  # Large position built up
                self.market_phase = "defensive"
                self.phase_counter = 0
                print(f"PHASE CHANGE: Opportunistic -> Defensive (position: {position})")
            return
    
    def calculate_trend(self, product):
        """Calculate price trend (negative = downtrend)"""
        prices = self.price_history.get(product, [])
        
        if len(prices) < 5:
            return 0  # Not enough data
            
        # Calculate short-term trend (last 5 prices)
        recent_prices = prices[-5:]
        if recent_prices[0] == 0:  # Avoid division by zero
            return 0
            
        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    
    def determine_conversion(self, product, position):
        """Determine if conversion is needed based on position and observation factors"""
        if product != "MAGNIFICENT_MACARONS":
            return 0
            
        # If position is approaching limits, convert to free up capacity
        if position <= -70:
            return 10  # Convert short to buy and free up capacity
        elif position >= 70:
            return 10  # Convert long to free up capacity
        
        # Check for dramatic changes in observation factors
        factor_change_detected = False
        for factor in self.observation_history:
            if len(self.observation_history[factor]) >= 2:
                latest = self.observation_history[factor][-1]
                previous = self.observation_history[factor][-2]
                
                # Calculate percentage change
                if previous != 0:
                    change_pct = abs((latest - previous) / previous)
                    
                    # If significant change detected
                    if change_pct > 0.05:  # 5% change threshold
                        factor_change_detected = True
                        break
        
        # If we have a significant position and detected factor changes
        if factor_change_detected and abs(position) >= 30:
            return 10  # Convert to adjust to new market conditions
            
        # If in aggressive_short phase with significant position
        if self.market_phase == "aggressive_short" and position <= -50:
            return 10  # Convert some shorts to establish new ones
            
        # If we have a position opposite to our strategy
        if self.market_phase in ["aggressive_short", "opportunistic"] and position >= 10:
            return 10  # Convert long position (against our strategy)
            
        return 0
    
    def get_best_bid(self, order_depth):
        """Get highest bid price and quantity"""
        if not order_depth.buy_orders:
            return None, None
        best_bid = max(order_depth.buy_orders.keys())
        return best_bid, order_depth.buy_orders[best_bid]
    
    def get_best_ask(self, order_depth):
        """Get lowest ask price and quantity"""
        if not order_depth.sell_orders:
            return None, None
        best_ask = min(order_depth.sell_orders.keys())
        return best_ask, order_depth.sell_orders[best_ask]
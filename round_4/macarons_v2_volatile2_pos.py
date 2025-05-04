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
        
        # Position limits and tracking
        self.max_position = 70  # Slightly below the 75 limit for safety
        self.preferred_position = -40  # Maintain a net short bias in bear market
        
        # Market tracking
        self.market_phase = "discovery"  # Start in discovery phase
        self.phase_counter = 0
        self.trades_this_round = 0
        self.last_mid = 0
        
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
        
        # Record price
        self.price_history[product].append(mid_price)
        
        # Keep only the most recent 20 prices
        if len(self.price_history[product]) > 20:
            self.price_history[product] = self.price_history[product][-20:]
        
        # Calculate price trend (negative value = downtrend)
        price_trend = self.calculate_trend(product)
        
        # Record position
        self.position_history.append(position)
        
        # Keep only most recent 10 positions
        if len(self.position_history) > 10:
            self.position_history = self.position_history[-10:]
        
        # Update market phase
        self.update_market_phase(price_trend, position)
        
        # CORE TRADING STRATEGY
        if self.market_phase == "discovery":
            # In discovery phase, take small positions to test the market
            orders = self.execute_discovery_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount)
        
        elif self.market_phase == "aggressive_short":
            # In aggressive shorting phase, build significant short position
            orders = self.execute_aggressive_short_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount)
        
        elif self.market_phase == "defensive":
            # In defensive phase, reduce exposure and protect capital
            orders = self.execute_defensive_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount)
        
        elif self.market_phase == "opportunistic":
            # In opportunistic phase, look for quick profits in both directions
            orders = self.execute_opportunistic_strategy(product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, price_trend)
        
        # Safety mechanism - verify we're not exceeding position limits
        self.trades_this_round += len(orders)
        self.last_mid = mid_price
        
        return orders
    
    def execute_discovery_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount):
        """Initial discovery phase - take small positions to test market direction"""
        orders = []
        
        # If we have a long position, try to reduce it
        if position > 0:
            # Sell at best bid to reduce long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DISCOVERY: Selling {sell_amount} @ {best_bid} to reduce long position")
        
        # If position is neutral or only slightly short, build moderate short position
        if position > -10:
            # Sell at bid to establish short position
            sell_amount = min(best_bid_amount, 20 + position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DISCOVERY: Selling {sell_amount} @ {best_bid} to establish short position")
        
        return orders
    
    def execute_aggressive_short_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount):
        """Aggressively build short position when market is declining"""
        orders = []
        
        # If we have a long position, exit immediately
        if position > 0:
            # Sell at best bid to exit long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"AGGRESSIVE SHORT: Selling {sell_amount} @ {best_bid} to exit long position")
                return orders  # Exit after handling long position
        
        # If we have room to add more short position
        if position > -self.max_position:
            target_short = min(self.max_position, 50)  # Target 50 short or max position
            
            if position > -target_short:
                # Calculate how much to sell
                sell_amount = min(best_bid_amount, target_short + position)
                if sell_amount > 0:
                    orders.append(Order(product, best_bid, -sell_amount))
                    print(f"AGGRESSIVE SHORT: Selling {sell_amount} @ {best_bid} (Current: {position}, Target: {-target_short})")
        
        return orders
    
    def execute_defensive_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount):
        """Defensive strategy - reduce exposure and protect capital"""
        orders = []
        
        # If we're too heavily short, reduce position
        if position < -50:
            # Buy at best ask to reduce short position
            buy_amount = min(-best_ask_amount, -position - 40)  # Target -40 position
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                print(f"DEFENSIVE: Buying {buy_amount} @ {best_ask} to reduce short position")
        
        # If we're long, get neutral
        elif position > 0:
            # Sell at best bid to exit long position
            sell_amount = min(best_bid_amount, position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"DEFENSIVE: Selling {sell_amount} @ {best_bid} to exit long position")
        
        return orders
    
    def execute_opportunistic_strategy(self, product, position, best_bid, best_bid_amount, best_ask, best_ask_amount, price_trend):
        """Opportunistic strategy - seek quick profits while maintaining short bias"""
        orders = []
        
        # If deeply underinvested (not enough short position), add more shorts
        if position > -20:
            # Add more shorts
            sell_amount = min(best_bid_amount, 20 + position)
            if sell_amount > 0:
                orders.append(Order(product, best_bid, -sell_amount))
                print(f"OPPORTUNISTIC: Selling {sell_amount} @ {best_bid} to establish minimum short position")
            return orders
            
        # If we're too heavily short in a possible uptrend, reduce position somewhat
        if position < -60 and price_trend > 0:
            # Buy to reduce short position
            buy_amount = min(-best_ask_amount, -position - 40)  # Target -40 position
            if buy_amount > 0:
                orders.append(Order(product, best_ask, buy_amount))
                print(f"OPPORTUNISTIC: Buying {buy_amount} @ {best_ask} to reduce heavy short in uptrend")
        
        # If we're near our preferred position, consider market making
        if abs(position - self.preferred_position) < 10:
            # Calculate the spread
            spread = best_ask - best_bid
            
            # Only market make if spread is decent
            if spread >= 2:
                # Place orders on both sides with a slight edge
                if position > self.preferred_position - 15:  # Only sell if not too short
                    sell_price = best_ask - 1  # Undercut the ask
                    sell_amount = min(5, self.max_position + position)
                    if sell_amount > 0:
                        orders.append(Order(product, sell_price, -sell_amount))
                        print(f"OPPORTUNISTIC MM: Selling {sell_amount} @ {sell_price}")
                
                if position < self.preferred_position + 15:  # Only buy if not too long
                    buy_price = best_bid + 1  # Improve the bid
                    buy_amount = min(5, self.max_position - position)
                    if buy_amount > 0:
                        orders.append(Order(product, buy_price, buy_amount))
                        print(f"OPPORTUNISTIC MM: Buying {buy_amount} @ {buy_price}")
        
        return orders
    
    def update_market_phase(self, price_trend, position):
        """Update market phase based on current conditions"""
        # Increment phase counter
        self.phase_counter += 1
        
        # If we're just starting or have only been in discovery for a short time
        if self.market_phase == "discovery" and self.phase_counter < 5:
            return  # Stay in discovery phase
            
        # Transition from discovery phase
        if self.market_phase == "discovery" and self.phase_counter >= 5:
            if price_trend < -0.001:  # Downtrend detected
                self.market_phase = "aggressive_short"
                self.phase_counter = 0
                print("PHASE CHANGE: Discovery -> Aggressive Short (downtrend detected)")
            else:
                self.market_phase = "opportunistic"
                self.phase_counter = 0
                print("PHASE CHANGE: Discovery -> Opportunistic (no clear downtrend)")
            return
            
        # Transitions from aggressive short phase
        if self.market_phase == "aggressive_short":
            if position <= -50:  # If we've built up a significant short position
                self.market_phase = "defensive"
                self.phase_counter = 0
                print("PHASE CHANGE: Aggressive Short -> Defensive (large short position established)")
            elif price_trend > 0.002:  # If market trend reverses
                self.market_phase = "defensive"
                self.phase_counter = 0
                print("PHASE CHANGE: Aggressive Short -> Defensive (uptrend detected)")
            return
            
        # Transitions from defensive phase
        if self.market_phase == "defensive" and self.phase_counter >= 3:
            self.market_phase = "opportunistic"
            self.phase_counter = 0
            print("PHASE CHANGE: Defensive -> Opportunistic (defense period complete)")
            return
            
        # Transitions from opportunistic phase
        if self.market_phase == "opportunistic":
            if price_trend < -0.002:  # Strong downtrend detected
                self.market_phase = "aggressive_short"
                self.phase_counter = 0
                print("PHASE CHANGE: Opportunistic -> Aggressive Short (downtrend detected)")
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
        """Determine if conversion is needed"""
        if product != "MAGNIFICENT_MACARONS":
            return 0
            
        # If position is approaching limits, convert to free up capacity
        if position <= -70:
            return 10  # Convert short to buy and free up capacity
        elif position >= 70:
            return 10  # Convert long to free up capacity
            
        # If we have a moderate position but want to increase profits in the market phase
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
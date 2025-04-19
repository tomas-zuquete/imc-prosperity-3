from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import jsonpickle
import math

class Trader:
    def __init__(self):
        # Strategy configuration
        self.use_smart_arbitrage = True  # New hybrid strategy
        
        # State tracking
        self.spreads_history = {}
        self.current_positions = {}
        self.total_pnl = 0
        self.fair_values = {}
        self.price_history = {"PICNIC_BASKET1": [], "PICNIC_BASKET2": [], 
                             "CROISSANTS": [], "JAMS": [], "DJEMBES": []}
        self.active_trades = {}
        self.vwap_prices = {}  # Track volume-weighted average prices
        self.execution_prices = {}  # Track our execution prices
        
        # Strategy parameters
        self.min_arb_edge = 5  # Minimum edge for arbitrage in price points
        self.max_exposure = 3  # Maximum position in any basket
        self.mm_spread_factor = 0.5  # Fraction of spread to capture when market making
        
        # Basket compositions
        self.basket_components = {
            "PICNIC_BASKET1": {
                "CROISSANTS": 6,
                "JAMS": 3,
                "DJEMBES": 1
            },
            "PICNIC_BASKET2": {
                "CROISSANTS": 4,
                "JAMS": 2
            }
        }
        
        # Maximum position we want to hold for each product
        self.max_positions = {
            "PICNIC_BASKET1": 5,
            "PICNIC_BASKET2": 5,
            "CROISSANTS": 30,
            "JAMS": 15,
            "DJEMBES": 5
        }
        
        # Products we're trading
        self.products_to_trade =["CROISSANTS"] # ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]
        
        # Basket compositions
        self.basket_components = {
            "PICNIC_BASKET1": {
                "CROISSANTS": 6,
                "JAMS": 3,
                "DJEMBES": 1
            },
            "PICNIC_BASKET2": {
                "CROISSANTS": 4,
                "JAMS": 2
            }
        }
        
        # Maximum position we want to hold for each product
        self.max_positions = {
            "PICNIC_BASKET1": 10,
            "PICNIC_BASKET2": 10,
            "CROISSANTS": 60,
            "JAMS": 30,
            "DJEMBES": 10
        }

    def run(self, state: TradingState):
        """
        Main trading logic entry point
        """
        # Prepare our state from previous iterations if it exists
        if state.traderData != "":
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.spreads_history = saved_state.get('spreads_history', {})
                self.current_positions = saved_state.get('current_positions', {})
                self.total_pnl = saved_state.get('total_pnl', 0)
                self.fair_values = saved_state.get('fair_values', {})
                self.price_history = saved_state.get('price_history', {"PICNIC_BASKET1": [], "PICNIC_BASKET2": [], 
                                                                 "CROISSANTS": [], "JAMS": [], "DJEMBES": []})
                self.active_trades = saved_state.get('active_trades', {})
                self.vwap_prices = saved_state.get('vwap_prices', {})
                self.execution_prices = saved_state.get('execution_prices', {})
            except Exception:
                pass  # Silently continue if there's an error
        
        # Update positions based on state
        for product in self.products_to_trade:
            self.current_positions[product] = state.position.get(product, 0)
        
        # Process new trades
        iteration_pnl = 0
        for product, trades in state.own_trades.items():
            for trade in trades:
                # Calculate trade PnL
                trade_pnl = trade.price * trade.quantity * (-1 if trade.buyer == "SUBMISSION" else 1)
                iteration_pnl += trade_pnl
                
                # Update VWAP prices - track our average execution prices
                if product not in self.vwap_prices:
                    self.vwap_prices[product] = {'buy': [], 'sell': []}
                
                if trade.buyer == "SUBMISSION":  # We bought
                    self.vwap_prices[product]['buy'].append((trade.price, abs(trade.quantity)))
                    # Preserve most recent 50 trades
                    if len(self.vwap_prices[product]['buy']) > 50:
                        self.vwap_prices[product]['buy'] = self.vwap_prices[product]['buy'][-50:]
                else:  # We sold
                    self.vwap_prices[product]['sell'].append((trade.price, abs(trade.quantity)))
                    if len(self.vwap_prices[product]['sell']) > 50:
                        self.vwap_prices[product]['sell'] = self.vwap_prices[product]['sell'][-50:]
                
                # Remove trade from active trades
                trade_key = f"{product}_{trade.price}"
                if trade_key in self.active_trades:
                    del self.active_trades[trade_key]
        
        # Update total PnL
        self.total_pnl += iteration_pnl
        
        # Initialize results dictionary
        result = {}
        
        # Calculate market information for all products
        market_info = self._calculate_market_info(state.order_depths)
        
        # Generate trading orders using our strategy
        if self.use_smart_arbitrage:
            orders = self._generate_smart_arbitrage_orders(state, market_info)
        else:
            orders = {}  # Fallback to do nothing
        
        # Add orders to result
        for product, product_orders in orders.items():
            if product not in result:
                result[product] = []
            result[product].extend(product_orders)
        
        # Log only at specified intervals to keep logs manageable
        if state.timestamp % 5000 == 0:
            print(f"PnL at {state.timestamp}: {self.total_pnl}")
            
            # Check for significant arbitrage opportunities
            for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
                if basket in market_info and 'arb_profit' in market_info[basket]:
                    arb_profit = market_info[basket]['arb_profit']
                    if abs(arb_profit) > 20:  # Only log significant opportunities
                        direction = "Buy basket, sell components" if arb_profit > 0 else "Sell basket, buy components"
                        print(f"{basket} arbitrage opportunity: {direction}, profit: {abs(arb_profit):.2f}")
        
        # Prepare trader data for next iteration
        return self._finish_trading(result, state)
        
        # Define the list of products we want to trade
        # Only trade basket arbitrage products
        products_to_trade = ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]
        
    def _calculate_market_info(self, order_depths):
        """
        Calculate market information for all products
        """
        market_info = {}
        
        # Calculate midprices, best bids/asks for all products
        for product in self.products_to_trade:
            if product not in order_depths:
                continue
                
            order_depth = order_depths[product]
            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                continue
                
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            best_ask_volume = abs(order_depth.sell_orders[best_ask])
            midprice = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            market_info[product] = {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'best_bid_volume': best_bid_volume,
                'best_ask_volume': best_ask_volume,
                'midprice': midprice,
                'spread': spread
            }
            
            # Record price history
            if product in self.price_history:
                self.price_history[product].append({
                    'bid': best_bid,
                    'ask': best_ask,
                    'mid': midprice,
                    'spread': spread
                })
                # Keep history at a reasonable size
                if len(self.price_history[product]) > 100:
                    self.price_history[product] = self.price_history[product][-100:]
        
        # Calculate arbitrage opportunities for baskets
        for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if basket not in market_info:
                continue
                
            # Check if all components are available
            components_available = True
            for component in self.basket_components[basket]:
                if component not in market_info:
                    components_available = False
                    break
                    
            if not components_available:
                continue
                
            # Calculate synthetic basket prices from components
            synth_bid = 0
            synth_ask = 0
            for component, qty in self.basket_components[basket].items():
                synth_bid += market_info[component]['best_bid'] * qty
                synth_ask += market_info[component]['best_ask'] * qty
            
            # Calculate arbitrage opportunities
            # Strategy 1: Buy basket, sell components
            strat1_cost = market_info[basket]['best_ask']  # Cost to buy basket
            strat1_revenue = synth_bid  # Revenue from selling components
            strat1_profit = strat1_revenue - strat1_cost
            
            # Strategy 2: Sell basket, buy components
            strat2_revenue = market_info[basket]['best_bid']  # Revenue from selling basket
            strat2_cost = synth_ask  # Cost to buy components
            strat2_profit = strat2_revenue - strat2_cost
            
            # Store arbitrage information
            market_info[basket].update({
                'synth_bid': synth_bid,
                'synth_ask': synth_ask,
                'buy_basket_profit': strat1_profit,
                'sell_basket_profit': strat2_profit,
                'arb_profit': max(strat1_profit, strat2_profit, 0),  # Take the more profitable strategy or 0
                'arb_direction': 1 if strat1_profit > strat2_profit and strat1_profit > 0 else 
                               (-1 if strat2_profit > 0 else 0)  # 1 for buy basket, -1 for sell basket, 0 for none
            })
            
        return market_info
        
    def _generate_smart_arbitrage_orders(self, state, market_info):
        """
        Generate orders based on a smart combined strategy:
        1. Look for pure arbitrage opportunities
        2. Fall back to market making with skewed pricing based on inventory
        """
        orders = {}
        
        # First try to capitalize on arbitrage opportunities
        arb_orders = self._generate_arbitrage_orders(state, market_info)
        for product, product_orders in arb_orders.items():
            if product not in orders:
                orders[product] = []
            orders[product].extend(product_orders)
        
        # If we didn't generate any arbitrage orders, do intelligent market making
        if not any(len(product_orders) > 0 for product in arb_orders.values()):
            mm_orders = self._generate_market_making_orders(state, market_info)
            for product, product_orders in mm_orders.items():
                if product not in orders:
                    orders[product] = []
                orders[product].extend(product_orders)
        
        return orders
        
    def _generate_arbitrage_orders(self, state, market_info):
        """
        Generate orders to capitalize on arbitrage opportunities
        """
        orders = {product: [] for product in self.products_to_trade}
        
        # Check for basket arbitrage opportunities
        for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if (basket not in market_info or 
                'arb_profit' not in market_info[basket] or 
                market_info[basket]['arb_profit'] <= self.min_arb_edge):
                continue
                
            # Check if we're within position limits
            current_pos = self.current_positions.get(basket, 0)
            arb_direction = market_info[basket]['arb_direction']
            
            # Skip if we would exceed position limits
            if (arb_direction > 0 and current_pos >= self.max_exposure) or \
               (arb_direction < 0 and current_pos <= -self.max_exposure):
                continue
                
            # Calculate arb trade size (limit to 1 to be conservative)
            trade_size = 1
            
            if arb_direction > 0:  # Buy basket, sell components
                # Add order to buy the basket
                orders[basket].append(Order(
                    basket, 
                    market_info[basket]['best_ask'], 
                    trade_size
                ))
                
                # Add orders to sell all components
                for component, qty in self.basket_components[basket].items():
                    if component in market_info:
                        orders[component].append(Order(
                            component,
                            market_info[component]['best_bid'],
                            -qty * trade_size  # Negative for sell
                        ))
            
            elif arb_direction < 0:  # Sell basket, buy components
                # Add order to sell the basket
                orders[basket].append(Order(
                    basket,
                    market_info[basket]['best_bid'],
                    -trade_size  # Negative for sell
                ))
                
                # Add orders to buy all components
                for component, qty in self.basket_components[basket].items():
                    if component in market_info:
                        orders[component].append(Order(
                            component,
                            market_info[component]['best_ask'],
                            qty * trade_size
                        ))
        
        return orders
        
    def _generate_market_making_orders(self, state, market_info):
        """
        Generate market making orders with inventory skew
        """
        orders = {product: [] for product in self.products_to_trade}
        
        for product in self.products_to_trade:
            if product not in market_info:
                continue
                
            current_pos = self.current_positions.get(product, 0)
            max_pos = self.max_positions.get(product, 5)
            
            # Calculate position ratio (-1 to 1)
            position_ratio = current_pos / max_pos if max_pos > 0 else 0
            
            # Skip if we're at max position
            if abs(position_ratio) >= 0.8:  # 80% of max position
                continue
                
            # Get market data
            best_bid = market_info[product]['best_bid']
            best_ask = market_info[product]['best_ask']
            spread = market_info[product]['spread']
            
            # Adjust prices based on our inventory
            # When we have long position, be more eager to sell (tighter ask)
            # When we have short position, be more eager to buy (tighter bid)
            bid_adjust = -position_ratio * spread * 0.5  # Positive when short, negative when long
            ask_adjust = -position_ratio * spread * 0.5  # Negative when short, positive when long
            
            # Calculate our prices
            our_bid = best_bid + int(bid_adjust)
            our_ask = best_ask + int(ask_adjust)
            
            # Ensure our prices are valid
            our_bid = min(our_bid, best_ask - 1)  # Don't cross the spread
            our_ask = max(our_ask, best_bid + 1)  # Don't cross the spread
            
            # Calculate order sizes
            # Baskets get smaller sizes
            if product.startswith("PICNIC_BASKET"):
                bid_size = 1
                ask_size = 1
            else:
                # Components get proportionally larger sizes
                comp_multiplier = 3 if product == "CROISSANTS" else (2 if product == "JAMS" else 1)
                bid_size = comp_multiplier
                ask_size = comp_multiplier
            
            # Adjust sizes based on inventory
            if position_ratio > 0.3:  # Reduce buys when long
                bid_size = 0
            elif position_ratio < -0.3:  # Reduce sells when short
                ask_size = 0
            
            # Place orders if sizes are positive
            if bid_size > 0:
                orders[product].append(Order(product, our_bid, bid_size))
            
            if ask_size > 0:
                orders[product].append(Order(product, our_ask, -ask_size))  # Negative for sell
        
        return orders
    
    def _calculate_synthetic_price(self, basket: str, midprices: Dict[str, float]) -> float:
        """
        Calculate synthetic price of a basket based on component prices
        """
        if basket not in self.basket_components:
            return None
        
        synthetic_price = 0
        all_components_available = True
        
        for component, quantity in self.basket_components[basket].items():
            if component not in midprices:
                all_components_available = False
                break
            synthetic_price += midprices[component] * quantity
        
        if not all_components_available:
            return None
            
        # Add a small premium to account for convenience value of the basket
        # This addresses the persistent negative spread we're seeing
        if basket == "PICNIC_BASKET1":
            synthetic_price += 10  # Add premium for PICNIC_BASKET1
        elif basket == "PICNIC_BASKET2":
            synthetic_price += 5   # Add premium for PICNIC_BASKET2
            
        # Store the fair value for monitoring
        self.fair_values[basket] = synthetic_price
        
        return synthetic_price
    
    def _generate_orders_for_basket(self, basket: str, z_score: float, 
                                   positions: Dict[str, int], 
                                   midprices: Dict[str, float],
                                   order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        """
        Generate orders based on the z-score signal for a basket
        """
        result = {}
        
        # Default positions to 0 if not present
        current_positions = {product: positions.get(product, 0) for product in self.max_positions}
        
        # Log current positions and synthetic price vs market price
        basket_pos = current_positions.get(basket, 0)
        basket_mid = midprices.get(basket, 0)
        synthetic_price = self.fair_values.get(basket, 0)
        print(f"Current {basket} position: {basket_pos}, Market price: {basket_mid}, Synthetic price: {synthetic_price}")
        
        # REVERSAL OF STRATEGY: Based on the negative PnL and log analysis,
        # it appears the spreads are persistently negative and not mean-reverting as expected
        
        # Determine trade direction and size based on z-score and current position
        if z_score > self.z_entry_threshold:  # Basket is very expensive (based on z-score)
            # REVERSED: Buy basket, sell components (opposite of traditional approach)
            trade_direction = 1  
            # Limit trade size if we already have a large position
            if current_positions.get(basket, 0) > 5:
                return result  # Don't increase position if already substantial
            trade_size = self._determine_trade_size(basket, trade_direction, current_positions)
            print(f"SIGNAL: {basket} expensive (z={z_score:.2f}), BUY basket")
            
        elif z_score < -self.z_entry_threshold:  # Basket is very cheap (based on z-score)
            # REVERSED: Sell basket, buy components (opposite of traditional approach)
            trade_direction = -1
            # Limit trade size if we already have a large negative position
            if current_positions.get(basket, 0) < -5:
                return result  # Don't increase short position if already substantial
            trade_size = self._determine_trade_size(basket, trade_direction, current_positions)
            print(f"SIGNAL: {basket} cheap (z={z_score:.2f}), SELL basket")
            
        elif abs(z_score) < self.z_exit_threshold:  # Close to mean, opportunity to exit
            # Only exit if we have a meaningful position
            if abs(current_positions.get(basket, 0)) >= 2:
                trade_direction = -1 if current_positions.get(basket, 0) > 0 else 1
                trade_size = abs(current_positions.get(basket, 0))
                print(f"SIGNAL: {basket} reverting (z={z_score:.2f}), EXIT position")
            else:
                return result
        else:
            return result  # No trade signal
        
        # No valid trade size
        if trade_size <= 0:
            return result
        
        # Limit trade size to be more conservative
        trade_size = min(trade_size, 2)
            
        # Generate orders
        orders = self._create_arbitrage_orders(
            basket, trade_direction, trade_size, order_depths, midprices
        )
        
        return orders
    
    def _determine_trade_size(self, basket: str, direction: int, positions: Dict[str, int]) -> int:
        """
        Calculate the appropriate trade size based on position limits
        """
        # Get current positions
        basket_position = positions.get(basket, 0)
        
        # Calculate remaining capacity based on direction
        if direction > 0:  # Buy basket
            remaining_capacity = self.max_positions[basket] - basket_position
            
            # Check component capacities for selling
            for component, qty in self.basket_components[basket].items():
                component_position = positions.get(component, 0)
                component_capacity = component_position + self.max_positions[component]
                remaining_capacity = min(remaining_capacity, component_capacity // qty)
        else:  # Sell basket
            remaining_capacity = basket_position + self.max_positions[basket]
            
            # Check component capacities for buying
            for component, qty in self.basket_components[basket].items():
                component_position = positions.get(component, 0)
                component_capacity = self.max_positions[component] - component_position
                remaining_capacity = min(remaining_capacity, component_capacity // qty)
        
        return max(0, remaining_capacity)
    
    def _create_arbitrage_orders(self, basket: str, direction: int, size: int,
                                order_depths: Dict[str, OrderDepth],
                                midprices: Dict[str, float]) -> Dict[str, List[Order]]:
        """
        Create the actual orders for arbitrage execution
        """
        result = {}
        
        # Create basket order
        if basket not in result:
            result[basket] = []
        
        # IMPROVED EXECUTION: Use limit orders with better prices to ensure execution
        basket_order_depth = order_depths.get(basket)
        if basket_order_depth:
            if direction > 0 and len(basket_order_depth.sell_orders) > 0:  # Buy basket
                # Get best sell price and add a small premium to ensure execution
                best_price = min(basket_order_depth.sell_orders.keys())
                # Use a more aggressive price (slightly higher than best ask)
                execution_price = best_price + 1
                result[basket].append(Order(basket, execution_price, size))
                print(f"Placing BUY order for {basket}: {size} @ {execution_price}")
                
            elif direction < 0 and len(basket_order_depth.buy_orders) > 0:  # Sell basket
                # Get best buy price and add a small discount to ensure execution
                best_price = max(basket_order_depth.buy_orders.keys())
                # Use a more aggressive price (slightly lower than best bid)
                execution_price = best_price - 1
                result[basket].append(Order(basket, execution_price, -size))
                print(f"Placing SELL order for {basket}: {size} @ {execution_price}")
        
        # Only create component orders if basket order is created
        if basket in result and result[basket]:
            # Create component orders (opposite direction)
            for component, qty in self.basket_components[basket].items():
                if component not in result:
                    result[component] = []
                
                component_size = qty * size
                component_order_depth = order_depths.get(component)
                
                if component_order_depth:
                    if direction > 0:  # Sell components (because we're buying basket)
                        if len(component_order_depth.buy_orders) > 0:
                            best_price = max(component_order_depth.buy_orders.keys())
                            # Use a slightly better price to ensure execution
                            execution_price = best_price - 1
                            result[component].append(Order(component, execution_price, -component_size))
                            print(f"Placing SELL order for {component}: {component_size} @ {execution_price}")
                    else:  # Buy components (because we're selling basket)
                        if len(component_order_depth.sell_orders) > 0:
                            best_price = min(component_order_depth.sell_orders.keys())
                            # Use a slightly better price to ensure execution
                            execution_price = best_price + 1
                            result[component].append(Order(component, execution_price, component_size))
                            print(f"Placing BUY order for {component}: {component_size} @ {execution_price}")
        
        return result
    
    def _finish_trading(self, result: Dict[str, List[Order]], state: TradingState):
        """
        Prepare the final result and save state data
        """
        # Calculate effective execution prices for reporting
        for product in self.products_to_trade:
            if product in self.vwap_prices and self.vwap_prices[product]['buy'] and self.vwap_prices[product]['sell']:
                # Calculate VWAP for buys
                total_qty_buy = sum(qty for _, qty in self.vwap_prices[product]['buy'])
                total_cost_buy = sum(price * qty for price, qty in self.vwap_prices[product]['buy'])
                buy_vwap = total_cost_buy / total_qty_buy if total_qty_buy > 0 else 0
                
                # Calculate VWAP for sells
                total_qty_sell = sum(qty for _, qty in self.vwap_prices[product]['sell'])
                total_revenue_sell = sum(price * qty for price, qty in self.vwap_prices[product]['sell'])
                sell_vwap = total_revenue_sell / total_qty_sell if total_qty_sell > 0 else 0
                
                # Store execution prices
                self.execution_prices[product] = {
                    'buy_vwap': buy_vwap,
                    'sell_vwap': sell_vwap,
                    'spread': sell_vwap - buy_vwap if buy_vwap > 0 and sell_vwap > 0 else 0
                }
        
        # Log important information every 10000 iterations
        if state.timestamp % 10000 == 0:
            print(f"\n----- Status at {state.timestamp} -----")
            print(f"Total PnL: {self.total_pnl}")
            
            # Report positions
            nonzero_positions = {p: pos for p, pos in self.current_positions.items() if pos != 0}
            if nonzero_positions:
                print("Current positions:")
                for product, pos in nonzero_positions.items():
                    print(f"  {product}: {pos}")
        
        # Serialize trader data for next iteration
        trader_data = {
            'spreads_history': self.spreads_history,
            'current_positions': self.current_positions,
            'total_pnl': self.total_pnl,
            'fair_values': self.fair_values,
            'price_history': self.price_history,
            'active_trades': self.active_trades,
            'vwap_prices': self.vwap_prices,
            'execution_prices': self.execution_prices
        }
        
        traderData = jsonpickle.encode(trader_data)
        conversions = 0  # We're not using conversions
        
        return result, conversions, traderData
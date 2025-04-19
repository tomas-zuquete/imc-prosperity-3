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
        self.recent_trades = []  # Store recent trades to monitor volatility
        self.volatility_metrics = {}  # Store volatility metrics
        
        # Strategy parameters - adjusted for more stability
        self.min_arb_edge = 8  # Increased minimum edge for arbitrage in price points
        self.max_exposure = 2  # Reduced maximum position in any basket
        self.mm_spread_factor = 0.3  # Reduced fraction of spread to capture when market making
        self.volatility_window = 50  # Window size for volatility calculation
        self.high_volatility_threshold = 0.8  # Threshold to consider market volatile
        
        # PnL tracking for position adjustments
        self.position_pnl = {}  # Track PnL per position
        self.trade_history = []  # Track all trades for analysis
        
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
            "PICNIC_BASKET1": 3,  # Reduced from 5
            "PICNIC_BASKET2": 3,  # Reduced from 5
            "CROISSANTS": 20,  # Reduced from 30
            "JAMS": 10,  # Reduced from 15
            "DJEMBES": 4   # Reduced from 5
        }
        
        # Products we're trading
        self.products_to_trade = ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]

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
                self.recent_trades = saved_state.get('recent_trades', [])
                self.volatility_metrics = saved_state.get('volatility_metrics', {})
                self.position_pnl = saved_state.get('position_pnl', {})
                self.trade_history = saved_state.get('trade_history', [])
            except Exception:
                pass  # Silently continue if there's an error
        
        # Update positions based on state
        for product in self.products_to_trade:
            self.current_positions[product] = state.position.get(product, 0)
        
        # Process new trades and calculate PnL
        iteration_pnl = 0
        for product, trades in state.own_trades.items():
            for trade in trades:
                # Calculate trade PnL
                trade_pnl = trade.price * trade.quantity * (-1 if trade.buyer == "SUBMISSION" else 1)
                iteration_pnl += trade_pnl
                
                # Record trade for volatility analysis
                trade_info = {
                    'timestamp': state.timestamp,
                    'product': product,
                    'price': trade.price,
                    'quantity': trade.quantity,
                    'side': 'buy' if trade.buyer == "SUBMISSION" else 'sell',
                    'pnl': trade_pnl
                }
                self.recent_trades.append(trade_info)
                self.trade_history.append(trade_info)
                
                # Update per-position PnL
                if product not in self.position_pnl:
                    self.position_pnl[product] = 0
                self.position_pnl[product] += trade_pnl
                
                # Update VWAP prices
                if product not in self.vwap_prices:
                    self.vwap_prices[product] = {'buy': [], 'sell': []}
                
                if trade.buyer == "SUBMISSION":  # We bought
                    self.vwap_prices[product]['buy'].append((trade.price, abs(trade.quantity)))
                    if len(self.vwap_prices[product]['buy']) > 50:
                        self.vwap_prices[product]['buy'] = self.vwap_prices[product]['buy'][-50:]
                else:  # We sold
                    self.vwap_prices[product]['sell'].append((trade.price, abs(trade.quantity)))
                    if len(self.vwap_prices[product]['sell']) > 50:
                        self.vwap_prices[product]['sell'] = self.vwap_prices[product]['sell'][-50:]
                
                # Track trade execution
                trade_key = f"{product}_{trade.price}"
                if trade_key in self.active_trades:
                    del self.active_trades[trade_key]
        
        # Keep recent trades limited to volatility window
        if len(self.recent_trades) > self.volatility_window:
            self.recent_trades = self.recent_trades[-self.volatility_window:]
        
        # Update total PnL
        self.total_pnl += iteration_pnl
        
        # Initialize results dictionary
        result = {}
        
        # Calculate market information for all products
        market_info = self._calculate_market_info(state.order_depths)
        
        # Calculate market volatility
        self._update_volatility_metrics()
        
        # Generate trading orders using our strategy
        if self.use_smart_arbitrage:
            orders = self._generate_stable_arbitrage_orders(state, market_info)
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
            # Print volatility metrics if available
            if self.volatility_metrics:
                for product, metrics in self.volatility_metrics.items():
                    if 'price_volatility' in metrics:
                        print(f"{product} volatility: {metrics['price_volatility']:.4f}")
        
        # Prepare trader data for next iteration
        return self._finish_trading(result, state)
        
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
        
    def _update_volatility_metrics(self):
        """
        Calculate and update volatility metrics based on recent trades
        """
        # Group recent trades by product
        trades_by_product = {}
        for trade in self.recent_trades:
            product = trade['product']
            if product not in trades_by_product:
                trades_by_product[product] = []
            trades_by_product[product].append(trade)
        
        # Calculate volatility for each product
        for product, trades in trades_by_product.items():
            if len(trades) < 5:  # Need minimum number of trades
                continue
                
            # Extract prices
            prices = [trade['price'] for trade in trades]
            
            # Calculate price volatility (standard deviation normalized by mean)
            mean_price = np.mean(prices)
            if mean_price > 0:
                price_volatility = np.std(prices) / mean_price
            else:
                price_volatility = 0
                
            # Calculate PnL volatility
            pnl_values = [trade['pnl'] for trade in trades]
            pnl_volatility = np.std(pnl_values) if len(pnl_values) > 0 else 0
            
            # Store metrics
            self.volatility_metrics[product] = {
                'price_volatility': price_volatility,
                'pnl_volatility': pnl_volatility,
                'is_volatile': price_volatility > self.high_volatility_threshold
            }
    
    def _generate_hedging_orders(self, state, market_info):
        """
        Generate orders to hedge overextended positions
        """
        orders = {product: [] for product in self.products_to_trade}
        
        # Check for products with extreme positions
        for product in self.products_to_trade:
            if product not in market_info:
                continue
                
            position = self.current_positions.get(product, 0)
            max_pos = self.max_positions.get(product, 5)
            
            # Calculate how far we are from neutral (as a ratio)
            position_ratio = position / max_pos if max_pos > 0 else 0
            
            # Check if position is extreme (beyond 70% of max)
            if abs(position_ratio) > 0.7:
                # Determine direction to reduce position
                hedge_direction = -1 if position > 0 else 1
                
                # Calculate hedge size (reduce by 50% of current position)
                hedge_size = min(abs(position) // 2, 1)  # At least reduce by 1, but not more than half
                if hedge_size == 0:
                    hedge_size = 1  # Ensure we hedge at least 1 unit
                
                # Create hedging order
                if hedge_direction > 0:  # Buy
                    price = market_info[product]['best_ask']
                    orders[product].append(Order(product, price, hedge_size))
                else:  # Sell
                    price = market_info[product]['best_bid']
                    orders[product].append(Order(product, price, -hedge_size))
        
        # For basket components, check if we need hedging based on basket composition
        for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if basket not in self.current_positions:
                continue
                
            basket_position = self.current_positions[basket]
            
            # If basket position is significant, check if components need hedging
            if abs(basket_position) >= 2:
                for component, qty in self.basket_components[basket].items():
                    # Calculate ideal component position based on basket position
                    ideal_component_position = -basket_position * qty  # Opposite of basket position
                    
                    # Get actual component position
                    actual_position = self.current_positions.get(component, 0)
                    
                    # Calculate imbalance
                    imbalance = actual_position - ideal_component_position
                    
                    # If imbalance is significant, hedge it
                    if abs(imbalance) >= qty:
                        # Determine direction to hedge
                        hedge_direction = -1 if imbalance > 0 else 1
                        
                        # Calculate hedge size (up to the imbalance, but capped at qty)
                        hedge_size = min(abs(imbalance), qty)
                        
                        # Create hedging order if component in market info
                        if component in market_info:
                            if hedge_direction > 0:  # Buy
                                price = market_info[component]['best_ask']
                                orders[component].append(Order(component, price, hedge_size))
                            else:  # Sell
                                price = market_info[component]['best_bid']
                                orders[component].append(Order(component, price, -hedge_size))
        
        return orders
    
    def _generate_conservative_mm_orders(self, state, market_info):
        """
        Generate conservative market making orders with enhanced stability features
        """
        orders = {product: [] for product in self.products_to_trade}
        
        for product in self.products_to_trade:
            if product not in market_info:
                continue
                
            # Get current position and limits
            current_pos = self.current_positions.get(product, 0)
            max_pos = self.max_positions.get(product, 5)
            
            # Calculate position ratio (-1 to 1)
            position_ratio = current_pos / max_pos if max_pos > 0 else 0
            
            # Skip if position is already significant
            if abs(position_ratio) > 0.6:  # Reduced from 0.8 for more conservative approach
                continue
                
            # Check if the market is volatile for this product
            is_volatile = False
            if product in self.volatility_metrics:
                is_volatile = self.volatility_metrics[product].get('is_volatile', False)
            
            # Get market data
            best_bid = market_info[product]['best_bid']
            best_ask = market_info[product]['best_ask']
            spread = market_info[product]['spread']
            
            # In volatile markets, widen our spread significantly
            mm_factor = self.mm_spread_factor
            if is_volatile:
                mm_factor = mm_factor * 0.5  # Reduce spread capture in volatile markets
            
            # Calculate position skew factor - more aggressive mean reversion
            # When long, be more eager to sell; when short, be more eager to buy
            skew_factor = -position_ratio * 0.7  # Increased from 0.5 for stronger mean reversion
            
            # Calculate our prices with position bias
            bid_adjust = max(-1, min(1, int(skew_factor * spread)))  # Limit adjustments to +/- 1 tick
            ask_adjust = max(-1, min(1, int(skew_factor * spread)))
            
            our_bid = best_bid + bid_adjust
            our_ask = best_ask + ask_adjust
            
            # Ensure our prices don't cross the spread
            our_bid = min(our_bid, best_ask - 1)
            our_ask = max(our_ask, best_bid + 1)
            
            # Very conservative order sizes based on product and volatility
            if product.startswith("PICNIC_BASKET"):
                bid_size = 1
                ask_size = 1
            else:
                # Smaller sizes for components in volatile markets
                if is_volatile:
                    bid_size = 1 if product == "CROISSANTS" else 1
                    ask_size = 1 if product == "CROISSANTS" else 1
                else:
                    bid_size = 2 if product == "CROISSANTS" else (2 if product == "JAMS" else 1)
                    ask_size = 2 if product == "CROISSANTS" else (2 if product == "JAMS" else 1)
            
            # Further reduce sizes based on position
            if position_ratio > 0.3:  # Long bias
                bid_size = 0  # Don't buy more when long
            elif position_ratio < -0.3:  # Short bias
                ask_size = 0  # Don't sell more when short
            
            # Place orders if sizes are positive
            if bid_size > 0:
                orders[product].append(Order(product, our_bid, bid_size))
            
            if ask_size > 0:
                orders[product].append(Order(product, our_ask, -ask_size))  # Negative for sell
        
        return orders

    def _generate_stable_arbitrage_orders(self, state, market_info):
        """
        Generate orders with enhanced stability features to reduce PnL volatility
        """
        orders = {product: [] for product in self.products_to_trade}
        
        # First, try to close any over-extended positions
        hedge_orders = self._generate_hedging_orders(state, market_info)
        for product, product_orders in hedge_orders.items():
            if product not in orders:
                orders[product] = []
            orders[product].extend(product_orders)
            
        # If we already have orders from hedging, skip arbitrage
        if any(len(product_orders) > 0 for product in hedge_orders.values()):
            return orders
            
        # Check each basket for arbitrage opportunities
        for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if (basket not in market_info or 
                'arb_profit' not in market_info[basket]):
                continue
                
            # Get arbitrage details
            arb_profit = market_info[basket]['arb_profit']
            arb_direction = market_info[basket]['arb_direction']
            
            # Check if the market is currently volatile for this basket
            is_volatile = False
            if basket in self.volatility_metrics:
                is_volatile = self.volatility_metrics[basket].get('is_volatile', False)
            
            # Adjust min_edge based on volatility
            effective_min_edge = self.min_arb_edge
            if is_volatile:
                effective_min_edge = self.min_arb_edge * 2  # Double the minimum edge in volatile markets
                
            # Skip if profit is below threshold
            if arb_profit <= effective_min_edge:
                continue
                
            # Check position limits
            current_pos = self.current_positions.get(basket, 0)
            
            # Skip if we would exceed position limits
            if (arb_direction > 0 and current_pos >= self.max_exposure) or \
               (arb_direction < 0 and current_pos <= -self.max_exposure):
                continue
                
            # Calculate trade size - more conservative in volatile markets
            trade_size = 1
            if is_volatile:
                trade_size = 1  # Always use minimum size in volatile markets
                
            # Execute the arbitrage
            if arb_direction > 0:  # Buy basket, sell components
                # Add order to buy the basket
                basket_price = market_info[basket]['best_ask']
                orders[basket].append(Order(basket, basket_price, trade_size))
                
                # Add orders to sell all components
                for component, qty in self.basket_components[basket].items():
                    if component in market_info:
                        component_price = market_info[component]['best_bid']
                        orders[component].append(Order(component, component_price, -qty * trade_size))
                        
            elif arb_direction < 0:  # Sell basket, buy components
                # Add order to sell the basket
                basket_price = market_info[basket]['best_bid']
                orders[basket].append(Order(basket, basket_price, -trade_size))
                
                # Add orders to buy all components
                for component, qty in self.basket_components[basket].items():
                    if component in market_info:
                        component_price = market_info[component]['best_ask']
                        orders[component].append(Order(component, component_price, qty * trade_size))
        
        # If no arbitrage opportunities, do conservative market making
        if not any(len(product_orders) > 0 for product in orders.values()):
            mm_orders = self._generate_conservative_mm_orders(state, market_info)
            for product, product_orders in mm_orders.items():
                if product not in orders:
                    orders[product] = []
                orders[product].extend(product_orders)
                
        return orders

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
        
        # Calculate stability metrics for position adjustment
        total_position_value = 0
        position_count = 0
        
        for product in self.products_to_trade:
            position = self.current_positions.get(product, 0)
            if position != 0:
                # Calculate approximate position value
                if product in state.order_depths:
                    depth = state.order_depths[product]
                    if len(depth.buy_orders) > 0 and len(depth.sell_orders) > 0:
                        mid_price = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
                        position_value = abs(position * mid_price)
                        total_position_value += position_value
                        position_count += 1
        
        # Log important information every 10000 iterations
        if state.timestamp % 10000 == 0:
            print(f"\n----- Status at {state.timestamp} -----")
            print(f"Total PnL: {self.total_pnl}")
            
            # Report positions and position value
            nonzero_positions = {p: pos for p, pos in self.current_positions.items() if pos != 0}
            if nonzero_positions:
                print("Current positions:")
                for product, pos in nonzero_positions.items():
                    print(f"  {product}: {pos}")
                
                if position_count > 0:
                    print(f"Average position value: {total_position_value/position_count:.2f}")
        
        # Keep trade history at manageable size
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        # Serialize trader data for next iteration
        trader_data = {
            'spreads_history': self.spreads_history,
            'current_positions': self.current_positions,
            'total_pnl': self.total_pnl,
            'fair_values': self.fair_values,
            'price_history': self.price_history,
            'active_trades': self.active_trades,
            'vwap_prices': self.vwap_prices,
            'execution_prices': self.execution_prices,
            'recent_trades': self.recent_trades,
            'volatility_metrics': self.volatility_metrics,
            'position_pnl': self.position_pnl,
            'trade_history': self.trade_history
        }
        
        traderData = jsonpickle.encode(trader_data)
        conversions = 0  # We're not using conversions
        
        return result, conversions, traderData
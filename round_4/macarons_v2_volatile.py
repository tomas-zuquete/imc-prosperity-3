from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle

class Trader:
    def __init__(self):
        self.mid_price_history = {}
        self.positions = {}
        self.position_limits = {"MAGNIFICENT_MACARONS": 75}
        self.best_pnl = 0
        self.last_pnl = 0
        self.pnl_history = []
        self.fair_values = {}
        self.volatility = {}
        self.risk_mode = "normal"  # Can be "normal", "conservative", or "aggressive"
        self.max_position_pct = 0.8  # Max % of position limit to use
        
    def run(self, state: TradingState):
        # Initialize or restore state
        if state.traderData != "":
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.mid_price_history = saved_state.get('mid_price_history', {})
                self.positions = saved_state.get('positions', {})
                self.best_pnl = saved_state.get('best_pnl', 0)
                self.last_pnl = saved_state.get('last_pnl', 0)
                self.pnl_history = saved_state.get('pnl_history', [])
                self.fair_values = saved_state.get('fair_values', {})
                self.volatility = saved_state.get('volatility', {})
                self.risk_mode = saved_state.get('risk_mode', "normal")
                self.max_position_pct = saved_state.get('max_position_pct', 0.8)
            except Exception as e:
                print(f"Error restoring state: {e}")
                self.reset_state()
        
        # Update positions from the state
        for product, pos in state.position.items():
            self.positions[product] = pos
            
            # Initialize history for new products
            if product not in self.mid_price_history:
                self.mid_price_history[product] = []
            if product not in self.fair_values:
                self.fair_values[product] = []
            if product not in self.volatility:
                self.volatility[product] = 0.5  # Default volatility
        
        # Prepare results dict for orders
        result = {}
        
        # Process MAGNIFICENT_MACARONS
        if "MAGNIFICENT_MACARONS" in state.order_depths:
            orders = self.trade_macarons(
                "MAGNIFICENT_MACARONS",
                state.order_depths["MAGNIFICENT_MACARONS"],
                self.positions.get("MAGNIFICENT_MACARONS", 0)
            )
            result["MAGNIFICENT_MACARONS"] = orders
        
        # Determine conversion need
        conversion_count = self.determine_conversion("MAGNIFICENT_MACARONS", self.positions.get("MAGNIFICENT_MACARONS", 0))
        
        # Save state
        trader_data = {
            'mid_price_history': self.mid_price_history,
            'positions': self.positions,
            'best_pnl': self.best_pnl,
            'last_pnl': self.last_pnl,
            'pnl_history': self.pnl_history,
            'fair_values': self.fair_values,
            'volatility': self.volatility,
            'risk_mode': self.risk_mode,
            'max_position_pct': self.max_position_pct
        }
        traderData = jsonpickle.encode(trader_data)
        
        return result, conversion_count, traderData
    
    def reset_state(self):
        """Reset state to defaults if there's an error"""
        self.mid_price_history = {}
        self.positions = {}
        self.best_pnl = 0
        self.last_pnl = 0
        self.pnl_history = []
        self.fair_values = {}
        self.volatility = {}
        self.risk_mode = "normal"
        self.max_position_pct = 0.8
    
    def trade_macarons(self, product, order_depth, position):
        """PnL-preserving strategy with focus on limiting drawdowns"""
        orders = []
        
        # Calculate mid price and fair value
        best_bid, best_bid_amount = self.get_best_bid(order_depth)
        best_ask, best_ask_amount = self.get_best_ask(order_depth)
        
        if not best_bid or not best_ask:
            return orders
            
        mid_price = (best_bid + best_ask) / 2
        
        # Initialize dictionaries for the product if they don't exist
        if product not in self.mid_price_history:
            self.mid_price_history[product] = []
        if product not in self.fair_values:
            self.fair_values[product] = []
            
        # Store mid price history
        self.mid_price_history[product].append(mid_price)
        
        # Only keep last 20 prices for trend analysis
        if len(self.mid_price_history[product]) > 20:
            self.mid_price_history[product] = self.mid_price_history[product][-20:]
        
        # Calculate fair value and store it
        fair_value = self.calculate_fair_value(product, best_bid, best_ask, mid_price)
        self.fair_values[product].append(fair_value)
        if len(self.fair_values[product]) > 10:
            self.fair_values[product] = self.fair_values[product][-10:]
        
        # Calculate market volatility
        self.calculate_volatility(product)
        
        # Detect market direction: 1 (up), -1 (down), 0 (sideways)
        market_direction = self.detect_market_direction(product)
        
        # Determine risk mode based on PnL history
        self.determine_risk_mode()
        
        # Calculate maximum position based on risk mode
        max_position = int(self.position_limits[product] * self.max_position_pct)
        
        # Base threshold on volatility and risk mode
        base_threshold = self.volatility[product] * 1.5
        
        if self.risk_mode == "conservative":
            base_threshold *= 1.5  # Higher threshold = fewer trades
            max_position = int(max_position * 0.7)  # Reduce position size
        elif self.risk_mode == "aggressive":
            base_threshold *= 0.8  # Lower threshold = more trades
        
        # CORE TRADING LOGIC
        
        # 1. Position unwinding when near peak PnL
        if self.risk_mode == "conservative" and position != 0:
            # If we're in conservative mode, focus on unwinding positions to preserve PnL
            if position > 0:  # Long position
                # Try to sell at bid or better
                sell_amount = min(best_bid_amount, position)
                if sell_amount > 0:
                    orders.append(Order(product, best_bid, -sell_amount))
                    position -= sell_amount
                    print(f"PNL PRESERVATION: Sold {sell_amount} @ {best_bid}")
                    return orders  # Exit early to focus on PnL preservation
            else:  # Short position
                # Try to buy at ask or better
                buy_amount = min(-best_ask_amount, -position)
                if buy_amount > 0:
                    orders.append(Order(product, best_ask, buy_amount))
                    position += buy_amount
                    print(f"PNL PRESERVATION: Bought {buy_amount} @ {best_ask}")
                    return orders  # Exit early to focus on PnL preservation
        
        # 2. Directional trading based on market trend
        if market_direction != 0:
            # Trading with the trend
            if market_direction < 0:  # Downtrend
                # If we have room to add short positions
                if position > -max_position:
                    # Check if price is above our fair value
                    if best_bid > fair_value - base_threshold:
                        # Sell at the bid
                        sell_amount = min(best_bid_amount, max_position + position)
                        # Scale position based on conviction (stronger downtrend = larger position)
                        sell_amount = int(sell_amount * min(0.9, abs(market_direction)))
                        sell_amount = max(1, sell_amount)  # At least 1 unit
                        
                        if sell_amount > 0:
                            orders.append(Order(product, best_bid, -sell_amount))
                            position -= sell_amount
                            print(f"TREND SELL: {sell_amount} @ {best_bid} (Direction: {market_direction:.2f})")
                
                # Only consider buying in downtrend if significantly oversold or to reduce extreme short position
                if position < -max_position * 0.8 or best_ask < fair_value - base_threshold * 2:
                    # Buy in small increments to reduce extreme short position
                    buy_amount = min(-best_ask_amount, max(1, -position // 4))
                    
                    if buy_amount > 0:
                        orders.append(Order(product, best_ask, buy_amount))
                        position += buy_amount
                        print(f"POSITION MGMT BUY: {buy_amount} @ {best_ask}")
            
            elif market_direction > 0:  # Uptrend
                # In uptrends, be very cautious about long positions
                if position < max_position * 0.4:  # Only use 40% of max position for longs
                    # Only buy if significantly below fair value
                    if best_ask < fair_value - base_threshold * 1.5:
                        buy_amount = min(-best_ask_amount, max_position * 0.4 - position)
                        buy_amount = int(buy_amount * min(0.7, market_direction))  # Scale down - less confidence in uptrends
                        buy_amount = max(1, buy_amount)  # At least 1 unit
                        
                        if buy_amount > 0:
                            orders.append(Order(product, best_ask, buy_amount))
                            position += buy_amount
                            print(f"CAUTIOUS BUY: {buy_amount} @ {best_ask} (Direction: {market_direction:.2f})")
                
                # Sell any long position if price is favorable
                if position > 0 and best_bid > fair_value:
                    sell_amount = min(best_bid_amount, position)
                    
                    if sell_amount > 0:
                        orders.append(Order(product, best_bid, -sell_amount))
                        position -= sell_amount
                        print(f"PROFIT TAKING: {sell_amount} @ {best_bid}")
        
        # 3. Position management regardless of trend
        position_pct = abs(position) / self.position_limits[product]
        
        # If position is too large, reduce it
        if position_pct > self.max_position_pct:
            if position > 0:  # Long position
                # Try to sell at or above fair value
                if best_bid >= fair_value:
                    sell_amount = min(best_bid_amount, position - int(max_position * 0.7))
                    if sell_amount > 0:
                        orders.append(Order(product, best_bid, -sell_amount))
                        position -= sell_amount
                        print(f"POSITION REDUCTION: Sold {sell_amount} @ {best_bid}")
            else:  # Short position
                # Try to buy at or below fair value
                if best_ask <= fair_value:
                    buy_amount = min(-best_ask_amount, -position - int(max_position * 0.7))
                    if buy_amount > 0:
                        orders.append(Order(product, best_ask, buy_amount))
                        position += buy_amount
                        print(f"POSITION REDUCTION: Bought {buy_amount} @ {best_ask}")
        
        # 4. Market making (if appropriate)
        spread = best_ask - best_bid
        
        # Only do market making if spread is decent and we're not in conservative mode
        if spread > 1.5 and self.risk_mode != "conservative" and abs(position) < max_position * 0.6:
            # Adjust order sizes based on current position
            position_bias = position / max_position  # -1 to 1 scale
            
            # Place buy orders near the bid if we have room to buy
            if position < max_position * (0.5 - position_bias * 0.3):
                buy_price = best_bid + 1  # Outbid the current best bid
                buy_size = min(3, max_position - position)  # Small size for market making
                
                if buy_size > 0:
                    orders.append(Order(product, buy_price, buy_size))
                    print(f"MARKET MAKING BUY: {buy_size} @ {buy_price}")
            
            # Place sell orders near the ask if we have room to sell
            if position > -max_position * (0.5 + position_bias * 0.3):
                sell_price = best_ask - 1  # Undercut the current best ask
                sell_size = min(3, max_position + position)  # Small size for market making
                
                if sell_size > 0:
                    orders.append(Order(product, sell_price, -sell_size))
                    print(f"MARKET MAKING SELL: {sell_size} @ {sell_price}")
        
        # Update position
        self.positions[product] = position
        
        return orders
    
    def determine_risk_mode(self):
        """Determine risk mode based on PnL history"""
        if len(self.pnl_history) < 5:
            return  # Not enough history
        
        # Calculate percentage of best PnL
        if self.best_pnl > 0:
            current_pnl_pct = self.pnl_history[-1] / self.best_pnl
            
            if current_pnl_pct > 0.95:
                # We're close to best PnL - get conservative
                self.risk_mode = "conservative"
                self.max_position_pct = 0.6  # Reduce position size
                print(f"RISK MODE: Conservative (PnL: {current_pnl_pct:.2%} of peak)")
            elif current_pnl_pct < 0.7:
                # We've dropped significantly - get aggressive to recover
                self.risk_mode = "aggressive"
                self.max_position_pct = 0.9  # Increase position size
                print(f"RISK MODE: Aggressive (PnL: {current_pnl_pct:.2%} of peak)")
            else:
                self.risk_mode = "normal"
                self.max_position_pct = 0.8
                print(f"RISK MODE: Normal (PnL: {current_pnl_pct:.2%} of peak)")
    
    def calculate_fair_value(self, product, best_bid, best_ask, mid_price):
        """Calculate fair value based on historical prices and market dynamics"""
        # Simple fair value based on mid price
        fair_value = mid_price
        
        # Adjust based on price history if available
        if len(self.mid_price_history[product]) >= 5:
            # Use exponential moving average for fair value
            weights = [0.05, 0.1, 0.15, 0.25, 0.45]  # More weight to recent prices
            prices = self.mid_price_history[product][-5:]
            weighted_price = sum(p * w for p, w in zip(prices, weights))
            
            # Blend current mid price with weighted historical price
            fair_value = 0.7 * mid_price + 0.3 * weighted_price
        
        return fair_value
    
    def calculate_volatility(self, product):
        """Calculate market volatility based on recent price movements"""
        if len(self.mid_price_history[product]) < 5:
            self.volatility[product] = 0.5  # Default
            return
            
        # Calculate average absolute price change
        prices = self.mid_price_history[product]
        changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_change = sum(changes) / len(changes)
        
        # Normalize volatility to a scale of 0.2 to 1.0
        self.volatility[product] = min(1.0, max(0.2, avg_change / 3.0))
    
    def detect_market_direction(self, product):
        """Detect market direction based on price history
        Returns a value from -1 (strong downtrend) to 1 (strong uptrend)
        """
        prices = self.mid_price_history[product]
        
        if len(prices) < 5:
            return 0  # Not enough data
            
        # Short-term direction (last 3 prices)
        short_term = prices[-3:]
        short_trend = (short_term[-1] - short_term[0]) / short_term[0] if short_term[0] != 0 else 0
        
        # Medium-term direction (last 7 prices, if available)
        medium_term = prices[-7:] if len(prices) >= 7 else prices
        medium_trend = (medium_term[-1] - medium_term[0]) / medium_term[0] if medium_term[0] != 0 else 0
        
        # Longer-term direction (last 15 prices, if available)
        long_term = prices[-15:] if len(prices) >= 15 else prices
        long_trend = (long_term[-1] - long_term[0]) / long_term[0] if long_term[0] != 0 else 0
        
        # Weighted direction, emphasizing recent movements
        direction = 0.5 * short_trend + 0.3 * medium_trend + 0.2 * long_trend
        
        # Calibrate signal strength
        # This will give a value from -1 to 1, with 0.5/-0.5 representing a moderate trend
        if direction > 0:
            direction = min(1.0, direction * 20)  # Scale up positive direction
        else:
            direction = max(-1.0, direction * 20)  # Scale up negative direction
            
        return direction
    
    def determine_conversion(self, product, position):
        """Determine if conversion is needed"""
        # Conversion limit for MAGNIFICENT_MACARONS is 10
        if product != "MAGNIFICENT_MACARONS" or abs(position) < 10:
            return 0
        
        position_limit = self.position_limits.get(product, 100)
        max_position = int(position_limit * self.max_position_pct)
        
        # If position exceeds our desired maximum, convert to reduce
        if abs(position) > max_position:
            # If we're long and exceeding max position, convert to reduce
            if position > max_position:
                return 10
            # If we're short and exceeding max position, convert to reduce
            elif position < -max_position:
                return 10
        
        # If in conservative mode, consider conversions to lock in profits
        if self.risk_mode == "conservative" and abs(position) >= 10:
            return 10
            
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
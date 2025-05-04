from round_5.sub.datamodel import OrderDepth, TradingState, Order
from typing import List
import numpy as np
import jsonpickle

class Trader:
    def __init__(self):
        # Position limits
        self.position_limits = {
            "KELP": 20,
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            "BERRIES": 250,
            "DIVING_GEAR": 50,
            "DOLPHIN_SIGHTINGS": 10,
            "STARFRUIT": 20,
            "SHELL_NECKLACE": 70,
            "SQUID_INK": 300,
            "RAINFOREST_RESIN": 60,
        }
        
        # Simple parameters
        self.ma_window = 5       # Very small window to start
        self.prices = {}         # Store price history
        self.positions = {}      # Track positions
        self.trades = {}         # Track all trades
        self.pnl = {}            # Track PnL
        
    def run(self, state: TradingState):
        """
        Detailed mean reversion strategy with thorough logging
        """
        # Initialize result
        result = {}
        
        # Load state from previous iteration
        if state.traderData:
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.prices = saved_state.get("prices", {})
                self.positions = saved_state.get("positions", {})
                self.trades = saved_state.get("trades", {})
                self.pnl = saved_state.get("pnl", {})
            except Exception as e:
                print(f"Error loading state: {e}")
        
        # Process KELP only
        product = "KELP"
        
        # Skip if product not available
        if product not in state.order_depths:
            traderData = jsonpickle.encode({
                "prices": self.prices,
                "positions": self.positions,
                "trades": self.trades,
                "pnl": self.pnl
            })
            return result, 0, traderData
        
        # Get order depth
        order_depth = state.order_depths[product]
        
        # Initialize orders list
        orders = []
        
        # Initialize tracking for this product if needed
        if product not in self.prices:
            self.prices[product] = []
        
        if product not in self.positions:
            self.positions[product] = []
        
        if product not in self.trades:
            self.trades[product] = []
        
        if product not in self.pnl:
            self.pnl[product] = []
        
        # Extract market data
        has_bids = len(order_depth.buy_orders) > 0
        has_asks = len(order_depth.sell_orders) > 0
        
        if has_bids and has_asks:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            # Calculate mid price and spread
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Update price history
            self.prices[product].append(mid_price)
            
            # Keep history at manageable size
            if len(self.prices[product]) > 50:
                self.prices[product] = self.prices[product][-50:]
            
            # Get current position
            current_position = state.position.get(product, 0)
            self.positions[product].append(current_position)
            
            # Get current PnL if available in the state
            if product in state.observations.plainValueObservations:
                current_pnl = state.observations.plainValueObservations[product]
                self.pnl[product].append(current_pnl)
            else:
                # If PnL not available, append the last known value or 0
                current_pnl = self.pnl[product][-1] if self.pnl[product] else 0
                self.pnl[product].append(current_pnl)
            
            # Wait until we have enough data
            if len(self.prices[product]) >= self.ma_window:
                # Calculate moving average
                ma = np.mean(self.prices[product][-self.ma_window:])
                
                # Calculate detailed metrics for logging
                deviation = mid_price - ma
                deviation_pct = (deviation / ma) * 100
                
                # Detailed logging of current state
                print(f"===== KELP DETAILED STATE =====")
                print(f"Timestamp: {state.timestamp}")
                print(f"Mid Price: {mid_price}")
                print(f"MA({self.ma_window}): {ma:.2f}")
                print(f"Deviation: {deviation:.2f} ({deviation_pct:.2f}%)")
                print(f"Spread: {spread} ({(spread/mid_price)*100:.2f}%)")
                print(f"Current Position: {current_position}")
                print(f"Current PnL: {current_pnl}")
                
                # Very simple mean reversion logic (same as before)
                if mid_price < ma and current_position < self.position_limits[product]:
                    # Price below average - BUY signal
                    order_size = 1
                    
                    if current_position + order_size <= self.position_limits[product]:
                        # Buy at ask
                        orders.append(Order(product, best_ask, order_size))
                        
                        # Record the trade
                        trade = {
                            "timestamp": state.timestamp,
                            "type": "BUY",
                            "price": best_ask,
                            "quantity": order_size,
                            "position_before": current_position,
                            "position_after": current_position + order_size,
                            "ma": ma,
                            "deviation": deviation
                        }
                        self.trades[product].append(trade)
                        
                        print(f"BUY {order_size}x {product} @ {best_ask}")
                        print(f"Trade Cost: {best_ask * order_size}")
                
                elif mid_price > ma and current_position > -self.position_limits[product]:
                    # Price above average - SELL signal
                    order_size = 1
                    
                    if current_position - order_size >= -self.position_limits[product]:
                        # Sell at bid
                        orders.append(Order(product, best_bid, -order_size))
                        
                        # Record the trade
                        trade = {
                            "timestamp": state.timestamp,
                            "type": "SELL",
                            "price": best_bid,
                            "quantity": order_size,
                            "position_before": current_position,
                            "position_after": current_position - order_size,
                            "ma": ma,
                            "deviation": deviation
                        }
                        self.trades[product].append(trade)
                        
                        print(f"SELL {order_size}x {product} @ {best_bid}")
                        print(f"Trade Revenue: {best_bid * order_size}")
            
            # Now that we have additional data, analyze trade performance
            if len(self.trades[product]) > 0:
                total_buy_cost = sum(t["price"] * t["quantity"] for t in self.trades[product] if t["type"] == "BUY")
                total_sell_revenue = sum(t["price"] * t["quantity"] for t in self.trades[product] if t["type"] == "SELL")
                total_trades = len(self.trades[product])
                
                print(f"===== TRADE SUMMARY =====")
                print(f"Total Trades: {total_trades}")
                print(f"Total Buy Cost: {total_buy_cost}")
                print(f"Total Sell Revenue: {total_sell_revenue}")
                print(f"Gross P&L: {total_sell_revenue - total_buy_cost}")
        
        # Add orders to result
        result[product] = orders
        
        # Save state for next iteration
        traderData = jsonpickle.encode({
            "prices": self.prices,
            "positions": self.positions,
            "trades": self.trades,
            "pnl": self.pnl
        })
        
        return result, 0, traderData
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle

class Trader:
    def run(self, state: TradingState):
        # Initialize trader data from previous state or create new
        if state.traderData == "":
            trader_data = {
                "counterparty_analysis": {},
                "last_timestamp": 0
            }
        else:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {
                    "counterparty_analysis": {},
                    "last_timestamp": 0
                }
        
        counterparty_analysis = trader_data["counterparty_analysis"]
        
        # Update counterparty analysis with new trades
        print(f"=== TIMESTAMP: {state.timestamp} ===")
        print(f"Own trades products: {list(state.own_trades.keys())}")
        
        # Count total trades
        total_trades = sum(len(trades) for trades in state.own_trades.values())
        print(f"Total trades: {total_trades}")
        
        # Analyze new trade data
        if total_trades > 0:
            print("\n=== COUNTERPARTY INFORMATION ===")
            
            for product in state.own_trades:
                if len(state.own_trades[product]) > 0:
                    print(f"Product: {product}")
                    
                    for trade in state.own_trades[product]:
                        trade_type = "BUY" if trade.quantity > 0 else "SELL"
                        
                        # Get counterparty
                        counterparty = trade.counter_party if hasattr(trade, 'counter_party') else "Unknown"
                        
                        # Record trade in counterparty analysis
                        if counterparty not in counterparty_analysis:
                            counterparty_analysis[counterparty] = {
                                "trades": 0,
                                "products": {},
                                "buy_count": 0,
                                "sell_count": 0,
                                "total_volume": 0
                            }
                        
                        # Update counterparty statistics
                        counterparty_analysis[counterparty]["trades"] += 1
                        if trade_type == "BUY":
                            counterparty_analysis[counterparty]["buy_count"] += 1
                        else:
                            counterparty_analysis[counterparty]["sell_count"] += 1
                        
                        # Update product-specific statistics
                        if trade.symbol not in counterparty_analysis[counterparty]["products"]:
                            counterparty_analysis[counterparty]["products"][trade.symbol] = {
                                "trades": 0,
                                "volume": 0
                            }
                        
                        counterparty_analysis[counterparty]["products"][trade.symbol]["trades"] += 1
                        counterparty_analysis[counterparty]["products"][trade.symbol]["volume"] += abs(trade.quantity)
                        counterparty_analysis[counterparty]["total_volume"] += abs(trade.quantity)
                        
                        print(f"  {trade_type} - Price: {trade.price}, Quantity: {abs(trade.quantity)}, Counterparty: {counterparty}")
                    
                    print("-" * 40)
        
        # Simple market making strategy that's guaranteed to generate orders
        result = {}
        for product in state.order_depths:
            orders = []
            order_depth = state.order_depths[product]
            
            # Buy orders - take the best asks
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                # Limit quantity to avoid position issues
                buy_quantity = min(abs(best_ask_volume), 2)  # Limit to 2 units
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
            
            # Sell orders - take the best bids
            if len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                # Limit quantity to avoid position issues
                sell_quantity = min(best_bid_volume, 2)  # Limit to 2 units
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
            
            result[product] = orders
        
        # Log summary of counterparty analysis
        if len(counterparty_analysis) > 0:
            print("\n=== COUNTERPARTY ANALYSIS SUMMARY ===")
            for cp, data in counterparty_analysis.items():
                print(f"Counterparty: {cp}")
                print(f"  Total trades: {data['trades']}")
                print(f"  Buy/Sell ratio: {data['buy_count']}/{data['sell_count']}")
                print(f"  Total volume: {data['total_volume']}")
                print(f"  Products traded: {list(data['products'].keys())}")
                print("-" * 40)
        
        # Update trader data for next iteration
        trader_data["counterparty_analysis"] = counterparty_analysis
        trader_data["last_timestamp"] = state.timestamp
        
        return result, 0, jsonpickle.encode(trader_data)
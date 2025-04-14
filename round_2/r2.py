import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import OrderDepth, UserId, TradingState, Order
import pandas as pd


# storing string as const to avoid typos, 1.7k pnl
SUBMISSION = "SUBMISSION"

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
PRODUCTS = [
    RESIN,
    KELP,
    SQUID_INK
]

DEFAULT_PRICES = {
    RESIN :10000,
    KELP:   2023,
    SQUID_INK: 1972
}

class Trader:

    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            RESIN: 50,
            KELP: 50, 
            SQUID_INK: 50
        }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position

        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None
        self.starfruit_prices = []  # List to store historical STARFRUIT prices for calculation
        self.amethysts_prices = []
        self.orchid_prices = []   # List to store historical ORCHID prices for calculation
        self.starfruit_log_return = []
        self.amethysts_log_return = []
        self.ink_prev_mid_price = None

        self.ema_param = 0.5
        # self.df_log_return = pd.DataFrame([], columns=pd.Index(['Log_Return']))


    # utils
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    


    def get_order_ratio(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        if len(market_asks) > 0 and len(market_bids) > 0:
            return (sum(market_bids) - sum(market_asks)) / (sum(market_bids) + sum(market_asks)) 


    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2

    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)

    def update_pnl(self, state : TradingState):
        """
        Updates the pnl.
        """
        def update_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price

        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value

        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()

    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]


    def ink_strategy(self, state: TradingState):
      perc_diff = 0 
      mean = -0.00001
      std =  0.00169
      print("ink")
      position_ink = self.get_position(SQUID_INK, state)
      print("position_int:", position_ink)  
      if self.ink_prev_mid_price:
        perc_diff = (self.get_mid_price(SQUID_INK, state) - self.ink_prev_mid_price) / self.ink_prev_mid_price
      else:
          self.ink_prev_mid_price = self.get_mid_price(SQUID_INK, state)

      bid_volume = self.position_limit[SQUID_INK] - position_ink
      ask_volume = -self.position_limit[SQUID_INK] - position_ink

      print(f"Position resin: {position_ink}")
      print(f"Bid Volume: {bid_volume}")
      print(f"Ask Volume: {ask_volume}")

      orders = []
      # order_ratio = self.get_order_ratio(SQUILD_INK, state)
      mid_price = self.get_mid_price(SQUID_INK, state)
      best_ask, best_ask_amount = list(state.order_depths[SQUID_INK].sell_orders.items())[0]
      best_bid, best_bid_amount = list(state.order_depths[SQUID_INK].buy_orders.items())[0]
      if perc_diff != 0: # tried 0.5 , 0.6 is worst, now tired 0.55, if no improvement then keep 0.5
          if perc_diff >= mean + 0.7 * std: # sell now buy later
                orders.append(Order(SQUID_INK, int(best_bid) , bid_volume)) # try bid_volume and ask_volume
                orders.append(Order(SQUID_INK, int(mid_price), ask_volume))

          elif perc_diff <= mean - 0.55 * std: # buy now sell later 
                orders.append(Order(SQUID_INK, int(best_ask), ask_volume))
                orders.append(Order(SQUID_INK, int(mid_price) , bid_volume))
      print("orders: ", orders)

      return orders

    def resin_strategy(self, state: TradingState, correction=1.0):

      position_resin = self.get_position(RESIN, state)

      bid_volume = self.position_limit[RESIN] - position_resin
      ask_volume = -self.position_limit[RESIN] - position_resin

      #print(f"Position resin: {position_resin}")
      #print(f"Bid Volume: {bid_volume}")
      #print(f"Ask Volume: {ask_volume}")

      orders = []
      best_ask, _volume1 = next(
          iter(state.order_depths[RESIN].sell_orders.items()), (None, None)
      )
      best_bid, __volume2 = next(
          iter(state.order_depths[RESIN].buy_orders.items()), (None, None)
      )
      mid_price = (best_bid + best_ask) / 2
      # self.resin_prices.append(mid_price)
      #log_return = np.log(mid_price/self.resin_prices[len(self.resin_prices)-2])
      #self.resin_log_return.append(log_return)
      print(f"best ask: {best_ask}\n")
      print(f"best bid: {best_bid}\n")
      
      adjustment = round((DEFAULT_PRICES[RESIN] - mid_price) * correction)
      
      # 20: 1.686k, 15: 1.706, 10: 1.721, 5: 1.759, 4: 1.759, 3: 1.759, 2: 1.759
      extra_adjustment_bid = 0
      extra_adjustment_ask = 0
      if position_resin > 5:
          extra_adjustment_ask = - 1
      if position_resin < -5:
          extra_adjustment_bid = 1
          
          
      orders.append(Order(RESIN, min(DEFAULT_PRICES[RESIN] - 1, best_bid + adjustment + extra_adjustment_bid), bid_volume))  # buy order
      orders.append(Order(RESIN, max(DEFAULT_PRICES[RESIN] + 1, best_ask + adjustment + extra_adjustment_ask), ask_volume))  # sell order
      return orders
  
    # def kelp_strategy(self, state: TradingState, correction=1.0):

    #   position_kelp = self.get_position(KELP, state)

    #   bid_volume = self.position_limit[KELP] - position_kelp
    #   ask_volume = -self.position_limit[KELP] - position_kelp

    #   #print(f"Position kelp: {position_kelp}")
    #   #print(f"Bid Volume: {bid_volume}")
    #   #print(f"Ask Volume: {ask_volume}")

    #   orders = []
    #   best_ask, _volume1 = next(
    #       iter(state.order_depths[KELP].sell_orders.items()), (None, None)
    #   )
    #   best_bid, __volume2 = next(
    #       iter(state.order_depths[KELP].buy_orders.items()), (None, None)
    #   )
    #   mid_price = (best_bid + best_ask) / 2
    #   self.past_prices[KELP].append(mid_price)
    #   # self.kelp_prices.append(mid_price)
    #   #log_return = np.log(mid_price/self.kelp_prices[len(self.kelp_prices)-2])
    #   #self.kelp_log_return.append(log_return)
    #   print(f"best ask: {best_ask}\n")
    #   print(f"best bid: {best_bid}\n")
      
    #   '''if len(self.past_prices[KELP]) < 10:
    #       return orders'''
      
      
    #   if mid_price > DEFAULT_PRICES[KELP]:
    #       ask_adjustment = -1
    #       bid_adjustment = 0
    #   elif mid_price < DEFAULT_PRICES[KELP]:
    #       ask_adjustment = 0
    #       bid_adjustment = 1
    
    #   else:
    #       ask_adjustment = 0
    #       bid_adjustment = 0
        
          
    #   orders.append(Order(KELP, best_bid + bid_adjustment, bid_volume))  # buy order
    #   orders.append(Order(KELP, best_ask + ask_adjustment, ask_volume))  # sell order
    #   return orders
 
    # def kelp_mean_reversion_strategy(self, state: TradingState):
    #     """
    #     Simple mean reversion strategy for KELP
    #     """
    #     position_kelp = self.get_position(KELP, state)
        
    #     # Get current mid price
    #     mid_price = self.get_mid_price(KELP, state)
        
    #     # Add to price history
    #     self.past_prices[KELP].append(mid_price)
        
    #     # Need enough price history to calculate z-score
    #     window_size = 9000  # Moving average window
    #     if len(self.past_prices[KELP]) < window_size:
    #         return []  # Not enough data yet
        
    #     # Calculate moving average
    #     ma_window = self.past_prices[KELP][-window_size:]
    #     ma = sum(ma_window) / window_size
        
    #     # Calculate standard deviation
    #     std_dev = math.sqrt(sum((price - ma) ** 2 for price in ma_window) / window_size)
        
    #     # Calculate z-score
    #     z_score = (mid_price - ma) / std_dev if std_dev > 0 else 0
        
    #     print(f"KELP Z-Score: {z_score:.2f}, MA: {ma:.2f}, Current: {mid_price:.2f}")
        
    #     # Define strategy parameters
    #     entry_threshold = 0.9
    #     exit_threshold = 0.1
        
    #     # Available position space
    #     bid_volume = self.position_limit[KELP] - position_kelp  # How much we can buy
    #     ask_volume = -self.position_limit[KELP] - position_kelp  # How much we can sell
        
    #     orders = []
        
    #     # Get best bid and ask from the market
    #     if len(state.order_depths[KELP].sell_orders) > 0:
    #         best_ask = min(state.order_depths[KELP].sell_orders.keys())
    #         print(f"best ask price: {best_ask}")
    #     else:
    #         best_ask = mid_price + 1  # Default if no ask orders
            
    #     if len(state.order_depths[KELP].buy_orders) > 0:
    #         best_bid = max(state.order_depths[KELP].buy_orders.keys())
    #         print(f"best bid price: {best_ask}")

    #     else:
    #         best_bid = mid_price - 1  # Default if no bid orders
        
    #     # Trading logic based on z-score
    #     if z_score < -entry_threshold and bid_volume > 0:
    #         # Price is below average (negative z-score), expect rise - BUY
    #         # Place a buy order at the best ask price
    #         orders.append(Order(KELP, best_ask, bid_volume))
    #         print(f"BUY signal: z_score={z_score:.2f}, price={best_ask}, volume={bid_volume}")
            
    #     elif z_score > entry_threshold and ask_volume < 0:
    #         # Price is above average (positive z-score), expect fall - SELL
    #         # Place a sell order at the best bid price
    #         orders.append(Order(KELP, best_bid, ask_volume))
    #         print(f"SELL signal: z_score={z_score:.2f}, price={best_bid}, volume={ask_volume}")
            
    #     elif position_kelp > 0 and z_score > -exit_threshold:
    #         # We have a long position and z-score has reverted - EXIT LONG
    #         # Sell our position at the best bid
    #         orders.append(Order(KELP, best_bid, -position_kelp))
    #         print(f"EXIT LONG signal: z_score={z_score:.2f}, price={best_bid}, volume={-position_kelp}")
            
    #     elif position_kelp < 0 and z_score < exit_threshold:
    #         # We have a short position and z-score has reverted - EXIT SHORT
    #         # Buy back our position at the best ask
    #         orders.append(Order(KELP, best_ask, -position_kelp))
    #         print(f"EXIT SHORT signal: z_score={z_score:.2f}, price={best_ask}, volume={-position_kelp}")
        
    #     return orders
    
    
    def kelp_mean_reversion_strategy(self, state: TradingState):
        """
        Simple and robust mean reversion strategy for KELP
        """
        # Check if KELP is in order depths
        if KELP not in state.order_depths:
            return []
            
        position_kelp = self.get_position(KELP, state)
        
        # Get current prices
        market_bids = state.order_depths[KELP].buy_orders
        market_asks = state.order_depths[KELP].sell_orders
        
        if len(market_bids) == 0 or len(market_asks) == 0:
            return []
            
        best_bid = max(market_bids.keys())
        best_ask = min(market_asks.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Add to price history
        self.past_prices[KELP].append(mid_price)
        
        # Need enough price history
        if len(self.past_prices[KELP]) < 30:
            return []
        
        # Calculate simple moving average of last 30 prices
        ma = sum(self.past_prices[KELP][-30:]) / 30
        
        # For stable products like KELP, use a simpler approach
        # Trade only when price deviates significantly from expected value
        expected_price = 2033  # Based on historical data
        
        # Conservative trading approach
        max_position = 10  # Limit max position
        
        # Available position capacity
        buy_capacity = max(0, max_position - position_kelp)
        sell_capacity = max(0, max_position + position_kelp)
        
        orders = []
        
        # Simple rules for stable product:
        # 1. Buy when price is below expected and we have capacity
        # 2. Sell when price is above expected and we have capacity
        # 3. Exit positions when price moves against us
        
        # Rule 1: Buy low
        if best_bid < expected_price - 3 and buy_capacity > 0 and position_kelp < 5:
            # Buy at best bid + 1 to increase execution probability
            buy_price = best_bid + 1
            buy_volume = min(3, buy_capacity)  # Small positions only
            orders.append(Order(KELP, buy_price, buy_volume))
        
        # Rule 2: Sell high
        elif best_ask > expected_price + 3 and sell_capacity > 0 and position_kelp > -5:
            # Sell at best ask - 1 to increase execution probability
            sell_price = best_ask - 1
            sell_volume = -min(3, sell_capacity)  # Small positions only
            orders.append(Order(KELP, sell_price, sell_volume))
        
        # Rule 3: Exit long positions if price drops
        elif position_kelp > 3 and mid_price < ma:
            # Sell at market price
            exit_volume = -min(position_kelp, 5)  # Partial exit
            orders.append(Order(KELP, best_bid, exit_volume))
        
        # Rule 4: Exit short positions if price rises
        elif position_kelp < -3 and mid_price > ma:
            # Buy at market price
            exit_volume = min(-position_kelp, 5)  # Partial exit
            orders.append(Order(KELP, best_ask, exit_volume))
        
        # Rule 5: Don't get caught with large positions
        elif abs(position_kelp) > 15:
            # Emergency exit if position too large
            if position_kelp > 0:
                orders.append(Order(KELP, best_bid, -min(position_kelp, 10)))
            else:
                orders.append(Order(KELP, best_ask, min(-position_kelp, 10)))
        
        return orders
    
 
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Any, Any]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)

        print(f"Log round {self.round}")
        
        # For debugging
        debug_info = [f"Round {self.round}", f"PnL {pnl}"]

        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)

        print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}")
        print(f"\tPnL {pnl}")

        self.below_110 = False
        


        # Initialize the method output dict as an empty dict
        result = {}

        

        # try:
        #     result[RESIN] = self.resin_strategy(state)           
        # except Exception as e:
        #     print("Error: ")
        #     print("Error occurred while executing resin_strategy:")
        #     print(f"Exception Type: {type(e).__name__}")
        #     print(f"Exception Message: {str(e)}")
        #     print("Stack Trace:") 
            
        # try:
        #     result[KELP] = self.kelp_mean_reversion_strategy(state)
        #     print(f"KELP orders: {result[KELP]}")  
        # except Exception as e:
        #     print("Error: ")
        #     print("Error occurred while executing kelp_strategy:")
        #     print(f"Exception Type: {type(e).__name__}")
        #     print(f"Exception Message: {str(e)}")
        #     print("Stack Trace:")
            
        try:
            kelp_orders, kelp_debug = self.kelp_mean_reversion_strategy(state, debug_info)
            result[KELP] = kelp_orders
            debug_info = kelp_debug  # Update with KELP strategy debug info
        except Exception as e:
            debug_info.append(f"KELP error: {str(e)}")

        # Convert debug info to a single string for trader data
        trader_data = "\nlogzzz".join(debug_info)

        # try:
        #     result[KELP] = self.kelp_strategy(state)
        # except Exception as e:
        #     print("Error: ")
        #     print("Error occurred while executing kelp_strategy:")
        #     print(f"Exception Type: {type(e).__name__}")
        #     print(f"Exception Message: {str(e)}")
        #     print("Stack Trace:")

        # try:
        #     result[SQUID_INK] = self.ink_strategy(state)
        # except Exception as e:
        #     print("Error: ")
        #     print("Error occurred while executing ink_strategy:")
        #     print(f"Exception Type: {type(e).__name__}")
        #     print(f"Exception Message: {str(e)}")
        #     print("Stack Trace:")

        conversions = 1


        return result, conversions, trader_data
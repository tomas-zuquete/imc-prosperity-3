import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
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

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()



class Trader:

    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            RESIN: 50,
            KELP: 50, 
            SQUID_INK: 50
        }
        trader_data = ""

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


    # def resin_strategy(self, state: TradingState):

    #   position_resin = self.get_position(RESIN, state)

    #   bid_volume = self.position_limit[RESIN] - position_resin
    #   ask_volume = -self.position_limit[RESIN] - position_resin

    #   print(f"Position resin: {position_resin}")
    #   print(f"Bid Volume: {bid_volume}")
    #   print(f"Ask Volume: {ask_volume}")

    #   orders = []
    #   order_ratio = self.get_order_ratio(RESIN, state)
    #   mid_price = self.get_mid_price(RESIN, state)

    #   best_ask, best_ask_amount = list(state.order_depths[RESIN].sell_orders.items())[0]
    #   best_bid, best_bid_amount = list(state.order_depths[RESIN].buy_orders.items())[0]
    #   print("best_ask_amt: ", best_ask_amount)
    #   print("best_bid_amt: ", best_bid_amount)
    #   print(order_ratio)
    #   if order_ratio > 0.3:
    #     orders.append(Order("RAINFOREST_RESIN", int(best_ask) - 1, ask_volume))  # buy order
    #     print("buy at: ", int(best_ask) - 1, -best_ask_amount)
    #   elif -1 >= order_ratio > -0.3:
    #     orders.append(Order("RAINFOREST_RESIN", int(best_bid) + 1, bid_volume))  # sell order
    #     print("sell at: ", int(best_bid) + 1, -best_bid_amount)
    #   else: 
    #     if position_resin > 0:
    #         orders.append(Order("RAINFOREST_RESIN", int(best_bid), bid_volume))  #sell
    #         print("sell at: ", int(best_bid) + 1, bid_volume)
    #     elif position_resin < 0:
    #         orders.append(Order("RAINFOREST_RESIN", int(best_ask), ask_volume)) 
    #         print("buy at: ", int(best_bid) - 1, ask_volume)
    #     else:
    #         adjustment = round((DEFAULT_PRICES[RESIN] - mid_price) * 0.15)
      
    #   # 20: 1.686k, 15: 1.706, 10: 1.721, 5: 1.759, 4: 1.759, 3: 1.759, 2: 1.759
    #     extra_adjustment_bid = 0
    #     extra_adjustment_ask = 0
    #     if position_resin > 5:
    #         extra_adjustment_ask = - 1
    #     if position_resin < -5:
    #         extra_adjustment_bid = 1
            
            
    #     orders.append(Order(RESIN, min(DEFAULT_PRICES[RESIN] - 1, best_bid + adjustment + extra_adjustment_bid), bid_volume))  # buy order
    #     orders.append(Order(RESIN, max(DEFAULT_PRICES[RESIN] + 1, best_ask + adjustment + extra_adjustment_ask), ask_volume)) 
            
    #   print("orders:", orders)
    #   return orders

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

    # def kelp_strategy(self, state: TradingState):
    #     position_KELP = self.get_position(KELP, state)

    #     bid_volume = self.position_limit[KELP] - position_KELP
    #     ask_volume = -self.position_limit[KELP] - position_KELP

    #     print(f"Position KELP: {position_KELP}")
    #     print(f"Bid Volume: {bid_volume}")
    #     print(f"Ask Volume: {ask_volume}")

    #     orders = []
    #     order_ratio = self.get_order_ratio(KELP, state)
    #     mid_price = self.get_mid_price(KELP, state)

    #     best_ask, best_ask_amount = list(state.order_depths[KELP].sell_orders.items())[0]
    #     best_bid, best_bid_amount = list(state.order_depths[KELP].buy_orders.items())[0]
    #     print("best_ask_amt: ", best_ask_amount)
    #     print("best_bid_amt: ", best_bid_amount)
    #     print(order_ratio)
    #     if order_ratio > 0.1: # mid_price will likely drop, sell now, buy later
    #         orders.append(Order(KELP, int(best_bid) - 1, bid_volume)) 
    #         orders.append(Order(KELP, int(best_ask) + 1, ask_volume))  
            
    #     elif order_ratio < -0.1: 
    #         orders.append(Order(KELP, int(best_bid) + 1, bid_volume))  
    #         orders.append(Order(KELP, int(best_ask) - 1, ask_volume))
    #     print("orders:", orders)
    #     return orders
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
  
    def kelp_strategy(self, state: TradingState, correction=1.0):

      position_kelp = self.get_position(KELP, state)

      bid_volume = self.position_limit[KELP] - position_kelp
      ask_volume = -self.position_limit[KELP] - position_kelp

      #print(f"Position kelp: {position_kelp}")
      #print(f"Bid Volume: {bid_volume}")
      #print(f"Ask Volume: {ask_volume}")

      orders = []
      best_ask, _volume1 = next(
          iter(state.order_depths[KELP].sell_orders.items()), (None, None)
      )
      best_bid, __volume2 = next(
          iter(state.order_depths[KELP].buy_orders.items()), (None, None)
      )
      mid_price = (best_bid + best_ask) / 2
      self.past_prices[KELP].append(mid_price)
      # self.kelp_prices.append(mid_price)
      #log_return = np.log(mid_price/self.kelp_prices[len(self.kelp_prices)-2])
      #self.kelp_log_return.append(log_return)
      print(f"best ask: {best_ask}\n")
      print(f"best bid: {best_bid}\n")
      
      '''if len(self.past_prices[KELP]) < 10:
          return orders'''
      
      
      if mid_price > DEFAULT_PRICES[KELP]:
          ask_adjustment = -1
          bid_adjustment = 0
      elif mid_price < DEFAULT_PRICES[KELP]:
          ask_adjustment = 0
          bid_adjustment = 1
    
      else:
          ask_adjustment = 0
          bid_adjustment = 0
        
          
      orders.append(Order(KELP, best_bid + bid_adjustment, bid_volume))  # buy order
      orders.append(Order(KELP, best_ask + ask_adjustment, ask_volume))  # sell order
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

        

        try:
            result[RESIN] = self.resin_strategy(state)
            
           
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing resin_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")

        try:
            result[KELP] = self.kelp_strategy(state)
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing kelp_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")

        try:
            result[SQUID_INK] = self.ink_strategy(state)
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing ink_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")

        conversions = 1

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, None
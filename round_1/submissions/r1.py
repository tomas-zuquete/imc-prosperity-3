import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState
import pandas as pd


# storing string as const to avoid typos, 1.7k pnl
SUBMISSION = "SUBMISSION"

RESIN = "RAINFOREST_RESIN"


PRODUCTS = [
    RESIN
]

DEFAULT_PRICES = {
    RESIN :10000
}

class Trader:

    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            RESIN: 50
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
        self.orchid_log_return = []


        self.ema_param = 0.5
        # self.df_log_return = pd.DataFrame([], columns=pd.Index(['Log_Return']))


    # utils
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    

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


    def resin_strategy(self, state: TradingState):

      position_resin = self.get_position(RESIN, state)

      bid_volume = self.position_limit[RESIN] - position_resin
      ask_volume = -self.position_limit[RESIN] - position_resin

      print(f"Position resin: {position_resin}")
      print(f"Bid Volume: {bid_volume}")
      print(f"Ask Volume: {ask_volume}")

      orders = []
      best_ask, _volume1 = next(
          iter(state.order_depths[RESIN].sell_orders.items()), (None, None)
      )
      best_bid, __volume2 = next(
          iter(state.order_depths[RESIN].buy_orders.items()), (None, None)
      )
      # mid_price = (best_bid + best_ask) / 2
      # self.resin_prices.append(mid_price)
      #log_return = np.log(mid_price/self.resin_prices[len(self.resin_prices)-2])
      #self.resin_log_return.append(log_return)
      print(f"best ask: {best_ask}\n")
      print(f"best bid: {best_bid}\n")
      orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] - 2, bid_volume))  # buy order
      orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] + 2, ask_volume))  # sell order
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
        #check if pnl has hit 110,000
        self.below_110 = False
        


        # Initialize the method output dict as an empty dict
        result = {}

        

        try:
            result[RESIN] = self.resin_strategy(state)
        except Exception as e:
            print("Error in resin strategy")
            print(e)
        
        conversions = 1


        return result, conversions, None
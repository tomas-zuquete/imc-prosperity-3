import math
from typing import Dict, List, Any, Tuple
import numpy as np
from datamodel import Order, TradingState
import pandas as pd


# storing string as const to avoid typos, 1.7k pnl
SUBMISSION = "SUBMISSION"

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"
PRODUCTS = [
    RESIN,
    KELP,
    SQUID_INK,
    CROISSANTS,
    JAMS,
    DJEMBES,
    PICNIC_BASKET1,
    PICNIC_BASKET2
]

DEFAULT_PRICES = {
    RESIN :10000,
    KELP:  2023,
    SQUID_INK: 1972,
    CROISSANTS: 500,
    JAMS: 500,
    DJEMBES: 500,
    PICNIC_BASKET1: 500,
    PICNIC_BASKET2: 500
}

def compute_SMA(prices: List[float], window: int) -> float:
        if len(prices) < window:
            return sum(prices) / len(prices)
        return sum(prices[-window:]) / window

def compute_STD(prices: List[float], window: int) -> float:
    sma = compute_SMA(prices, window)
    var = sum((p - sma) ** 2 for p in prices[-window:]) / window
    return math.sqrt(var)

def compute_RSI(prices: List[float], window: int = 14) -> float:
    if len(prices) < window + 1:
        return 50.0
    gains = [max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices))][-window:]
    losses = [max(prices[i - 1] - prices[i], 0) for i in range(1, len(prices))][-window:]
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))

def compute_stochastic(prices: List[float], window: int = 14) -> float:
    if len(prices) < window:
        return 50.0
    lowest = min(prices[-window:])
    highest = max(prices[-window:])
    if highest == lowest:
        return 50.0
    return (prices[-1] - lowest) / (highest - lowest) * 100.0

def compute_ATR(prices: List[float], window: int = 14) -> float:
    if len(prices) < window + 1:
        return 0.0
    tr_values = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))][-window:]
    return sum(tr_values) / window

def update_EMA(prev_ema: float, price: float, window: int) -> float:
    alpha = 2 / (window + 1)
    return alpha * price + (1 - alpha) * prev_ema
    def get_order_ratio(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        if len(market_asks) > 0 and len(market_bids) > 0:
            return (sum(market_bids) - sum(market_asks)) / (sum(market_bids) + sum(market_asks)) 

class Trader:

    def __init__(self) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            RESIN: 50,
            KELP: 50, 
            SQUID_INK: 50,
            CROISSANTS: 250,
            JAMS: 350,
            DJEMBES: 60,
            PICNIC_BASKET1: 60,
            PICNIC_BASKET2: 100
        }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position
        self.ou_params = {"mu": -0.000013, "theta": 0.01, "sigma": 0.000025}
        self.kelp_ou_params = {"mu": -0.000013, "theta": 0.01, "sigma": 0.000025}
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        
        self.ema_param = 0.5
        self.new_history: Dict[str, List[float]] = {CROISSANTS: [], JAMS: [], DJEMBES: []}
        self.basket1_history: List[float] = []
        self.basket1_ema: float = None
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.ink_prev_mid_price = None
        self.kelp_prev_mid_price = None
        self.ink_mid_prices = []
        self.kelp_mid_prices = []
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

    def croissants_strategy(self, state: TradingState) -> List[Order]:
        product = CROISSANTS
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        self.new_history[product].append(mid)
        prices = self.new_history[product]
        window = 20
        sma20 = compute_SMA(prices, window) if len(prices) >= window else compute_SMA(prices, len(prices))
        std20 = compute_STD(prices, window) if len(prices) >= window else compute_STD(prices, len(prices))
        lower_band = sma20 - 2 * std20
        rsi = compute_RSI(prices, 14)
        signal = 0
        if mid <= lower_band and rsi < 28:
            signal = 1
        elif mid >= sma20 or rsi > 72:
            signal = -1
        if signal == 1:
            bid_price = int(mid - 1)
            ask_price = int(mid + 2)
        elif signal == -1:
            bid_price = int(mid - 2)
            ask_price = int(mid + 1)
        else:
            bid_price = int(mid - 1)
            ask_price = int(mid + 1)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    def jams_strategy(self, state: TradingState) -> List[Order]:
        product = JAMS
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        self.new_history[product].append(mid)
        prices = self.new_history[product]
        stoch_val = compute_stochastic(prices, 14)
        signal = 0
        if stoch_val < 25:
            signal = 1
        elif stoch_val > 75:
            signal = -1
        if signal == 1:
            bid_price = int(mid - 1)
            ask_price = int(mid + 2)
        elif signal == -1:
            bid_price = int(mid - 2)
            ask_price = int(mid + 1)
        else:
            bid_price = int(mid - 1)
            ask_price = int(mid + 1)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    def djembes_strategy(self, state: TradingState) -> List[Order]:
        product = DJEMBES
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        self.new_history[product].append(mid)
        prices = self.new_history[product]
        rsi = compute_RSI(prices, 14)
        atr = compute_ATR(prices, 14)
        atr_med = np.median(prices[-14:]) if len(prices) >= 14 else atr
        signal = 0
        if rsi < 28 and atr > atr_med:
            signal = 1
        elif rsi > 72 and atr > atr_med:
            signal = -1
        if signal == 1:
            bid_price = int(mid - 1)
            ask_price = int(mid + 2)
        elif signal == -1:
            bid_price = int(mid - 2)
            ask_price = int(mid + 1)
        else:
            bid_price = int(mid - 1)
            ask_price = int(mid + 1)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    def picnic_basket1_strategy(self, state: TradingState) -> List[Order]:
        product = PICNIC_BASKET1
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        crois_mid = self.get_mid_price(CROISSANTS, state)
        jams_mid = self.get_mid_price(JAMS, state)
        djembes_mid = self.get_mid_price(DJEMBES, state)
        fair_val = (6 * crois_mid + 3 * jams_mid + 1 * djembes_mid) / 10.0
        composite_mid = fair_val
        self.basket1_history.append(composite_mid)
        if len(self.basket1_history) < 50:
            basket_ema = sum(self.basket1_history) / len(self.basket1_history)
        else:
            alpha = 2 / (50 + 1)
            if self.basket1_ema is None:
                basket_ema = sum(self.basket1_history[-50:]) / 50
            else:
                basket_ema = alpha * composite_mid + (1 - alpha) * self.basket1_ema
            self.basket1_ema = basket_ema
        signal = 1 if composite_mid > basket_ema else -1
        delta = 1
        if signal == 1:
            bid_price = int(composite_mid - delta)
            ask_price = int(composite_mid + delta + 1)
        else:
            bid_price = int(composite_mid - delta - 1)
            ask_price = int(composite_mid + delta)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]

    def picnic_basket2_strategy(self, state: TradingState) -> List[Order]:
        product = PICNIC_BASKET2
        pos = self.get_position(product, state)
        mid = self.get_mid_price(product, state)
        crois_mid = self.get_mid_price(CROISSANTS, state)
        jams_mid = self.get_mid_price(JAMS, state)
        fair_val = (4 * crois_mid + 2 * jams_mid) / 6.0
        delta = 1
        bid_price = int(min(mid, fair_val) - delta)
        ask_price = int(max(mid, fair_val) + delta)
        bid_volume = self.position_limit[product] - pos
        ask_volume = -self.position_limit[product] - pos
        return [Order(product, bid_price, bid_volume), Order(product, ask_price, ask_volume)]
    
    def resin_strategy(self, state: TradingState):

      position_resin = self.get_position(RESIN, state)

      bid_volume = self.position_limit[RESIN] - position_resin
      ask_volume = -self.position_limit[RESIN] - position_resin

      print(f"Position resin: {position_resin}")
      print(f"Bid Volume: {bid_volume}")
      print(f"Ask Volume: {ask_volume}")

      orders = []
      order_ratio = self.get_order_ratio(RESIN, state)
      mid_price = self.get_mid_price(RESIN, state)

      best_ask, best_ask_amount = list(state.order_depths[RESIN].sell_orders.items())[0]
      best_bid, best_bid_amount = list(state.order_depths[RESIN].buy_orders.items())[0]
      print("best_ask_amt: ", best_ask_amount)
      print("best_bid_amt: ", best_bid_amount)
      print(order_ratio)
      if order_ratio > 0.3:
        orders.append(Order("RAINFOREST_RESIN", int(best_ask) - 1, ask_volume))  # buy order
        print("buy at: ", int(best_ask) - 1, -best_ask_amount)
      elif -1 >= order_ratio > -0.3:
        orders.append(Order("RAINFOREST_RESIN", int(best_bid) + 1, bid_volume))  # sell order
        print("sell at: ", int(best_bid) + 1, -best_bid_amount)
      else: 
        if position_resin > 0:
            orders.append(Order("RAINFOREST_RESIN", int(best_bid), bid_volume))  #sell
            print("sell at: ", int(best_bid) + 1, bid_volume)
        elif position_resin < 0:
            orders.append(Order("RAINFOREST_RESIN", int(best_ask), ask_volume)) 
            print("buy at: ", int(best_bid) - 1, ask_volume)
        else:
            adjustment = round((DEFAULT_PRICES[RESIN] - mid_price) * 0.15)
      
      # 20: 1.686k, 15: 1.706, 10: 1.721, 5: 1.759, 4: 1.759, 3: 1.759, 2: 1.759
        extra_adjustment_bid = 0
        extra_adjustment_ask = 0
        if position_resin > 5:
            extra_adjustment_ask = - 1
        if position_resin < -5:
            extra_adjustment_bid = 1
            
            
        orders.append(Order(RESIN, min(DEFAULT_PRICES[RESIN] - 1, best_bid + adjustment + extra_adjustment_bid), bid_volume))  # buy order
        orders.append(Order(RESIN, max(DEFAULT_PRICES[RESIN] + 1, best_ask + adjustment + extra_adjustment_ask), ask_volume)) 
            
      print("orders:", orders)
      return orders
    
    def fit_ou(self, X: list[float]) -> tuple[float, float, float]:
        if len(X) < 20:
            return self.ou_params["mu"], self.ou_params["theta"], self.ou_params["sigma"]
        X = np.array(X)
        n = len(X)
        dt = 1.0
        mu = X.mean()
        X_t = X[:-1]
        X_t1 = X[1:]
        cov = np.cov(X_t, X_t1)[0, 1]
        var = np.var(X_t)
        theta = -np.log(cov / var) / dt if var > 0 else 0.1
        theta = max(0.01, min(theta, 1.0))
        drift = mu + (X_t - mu) * np.exp(-theta * dt)
        residuals = X_t1 - drift
        sigma = np.sqrt(2 * theta * np.var(residuals) / (1 - np.exp(-2 * theta * dt)))
        sigma = max(1e-6, sigma)
        return mu, theta, sigma
    
    # def ink_strategy(self, state: TradingState):
    #   perc_diff = 0 
    #   mean = -0.00001
    #   std =  0.00169
    #   position_ink = self.get_position(SQUID_INK, state)
    #   if self.ink_prev_mid_price:
    #     perc_diff = (self.get_mid_price(SQUID_INK, state) - self.ink_prev_mid_price) / self.ink_prev_mid_price
    #   else:
    #       self.ink_prev_mid_price = self.get_mid_price(SQUID_INK, state)

    #   bid_volume = self.position_limit[SQUID_INK] - position_ink
    #   ask_volume = -self.position_limit[SQUID_INK] - position_ink

    #   orders = []
    #   # order_ratio = self.get_order_ratio(SQUILD_INK, state)
    #   mid_price = self.get_mid_price(SQUID_INK, state)
    #   best_ask, best_ask_amount = list(state.order_depths[SQUID_INK].sell_orders.items())[0]
    #   best_bid, best_bid_amount = list(state.order_depths[SQUID_INK].buy_orders.items())[0]
    #   if perc_diff != 0: # tried 0.5 , 0.6 is worst, now tired 0.55, if no improvement then keep 0.5
    #       if perc_diff >= mean + 0.7 * std: # sell now buy later
    #             orders.append(Order(SQUID_INK, int(best_bid) , best_bid_amount)) # try bid_volume and ask_volume
    #             orders.append(Order(SQUID_INK, int(mid_price), best_ask_amount))

    #       elif perc_diff <= mean - 1 * std: # buy now sell later # tried 0.55
    #             orders.append(Order(SQUID_INK, int(best_ask), best_ask_amount))
    #             orders.append(Order(SQUID_INK, int(mid_price) , best_bid_amount))
    #   print("orders: ", orders)

    #   return orders

    def ink_strategy(self, state: TradingState):
        mid_price = self.get_mid_price(SQUID_INK, state)
        
        # Store mid-price for real-time updates
        self.ink_mid_prices.append(mid_price)
        
        # Calculate percentage change
        perc_diff = 0
        if self.ink_prev_mid_price and self.ink_prev_mid_price != 0:
            perc_diff = (mid_price - self.ink_prev_mid_price) / self.ink_prev_mid_price
        self.ink_prev_mid_price = mid_price
        
        # Update OU parameters periodically
        if len(self.ink_mid_prices) >= 20 and state.timestamp % 20 == 0:
            X = [(self.ink_mid_prices[i] - self.ink_mid_prices[i-1]) / self.ink_mid_prices[i-1]
                for i in range(1, len(self.ink_mid_prices)) if self.ink_mid_prices[i-1] != 0]
            if X:
                mu, theta, sigma = self.fit_ou(X)
                self.ou_params.update({"mu": mu, "theta": theta, "sigma": sigma})
                print(f"Updated OU params: mu={mu:.6f}, theta={theta:.4f}, sigma={sigma:.6f}")
        
        # Trading logic
        mu, theta, sigma = self.ou_params["mu"], self.ou_params["theta"], self.ou_params["sigma"]
        position_ink = self.get_position(SQUID_INK, state)
        z_score = (perc_diff - mu) / (sigma / np.sqrt(2 * theta)) if sigma > 0 else 0
        
        orders = []
        max_trade_size = 50
        
        bid_volume = self.position_limit[SQUID_INK] - position_ink
        ask_volume = -self.position_limit[SQUID_INK] - position_ink
        
        best_ask = min(state.order_depths[SQUID_INK].sell_orders.keys(), default=int(mid_price + 1))
        best_bid = max(state.order_depths[SQUID_INK].buy_orders.keys(), default=int(mid_price - 1))
        
        if perc_diff != 0:
            if z_score >= 4 and ask_volume > 0:
                size = min(max_trade_size, ask_volume, abs(state.order_depths[SQUID_INK].buy_orders.get(best_bid, 0)))
                if size > 0:
                    orders.append(Order(SQUID_INK, best_bid, -size))
            elif z_score <= -4 and bid_volume > 0:
                size = min(max_trade_size, bid_volume, abs(state.order_depths[SQUID_INK].sell_orders.get(best_ask, 0)))
                if size > 0:
                    orders.append(Order(SQUID_INK, best_ask, size))
            else:
                fair_price = mid_price * (1 + mu)
                bid_price = int(fair_price - 2)
                ask_price = int(fair_price + 2)
                
                orders.append(Order(SQUID_INK, bid_price, bid_volume))
                orders.append(Order(SQUID_INK, ask_price, ask_volume))
        
        print(f"z_score: {z_score:.2f}, perc_diff: {perc_diff:.6f}, orders: {[("Ink", o.price, o.quantity) for o in orders]}")
        
        return orders

    def kelp_strategy(self, state: TradingState):
        mid_price = self.get_mid_price(KELP, state)
        
        # Store mid-price for real-time updates
        self.kelp_mid_prices.append(mid_price)
        
        # Calculate percentage change
        perc_diff = 0
        if self.kelp_prev_mid_price and self.kelp_prev_mid_price != 0:
            perc_diff = (mid_price - self.kelp_prev_mid_price) / self.kelp_prev_mid_price
        self.kelp_prev_mid_price = mid_price
        
        # Update OU parameters periodically
        if len(self.kelp_mid_prices) >= 20 and state.timestamp % 20 == 0:
            X = [(self.kelp_mid_prices[i] - self.kelp_mid_prices[i-1]) / self.kelp_mid_prices[i-1]
                for i in range(1, len(self.kelp_mid_prices)) if self.kelp_mid_prices[i-1] != 0]
            if X:
                mu, theta, sigma = self.fit_ou(X)
                self.kelp_ou_params.update({"mu": mu, "theta": theta, "sigma": sigma})
                print(f"Updated OU params: mu={mu:.6f}, theta={theta:.4f}, sigma={sigma:.6f}")
        
        # Trading logic
        mu, theta, sigma = self.kelp_ou_params["mu"], self.kelp_ou_params["theta"], self.ou_pkelp_ou_paramsarams["sigma"]
        position_ink = self.get_position(KELP, state)
        z_score = (perc_diff - mu) / (sigma / np.sqrt(2 * theta)) if sigma > 0 else 0
        
        orders = []
        max_trade_size = 50
        
        bid_volume = self.position_limit[KELP] - position_ink
        ask_volume = -self.position_limit[KELP] - position_ink
        
        best_ask = min(state.order_depths[KELP].sell_orders.keys(), default=int(mid_price + 1))
        best_bid = max(state.order_depths[KELP].buy_orders.keys(), default=int(mid_price - 1))
        
        if perc_diff != 0:
            if z_score >= 0.5 and ask_volume > 0:
                size = min(max_trade_size, ask_volume, abs(state.order_depths[KELP].buy_orders.get(best_bid, 0)))
                if size > 0:
                    orders.append(Order(KELP, best_bid, -size))
            elif z_score <= -0.5 and bid_volume > 0:
                size = min(max_trade_size, bid_volume, abs(state.order_depths[KELP].sell_orders.get(best_ask, 0)))
                if size > 0:
                    orders.append(Order(KELP, best_ask, size))
            else:
                fair_price = mid_price * (1 + mu)
                bid_price = int(fair_price - 1)
                ask_price = int(fair_price + 1)
                
                orders.append(Order(KELP, bid_price, bid_volume))
                orders.append(Order(KELP, ask_price, ask_volume))
        
        print(f"z_score: {z_score:.2f}, perc_diff: {perc_diff:.6f}, orders: {[("Kelp", o.price, o.quantity) for o in orders]}")
        
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

    #     print(order_ratio)
    #     if order_ratio > 0.1: # mid_price will likely drop, sell now, buy later
    #         orders.append(Order(KELP, int(best_bid), bid_volume)) 
    #         orders.append(Order(KELP, int(mid_price) - 1, ask_volume))  
            
    #     elif order_ratio < -0.1: 
    #         orders.append(Order(KELP, int(best_ask), ask_volume))  
    #         orders.append(Order(KELP, int(mid_price) + 1, bid_volume))
    #     print("orders:", orders)
    #     return orders
    # def resin_strategy(self, state: TradingState, correction=1.0):

    #   position_resin = self.get_position(RESIN, state)

    #   bid_volume = self.position_limit[RESIN] - position_resin
    #   ask_volume = -self.position_limit[RESIN] - position_resin

    #   #print(f"Position resin: {position_resin}")
    #   #print(f"Bid Volume: {bid_volume}")
    #   #print(f"Ask Volume: {ask_volume}")

    #   orders = []
    #   best_ask, _volume1 = next(
    #       iter(state.order_depths[RESIN].sell_orders.items()), (None, None)
    #   )
    #   best_bid, __volume2 = next(
    #       iter(state.order_depths[RESIN].buy_orders.items()), (None, None)
    #   )
    #   mid_price = (best_bid + best_ask) / 2
    #   # self.resin_prices.append(mid_price)
    #   #log_return = np.log(mid_price/self.resin_prices[len(self.resin_prices)-2])
    #   #self.resin_log_return.append(log_return)
    #   print(f"best ask: {best_ask}\n")
    #   print(f"best bid: {best_bid}\n")
      
    #   adjustment = round((DEFAULT_PRICES[RESIN] - mid_price) * correction)
      
    #   # 20: 1.686k, 15: 1.706, 10: 1.721, 5: 1.759, 4: 1.759, 3: 1.759, 2: 1.759
    #   extra_adjustment_bid = 0
    #   extra_adjustment_ask = 0
    #   if position_resin > 5:
    #       extra_adjustment_ask = - 1
    #   if position_resin < -5:
    #       extra_adjustment_bid = 1
          
          
    #   orders.append(Order(RESIN, min(DEFAULT_PRICES[RESIN] - 1, best_bid + adjustment + extra_adjustment_bid), bid_volume))  # buy order
    #   orders.append(Order(RESIN, max(DEFAULT_PRICES[RESIN] + 1, best_ask + adjustment + extra_adjustment_ask), ask_volume))  # sell order
    #   return orders
  
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

        try:
            result[CROISSANTS] = self.croissants_strategy(state)
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing crossants_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")

        try:
            result[JAMS] = self.jams_strategy(state)
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing jams_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")

        try:
            result[DJEMBES] = self.djembes_strategy(state)
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing jams_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")

        try:
            result[PICNIC_BASKET1] = self.picnic_basket1_strategy(state)
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing jams_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")

        try:
            result[PICNIC_BASKET2] = self.picnic_basket2_strategy(state)
        except Exception as e:
            print("Error: ")
            print("Error occurred while executing jams_strategy:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Stack Trace:")




        conversions = 1


        return result, conversions, None
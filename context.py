import pandas as pd
import numpy as np
import talib
from datetime import datetime

class Position(object):
    def __init__(self, starting):
        self.starting = starting
        self.current = self.starting

    def reset(self):
        self.current = self.starting


class Context(object):
    def __init__(self):
        self.buys, self.sells = [], []
        self.indicators_funcs = {
            'SMA' : talib.SMA,
            'RSI' : talib.RSI,
            'EMA' : talib.EMA,
            'DEMA' : talib.DEMA,
            'T3' : talib.T3,
        }

    def buy(self, quantity):
        price = self.close() * quantity
        cost = price * (1 + self.commission_fee)
        if cost < self.cash.current and self.cash.current:
            self.cash.current -= cost
            self.position.current += quantity
            self.buys.append({'time': self.current_ts, 'price':self.close(), 'cost': cost, 'quantity': quantity})
            hr_ts = datetime.fromtimestamp(self.current_ts).strftime('%Y-%m-%d %H:%M:%S')
            print("Buying", quantity, "at", self.close(), "on", hr_ts)
            #print("Cash and position now", self.cash.current, self.position.current)
            return True

    def buy_all(self):
        if self.cash.current > 10:
            quantity = self.cash.current / (self.close() * (1.0 + self.commission_fee))
            return self.buy(quantity)

    def sell(self, quantity):
        price = self.close() * quantity
        if quantity <= self.position.current and quantity:
            self.cash.current += (price / (1.0 + self.commission_fee))
            self.position.current -= quantity
            self.sells.append({'time': self.current_ts, 'price': self.close(), 'cost': price / (1.0 + self.commission_fee), 'quantity': quantity})
            hr_ts = datetime.fromtimestamp(self.current_ts).strftime('%Y-%m-%d %H:%M:%S')
            print("Selling", quantity, "at", self.close(), "on", hr_ts)
            #print("Cash and position now", self.cash.current, self.position.current)
            return True

    def sell_all(self):
        if self.position.current > 0:
            return self.sell(self.position.current)


class NumpyContext(Context):
    def __init__(self, capital, starting_pos, all_candles, commission_fee, candle_size):
        super().__init__()
        self.starting_cash, self.starting_pos = capital, starting_pos
        self.all_candles = all_candles
        self.all_indicators = {}
        self.change_candle_size(candle_size)
        self.set_getters()
        self.current_idx = -1
        self.commission_fee = commission_fee
        self.set_record_object()
        self.cash = Position(capital)
        self.position = Position(starting_pos)
        self.reset_generator()

    @property
    def candles(self):
        return self.all_candles[self.current_candle_size]

    def change_candle_size(self, candle_size):
        self.current_candle_size = candle_size
        if self.current_candle_size not in self.all_indicators:
            self.all_indicators[self.current_candle_size] = {}

    def set_record_object(self):
        self.records = np.zeros((self.candles.shape[0], self.candles.shape[1]+2))
        self.records[0:self.candles.shape[0], 0:self.candles.shape[1]] = self.candles

    def reset(self):
        self.cash.reset()
        self.position.reset()
        self.current_idx = -1
        self.progress()

    def reset_generator(self):
        self.generator = [row for row in self.candles]

    def set_getters(self):
        self.open = lambda: self.current_candle[1]
        self.close = lambda: self.current_candle[2]
        self.low = lambda: self.current_candle[3]
        self.high = lambda: self.current_candle[4]
        self.volume = lambda: self.current_candle[5]

    @property
    def portfolio_value(self):
        return self.cash.current + self.position.current*self.close()

    @property
    def indicators(self):
        return self.all_indicators[self.current_candle_size]

    def require_indicator(self, indicator_name, timeperiod=3):
        if indicator_name not in self.indicators or timeperiod not in self.indicators[indicator_name]:
            self.compute_indicator(indicator_name, timeperiod=timeperiod)

    def compute_indicator(self, indicator_name, timeperiod=3):
        if indicator_name not in self.indicators:
            self.indicators[indicator_name] = {}
        self.indicators[indicator_name][timeperiod] = self.indicators_funcs[indicator_name](self.full_closes, timeperiod=timeperiod)

    def progress(self):
        self.jump_by(1)

    def jump_by(self, length):
        self.current_idx += length
        if self.current_idx >= len(self.generator):
            raise StopIteration
        self.current_candle = self.generator[self.current_idx]
        self.current_ts = self.current_candle[0]

    def record_current_portfolio(self):
        self.records[self.current_idx][6] = self.cash.current
        self.records[self.current_idx][7] = self.position.current

    @property
    def full_closes(self):
        return self.candles[:,2]

    @property
    def closes(self):
        return self.candles[:self.current_idx+1,2]

    def indicator_now(self, indicator_name, timeperiod=3):
        return self.indicators[indicator_name][timeperiod][self.current_idx]


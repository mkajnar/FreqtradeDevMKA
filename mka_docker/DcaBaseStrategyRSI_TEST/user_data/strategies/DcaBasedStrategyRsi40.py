import os
import sys
import time
import datetime
from datetime import timedelta
from threading import Thread
from typing import Optional
from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pickle

from .DcaBasedStrategyBase import DcaBasedStrategyBase


class DcaBasedStrategyRsi40(DcaBasedStrategyBase):

    def __init__(self, config: dict):
        super().__init__(config)

        self.rsi = 40

        self.stop_buy = IntParameter(0, 1, default=1, space='buy')
        self.timeframe = '5m'
        self.higher_timeframe = '1h'
        # Stoploss:
        self.stoploss = -0.10
        # Optimal timeframe
        # Rebuy feature
        self.position_adjustment_enable = True
        # Example specific variables
        self.max_dca_orders = 20
        # This number is explained a bit further down
        self.max_dca_multiplier = 5.5

        self.dca_koef = 0.25

        from .trailing_sl import use_sell_signal, trailing_stop, trailing_stop_positive, trailing_stop_positive_offset, \
            trailing_only_offset_is_reached
        self.use_sell_signal = use_sell_signal
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.trailing_only_offset_is_reached = trailing_only_offset_is_reached

        # slovnik pro DCA orders
        self.dca_orders = {}
        # self.blocked = []
        self.btc_candles = []

        Thread(target=lambda: self.start_btc_controller()).start()

        self.load_dca_orders()
        # self.load_blocked_pairs()

        # pomocny slovnik pro profity men - pouzito pro vypocty casove brzdy
        self.profits = {}

        # ROI table:
        self.minimal_roi = {
            "0": 0.10,
            "15": 0.07,
            "60": 0.05,
            "120": 0.03,
            "180": 0.01
        }

        self.unfilledtimeout = {
            'buy': 60 * 5,
            'sell': 60 * 10
        }

        self.order_types = {
            'buy': 'market',
            'sell': 'market',
            'stoploss': 'market',
            'stoploss_on_exchange': False
        }

        self.plot_config = {
            "main_plot": {
                "tema": {},
                "sar": {
                    "color": "white"
                },
                "sma9": {
                    "color": "#8dca58",
                    "type": "line"
                },
                "sma20": {
                    "color": "#62df8c",
                    "type": "line"
                }
            },
            "subplots": {
                "RSI": {
                    "rsi": {
                        "color": "red"
                    }
                }
            }
        }

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """

        try:
            # plneni slovniku profitu prubeznymi profity
            self.profits[trade.pair] = current_profit
            # vsechny nakup meny
            filled_buys = trade.select_filled_orders('buy')
            count_of_buys = len(filled_buys)
            last_candle, previous_candle = self.obtain_last_prev_candles(trade.pair)

            if last_candle is not None and previous_candle is not None:

                # pokud mena je ve slovniku, uz jsme nakupovali
                if trade.pair in self.dca_orders.keys():
                    # zjisteni rozdilu casu od posledniho nakupu
                    t = (datetime.datetime.now() - self.dca_orders[trade.pair])
                    # kontrola proti vypocitanemu koeficientu pro brzdu casu
                    if t.total_seconds() <= int(self.get_koef()) * 60:
                        return None
                else:
                    self.dca_orders[trade.pair] = datetime.datetime.now()
                    self.save_dca_orders()
                    return None

                # nakup pres DCA jen pokud je ztrata 1%
                if current_profit > -0.005:
                    return None

                # ochrana padajicich svici - pro ucely testu zakomentovat 4 radky
                # ochrana na ADX last and prev candle
                # ochrana na ADX mensi nez 25

                #if current_profit < 0:
                #    return None

                if last_candle['close'] <= previous_candle['close'] \
                        or ((last_candle['adx'] <= previous_candle['adx']) and last_candle['adx'] < 25) and last_candle['volume']==0:
                    return None

                if 0 < count_of_buys <= self.max_dca_orders:
                    try:
                        # This returns first order stake size
                        stake_amount = filled_buys[0].cost
                        # This then calculates current safety order size
                        stake_amount = stake_amount * (1 + (count_of_buys * self.dca_koef))

                        # zapis casu dle meny
                        self.dca_orders[trade.pair] = datetime.datetime.now()
                        # ulozit hned na disk
                        self.save_dca_orders()
                        return stake_amount
                    except Exception as exception:
                        return None

        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            pass

        return None

    def obtain_last_prev_candles(self, pair):
        try:
            # Obtain pair dataframe (just to show how to access it)
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            # Only buy when not actively falling price.
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            return last_candle, previous_candle
        except:
            return None, None
            pass

    def load_dca_orders(self):
        try:
            # nacteni DCA orders mezi restarty
            if os.path.exists(f'user_data/dca_orders_{self.rsi}'):
                with open(f'user_data/dca_orders_{self.rsi}', 'rb') as handle:
                    self.dca_orders = pickle.load(handle)
        except:
            pass

    def save_dca_orders(self):
        try:
            with open(f'user_data/dca_orders_{self.rsi}', 'wb') as handle:
                pickle.dump(self.dca_orders, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass

    # def load_blocked_pairs(self):
    #     try:
    #         # nacteni blocked pairs mezi restarty
    #         if os.path.exists('user_data/blocked_pairs'):
    #             with open('user_data/blocked_pairs', 'rb') as handle:
    #                 self.blocked = pickle.load(handle)
    #                 # vypada to, ze lock drzi v DB
    #                 # for _pair in self.blocked:
    #                 #     _block_year = datetime.datetime.now() + timedelta(days=365)
    #                 #     self.lock_pair(pair=_pair, until=_block_year)
    #     except:
    #         # exc_type, exc_obj, exc_tb = sys.exc_info()
    #         # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #         # print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
    #         pass

    # def save_blocked_pairs(self):
    #     try:
    #         with open('user_data/blocked_pairs', 'wb') as handle:
    #             pickle.dump(self.blocked, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     except:
    #         pass

    def get_koef(self):
        try:
            positive = 0
            negative = 0
            # hledani ve slovniku profitu a pocitani cervenych a zelenych pozic
            for k in self.profits.keys():
                if self.profits[k] > 0:
                    positive += 1
                else:
                    negative += 1
            k = (negative - positive)
            # pokud brzda vyjde mensi nez 5 minuty
            if k < 5:
                # tak natvrdo 5 minuty nastav
                k = 5

            return int(round(k))
        except:
            # pri chybe nastav 15 minut
            return 15
            pass

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        try:
            if sell_reason == 'stop_loss' or 'sell' in sell_reason:
                if 'force' in sell_reason or sell_reason.startswith('stop_loss'):
                    # self.blocked.append(pair)
                    # self.save_blocked_pairs()
                    _block_year = datetime.datetime.now() + timedelta(days=365)
                    self.lock_pair(pair=pair, until=_block_year, reason=sell_reason)
                    pass
                return True
            pr = trade.calc_profit_ratio(rate)
            if pr <= 0:
                return False
        except Exception as ex:
            pass

        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # self.get_recommendation(dataframe, metadata['pair'].replace('/', ''))
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        last_candle, previous_candle = self.obtain_last_prev_candles(metadata['pair'])
        if last_candle is not None and previous_candle is not None:
            if metadata['pair'] == 'BTC/USDT':
                _found = [s for s in self.btc_candles if
                          s['open'] == last_candle['open'] and s['close'] == last_candle['close']]
                if not _found:
                    self.btc_candles.append(last_candle)

        # dataframe['sar'] = ta.SAR(dataframe)
        # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['sma9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        # dataframe['hour'] = dataframe['date'].dt.hour
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=23)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bollinger_20_upperband'] = bollinger['upper']
        # dataframe['bollinger_20_lowerband'] = bollinger['lower']
        # dataframe['ema_fast_slow_pct'] = dataframe['ema_fast'] / dataframe['ema_slow'] * 100

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.stop_buy == 1:
            dataframe.loc[
                (
                        (dataframe['volume'].gt(0)) &
                        (dataframe['rsi'].gt(self.rsi))
                ),
                'buy'] = 0
        else:
            dataframe.loc[
                (
                        (dataframe['volume'].gt(0)) &
                        (dataframe['rsi'].gt(self.rsi))
                ),
                'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] >= 85) &
                    (dataframe['volume'].gt(0))
            ),
            'sell'] = 1
        return dataframe

    def start_btc_controller(self):
        while True:
            try:
                prices = list(set([s['close'] for s in self.btc_candles]))
                if len(prices) >= 3:
                    if prices[-1]  > prices[-2] > prices[-3]:
                        self.stop_buy = IntParameter(0, 1, default=0, space='buy')
                    else:
                        self.stop_buy = IntParameter(0, 1, default=1, space='buy')
                else:
                    self.stop_buy = IntParameter(0, 1, default=1, space='buy')
            except:
                self.stop_buy = IntParameter(0, 1, default=1, space='buy')
                pass
            time.sleep(3)

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

from user_data.strategies.dca_setting import dca_percent
from user_data.strategies.roi_settings import get_rois
from user_data.strategies.tsl_settings import stoploss, use_sell_signal, trailing_stop_positive, \
    trailing_stop_positive_offset, trailing_stop, trailing_only_offset_is_reached


class DcaBasedStrategy(IStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.buy_rsi = 50
        self.dca_rsi = 30
        self.timeframe = '1m'
        self.higher_timeframe = '1h'
        #jen debug
        self.dca_wait_secs = 600
        #self.dca_wait_secs = 300
        self.minimal_roi = get_rois()

        self.stoploss = stoploss
        self.use_sell_signal = use_sell_signal
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.trailing_only_offset_is_reached = trailing_only_offset_is_reached
        self.stop_buy = IntParameter(0, 1, default=1, space='buy')
        self.position_adjustment_enable = True
        self.max_dca_orders = 5
        self.max_dca_multiplier = 5
        self.dca_koef = 0.5
        self.dca_orders = {}
        self.profits = {}
        self.btc_candles = []

        self.load_dca_orders()

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
            last_candle, previous_candle = self.obtain_last_prev_candles(trade.pair, self.timeframe)

            if last_candle is not None and previous_candle is not None:

                if trade.pair in self.dca_orders.keys():
                    t = (datetime.datetime.now() - self.dca_orders[trade.pair][1])
                    if t.total_seconds() <= self.dca_wait_secs:
                        return None
                else:
                    self.dca_orders[trade.pair] = [filled_buys[0].cost, datetime.datetime.now(), False]
                    self.save_dca_orders()
                    return None

                if current_profit > dca_percent:
                    return None

                if last_candle['rsi'] > self.dca_rsi:
                    return None

                # if last_candle['close'] <= previous_candle['close'] \
                #         or ((last_candle['adx'] <= previous_candle['adx'])
                #             and last_candle['adx'] < 25) and last_candle['volume'] == 0:
                #     return None

                if 0 < count_of_buys <= self.max_dca_orders:
                    try:
                        # This returns first order stake size
                        stake_amount = filled_buys[0].cost
                        # This then calculates current safety order size
                        stake_amount = stake_amount * (1 + (count_of_buys * self.dca_koef))

                        # zapis casu dle meny
                        self.dca_orders[trade.pair] = [stake_amount, datetime.datetime.now(), False]
                        # ulozit hned na disk
                        self.save_dca_orders()
                        return stake_amount
                    except:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
                        return None

        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            pass

        return None

    def confirm_buy_higher_frame(self, pair):
        try:
            dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe=self.higher_timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            return previous_candle['close'] < last_candle['close']
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            pass
        return False

    def obtain_last_prev_candles(self, pair, timeframe):
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            return last_candle, previous_candle
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            return None, None

    def load_dca_orders(self):
        try:
            if os.path.exists(f'user_data/dca_orders_{self.buy_rsi}'):
                with open(f'user_data/dca_orders_{self.buy_rsi}', 'rb') as handle:
                    self.dca_orders = pickle.load(handle)
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            pass

    def save_dca_orders(self):
        try:
            with open(f'user_data/dca_orders_{self.buy_rsi}_history', 'a') as f:
                for k in self.dca_orders.keys():
                    if not self.dca_orders[k][2]:
                        f.write(f'{self.dca_orders[k][1]}: {k} - Stake: {self.dca_orders[k][0]}\n')
                        self.dca_orders[k][2] = True
                f.close()

            with open(f'user_data/dca_orders_{self.buy_rsi}', 'wb') as handle:
                pickle.dump(self.dca_orders, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            pass

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        try:
            if sell_reason == 'stop_loss' or 'sell' in sell_reason:
                if 'force' in sell_reason or sell_reason.startswith('stop_loss'):
                    _block_year = datetime.datetime.now() + timedelta(days=365)
                    self.lock_pair(pair=pair, until=_block_year, reason=sell_reason)
                    pass
                return True
            pr = trade.calc_profit_ratio(rate)
            if pr <= 0:
                return False
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            pass

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.higher_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # self.get_recommendation(dataframe, metadata['pair'].replace('/', ''))
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
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

        if self.confirm_buy_higher_frame(metadata['pair']):
            dataframe.loc[
                (
                        (dataframe['volume'].gt(0))
                        # &
                        # (qtpylib.crossed_above(dataframe['rsi'],self.buy_rsi))
                ),
                'buy'] = 1
        else:
            dataframe.loc[
                (
                        (dataframe['volume'].gt(0))
                        # &
                        # (qtpylib.crossed_above(dataframe['rsi'],self.buy_rsi))
                ),
                'buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] >= 100) &
                    (dataframe['volume'].gt(0))
            ),
            'sell'] = 1
        return dataframe
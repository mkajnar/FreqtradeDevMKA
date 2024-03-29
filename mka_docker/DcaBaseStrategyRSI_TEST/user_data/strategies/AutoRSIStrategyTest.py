import os
import sys
import time
import datetime
from collections import defaultdict
from datetime import timedelta
from threading import Thread
from typing import Optional
from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pickle
import json

from user_data.strategies.dca_setting import dca_percent
from user_data.strategies.roi_settings import get_rois
from user_data.strategies.rsi_stat import histogram
from user_data.strategies.tsl_settings import stoploss, use_sell_signal, trailing_stop_positive, \
    trailing_stop_positive_offset, trailing_stop, trailing_only_offset_is_reached

from user_data.strategies.Decorators import safe


class AutoRSIStrategyTest(IStrategy):
    @safe
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_max_list = {}
        self.histograms = defaultdict(object)
        self.btc_rsi_hist = []
        self.rsi_min = {}
        self.rsi_max = {}
        self.reason = {}
        self.timeframe = '5m'
        self.informative_timeframes = ['15m']
        self.higher_timeframe = '15m'
        self.minimal_roi = get_rois()
        self.stoploss = stoploss
        self.use_sell_signal = use_sell_signal
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.trailing_only_offset_is_reached = trailing_only_offset_is_reached
        # begin dca section
        self.position_adjustment_enable = True
        self.dca_rsi = 40
        self.dca_rsi_history = {}
        self.dca_rsi_profit = {}
        self.dca_debug = False
        # self.dca_wait_secs = 5 * 60
        self.max_dca_orders = 3
        self.max_dca_multiplier = 5.5
        self.dca_koef = 1
        self.dca_orders = {}
        self.profits = {}
        # end dca section

        self.initial_buys = {}

        self.unfilledtimeout = {
            'buy': 60 * 3,
            'sell': 60 * 10
        }
        self.order_types = {
            'buy': 'market',
            'sell': 'market',
            'stoploss': 'market',
            'stoploss_on_exchange': False
        }
        self.plot_config = {
            # "_main_plot": {
            #     "tema": {},
            #     "sar": {
            #         "color": "white"
            #     },
            #     "sma9": {
            #         "color": "#8dca58",
            #         "type": "line"
            #     },
            #     "sma20": {
            #         "color": "#62df8c",
            #         "type": "line"
            #     }
            # },
            "subplots": {
                "RSI": {
                    "rsi": {
                        "color": "red"
                    }
                }
            }
        }

    # This is called when placing the initial order (opening trade)
    @safe
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    @safe
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

        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!

        try:
            # plneni slovniku profitu prubeznymi profity
            self.profits[trade.pair] = current_profit
            # vsechny nakup meny
            filled_buys = trade.select_filled_orders('buy')
            count_of_buys = len(filled_buys)
            last_candle, previous_candle = self.obtain_last_prev_candles(trade.pair, self.timeframe)

            if last_candle is not None and previous_candle is not None:

                open(f'RSI_Profit_{trade.pair.replace("/", "_")}_{self.get_strategy_name()}_'
                     f'{self.timeframe}.csv', mode='a').write(f'{current_time};'
                                                              f'{trade.pair};'
                                                              f'{current_profit};'
                                                              f'{last_candle["rsi"]}\n')

                if trade.pair not in self.dca_rsi_history.keys():
                    self.dca_rsi_history[trade.pair] = []
                if trade.pair not in self.dca_rsi_profit.keys():
                    self.dca_rsi_profit[trade.pair] = []

                # if trade.pair in self.dca_orders.keys():
                #     t = (datetime.datetime.now() - self.dca_orders[trade.pair]["changed"])
                #     if t.total_seconds() <= self.dca_wait_secs:
                #         return None
                # else:
                #     dca_item = {"current_rate": current_rate, "current_profit": current_profit,
                #                 "stake_amount": filled_buys[0].cost, "changed": datetime.datetime.now(), "saved": False}
                #
                #     self.dca_orders[trade.pair] = dca_item
                #     self.save_dca_orders()
                #     return None

                if not self.dca_debug:

                    if current_profit >= dca_percent:
                        return None

                    if len(self.dca_rsi_history[trade.pair]) > 0 \
                            and last_candle['rsi'] <= self.dca_rsi_history[trade.pair][-1]:
                        self.save_last_rsi(last_candle, trade)
                        return None

                    if len(self.dca_rsi_profit[trade.pair]) > 0 \
                            and current_profit <= self.dca_rsi_profit[trade.pair][-1]:
                        self.save_last_profit(current_profit, trade)
                        return None

                    self.save_last_rsi(last_candle, trade)
                    self.save_last_profit(current_profit, trade)

                    if last_candle['close'] < previous_candle['close']:
                        return None

                if 0 < count_of_buys <= self.max_dca_orders:
                    try:
                        # This returns first order stake size
                        stake_amount = filled_buys[0].cost
                        # This then calculates current safety order size
                        stake_amount = stake_amount * (1 + (count_of_buys * self.dca_koef))

                        # zapis casu dle meny

                        dca_item = {"current_rate": current_rate, "current_profit": current_profit,
                                    "stake_amount": stake_amount, "changed": datetime.datetime.now(), "saved": False}

                        self.dca_orders[trade.pair] = dca_item
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

    def save_last_profit(self, current_profit, trade):
        if current_profit not in self.dca_rsi_profit[trade.pair]:
            self.dca_rsi_profit[trade.pair].append(current_profit)

    def save_last_rsi(self, last_candle, trade):
        if last_candle["rsi"] not in self.dca_rsi_history[trade.pair]:
            self.dca_rsi_history[trade.pair].append(last_candle["rsi"])

    @safe
    def obtain_last_prev_candles(self, pair, timeframe):
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            return last_candle, previous_candle
        except:
            return None, None

    @safe
    def load_dca_orders(self):
        if os.path.exists(f'user_data/dca_orders_{self.dca_rsi}_{self.get_strategy_name()}'):
            with open(f'user_data/dca_orders_{self.dca_rsi}_{self.get_strategy_name()}', 'rb') as handle:
                self.dca_orders = pickle.load(handle)

    @safe
    def save_dca_orders(self):
        with open(f'user_data/dca_orders_{self.dca_rsi}_history_{self.get_strategy_name()}', 'a') as f:
            for k in self.dca_orders.keys():
                if not self.dca_orders[k]["saved"]:
                    f.write(
                        f'{self.dca_orders[k]["changed"]}: {k} - Stake: {self.dca_orders[k]["stake_amount"]},'
                        f' Rate: {self.dca_orders[k]["current_rate"]}, Profit: {self.dca_orders[k]["current_profit"]}\n')
                    self.dca_orders[k]["saved"] = True
            f.close()

        with open(f'user_data/dca_orders_{self.dca_rsi}_{self.get_strategy_name()}', 'wb') as handle:
            pickle.dump(self.dca_orders, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @safe
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for _ in self.informative_timeframes:
            informative_pairs.extend([(pair, _) for pair in pairs])
        return informative_pairs

    @safe
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # higher_dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.higher_timeframe)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # higher_dataframe['rsi'] = ta.RSI(higher_dataframe, timeperiod=14)
        # find min a max RSI values by quantile +/- 1%
        self.min_max_list[metadata['pair']] = [dataframe['rsi'].quantile(0.01), dataframe['rsi'].quantile(0.99)]

        if 'BTC/USDT' in metadata['pair']:
            self.btc_rsi_hist.append([dataframe['rsi'].values[-1], dataframe['rsi'].median(), dataframe['rsi'].mean(),
                                      dataframe['rsi'].mode()[0]])
            print(
                f'\n\t\t\tBTC STATS:\tRSI {self.btc_rsi_hist[-1][0]}, MEDIAN {self.btc_rsi_hist[-1][1]}, MEAN {self.btc_rsi_hist[-1][2]}, MODE {self.btc_rsi_hist[-1][3]}')
            print(
                f'\t\t\tBTC BUY/SELL: {(self.btc_rsi_hist[-1][0] + self.btc_rsi_hist[-1][1]) / 2}\n')

            # self.btc_rsi_hist_higher.append(
            #     [higher_dataframe['rsi'].values[-1], higher_dataframe['rsi'].median(), higher_dataframe['rsi'].mean(),
            #      higher_dataframe['rsi'].mode()[0]])
            # print(
            #     f'\n\t\t\tBTC STATS - HIGHER:\tRSI {self.btc_rsi_hist_higher[-1][0]}, MEDIAN {self.btc_rsi_hist_higher[-1][1]}, MEAN {self.btc_rsi_hist_higher[-1][2]}, MODE {self.btc_rsi_hist_higher[-1][3]}')
            # print(
            #     f'\t\t\tBTC BUY/SELL - HIGHER: {(self.btc_rsi_hist_higher[-1][0] + self.btc_rsi_hist_higher[-1][1]) / 2}\n')

        self.histograms[metadata['pair']] = histogram(dataframe['rsi'].values)
        # self.higher_histograms[metadata['pair']] = histogram(higher_dataframe['rsi'].values)

        print(f"Normal RSI histogram for {metadata['pair']}: {json.dumps(histogram(dataframe['rsi'].values))}")
        # print(f"Higher RSI histogram for {metadata['pair']}: {json.dumps(histogram(higher_dataframe['rsi'].values))}")

        return dataframe

    @safe
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        pr = trade.calc_profit_ratio(rate)

        if 'stop_loss' == sell_reason:
            self.block_pair(pair=pair, sell_reason=sell_reason, minutes=24 * 60)
            return True

        if 'force_exit' == sell_reason:
            self.block_pair(pair=pair, sell_reason=sell_reason, minutes=60)
            return True

        # if 'sell_signal_btc_down' == sell_reason:
        #     if ((self.btc_rsi_hist[-1][0] + self.btc_rsi_hist[-1][1]) / 2 <= 50.0) and (
        #             (self.btc_rsi_hist[-1][0] + self.btc_rsi_hist[-1][1]) / 2 <= 50.0):
        #         self.block_pair(pair=pair, sell_reason=sell_reason, minutes=10)
        #         return True

        if 'trailing_stop_loss' == sell_reason:
            # self.block_pair(pair=pair, sell_reason=sell_reason, minutes=3)
            if pr > 0.01:
                self.set_timed_initial_buy_signal(pair=pair, value=1, ts=1)
                return True

        if 'exit_signal' == sell_reason:
            if pr > 0.10:
                self.set_timed_initial_buy_signal(pair=pair, value=1, ts=1)
                return True

        if 'roi' == sell_reason:
            self.set_timed_initial_buy_signal(pair=pair, value=1, ts=1)
            return True

        return False

    def set_timed_initial_buy_signal(self, pair, value, ts):
        th = Thread(target=lambda: self.set_initial_buy(pair=pair, value=value, ts=ts))
        th.start()

    @safe
    def set_initial_buy(self, pair, value, ts):
        time.sleep(ts)
        self.initial_buys[pair] = value
        pass

    @safe
    def block_pair(self, pair, sell_reason, minutes):
        _block_year = datetime.datetime.now() + timedelta(minutes=minutes)
        self.lock_pair(pair=pair, until=_block_year, reason=sell_reason)

    @safe
    def flip_initial_buy(self, pair):
        if pair not in self.initial_buys.keys():
            self.initial_buys[pair] = 0
        result = self.initial_buys[pair]

        self.initial_buys[pair] = 0
        return result

    @safe
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if metadata['pair'] not in self.initial_buys.keys():
            self.set_timed_initial_buy_signal(pair=metadata['pair'], value=1, ts=0.1)

        self.rsi_min[metadata['pair']] = [[int(x) for x in s[0].split('-')]
                                          for s in self.histograms[metadata['pair']] if s[1] > 10][0]

        self.reason[metadata['pair']] = 'buy_signal_rsi'
        dataframe.loc[
            (
                    ((dataframe['volume'].gt(0)) &
                        # (dataframe['rsi'].gt(self.rsi_min[metadata['pair']][0])) &
                        (dataframe['rsi'].lt(self.rsi_min[metadata['pair']][1])))
                    #| (self.flip_initial_buy(metadata['pair']) == 1)
            ),
            'buy'] = 1
        return dataframe

    @safe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.rsi_max[metadata['pair']] = [[int(x) for x in s[0].split('-')]
                                          for s in self.histograms[metadata['pair']] if s[1] > 3][-1]

        self.reason[metadata['pair']] = 'sell_signal_rsi'
        dataframe.loc[
            (
                    (dataframe['volume'].gt(0)) &
                    (dataframe['rsi'].gt(self.rsi_max[metadata['pair']][0]))
                # & (dataframe['rsi'].lt(self.rsi_max[metadata['pair']][1]))
            ),
            'sell'] = 1
        return dataframe

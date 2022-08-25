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
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import IStrategy, stoploss_from_absolute


class AutoRSIStrategyVer2(IStrategy):
    @safe
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_max_list = {}
        self.histograms = defaultdict(object)
        self.btc_rsi_hist = []
        self.rsi_min = {}
        self.rsi_max = {}
        self.reason = {}
        self.timeframe = '1m'
        self.timeframe_mins = timeframe_to_minutes(self.timeframe)
        self.informative_timeframes = ['5m']
        self.higher_timeframe = '5m'
        self.minimal_roi = get_rois(self.timeframe_mins)
        self.ignore_roi_if_entry_signal = True
        self.stoploss = stoploss
        self.use_custom_stoploss = True
        self.use_sell_signal = use_sell_signal
        self.use_exit_signal = True
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
        self.dca_wait_secs = 30 * 60
        self.max_dca_orders = 3
        self.max_dca_multiplier = 5.5
        self.dca_koef = 1
        self.dca_orders = {}
        self.profits = {}
        # end dca section

        self.unfilledtimeout = {
            'buy': 30,
            'sell': 30
        }
        self.order_types = {
            'buy': 'market',
            'sell': 'market',
            'stoploss': 'market',
            'stoploss_on_exchange': False
        }

    # This is called when placing the initial order (opening trade)
    @safe
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        return proposed_stake / self.max_dca_multiplier

    @safe
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = dataframe.iloc[-1].squeeze()
        return stoploss_from_absolute(current_rate - (candle['atr'] * 300), current_rate, is_short=trade.is_short)

    #@safe
    #def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
    #                current_profit: float, **kwargs):
    #    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #    last_candle = dataframe.iloc[-1].squeeze()

        # Above 20% profit, sell when rsi < 80
    #    if current_profit > 0.2:
    #        if last_candle['rsi'] < 80:
    #            return 'rsi_below_80'

        # Between 2% and 10%, sell if EMA-long above EMA-short
    #    if 0.05 < current_profit < 0.1:
    #        if last_candle['emalong'] > last_candle['emashort']:
    #            return 'ema_long_below_80'

        # Sell any positions at a loss if they are held for more than one day.
        # if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 1:
        #     return 'unclog'

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

                if trade.pair in self.dca_orders.keys():
                    t = (datetime.datetime.now() - self.dca_orders[trade.pair]["changed"])
                    if t.total_seconds() <= self.dca_wait_secs:
                        print(f'DCA Wait for {trade.pair}')
                        return None
                else:
                    dca_item = {"current_rate": current_rate, "current_profit": current_profit,
                                "stake_amount": filled_buys[0].cost, "changed": datetime.datetime.now(), "saved": False}
                    self.dca_orders[trade.pair] = dca_item
                    self.save_dca_orders()
                    print(f'DCA Wait for {trade.pair}')
                    return None

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
                        self.block_pair(pair=trade.pair, sell_reason='dca_buy', minutes=10)
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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
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
        if "BTC/USDT" not in pairs:
            pairs.append("BTC/USDT")
        informative_pairs = []
        for _ in self.informative_timeframes:
            informative_pairs.extend([(pair, _) for pair in pairs])
        return informative_pairs

    @property
    def plot_config(self):
        """
            There are a lot of solutions how to build the dictonary, hardcoded (as one big dictonary or add the elements one by one)
            Example 1:
                plot_config = {'main_plot': {}, 'subplots': {}}

            Example 2:
                plot_config = {}
                plot_config['main_plot'] = {}
                plot_config['subplots'] = {}
                plot_config['main_plot']['sma'] = {}
                plot_config['main_plot']['ema3'] = {}
                plot_config['main_plot']['ema5'] = {'color': 'green'}
                plot_config['subplots']['Other'] = {'macd': {'color': 'red'}, 'macdsignal': {'color': 'blue'}}

            Example 3: (use functions. here we resample 4h candles to 1d and 1w).
                plot_config = {}
                plot_config['main_plot'] = {}
                plot_config['subplots'] = {}
                plot_config['main_plot']['ema50'] = {'color': 'green'}
                plot_config['main_plot']['ema200'] = {'color': 'red'}
                plot_config['subplots']['Forecast'] = {'macd': {'color': 'red'}, 'macdsignal': {'color': 'blue'}}
                # some kind of append to the section.
                plot_config['subplots']['Forecast'].update({f'resample_{timeframe_to_minutes(self.timeframe) * 6}_macd': {}, f'resample_{timeframe_to_minutes(self.timeframe) * 6}_macdsignal': {}})
                plot_config['subplots']['Forecast'].update({f'resample_{timeframe_to_minutes(self.timeframe) * 6 * 7}_macd': {}, f'resample_{timeframe_to_minutes(self.timeframe) * 6 * 7}_macdsignal': {}})
        """
        plot_config = {}
        plot_config['main_plot'] = {}
        plot_config['main_plot']['ema20'] = {'color': 'yellow'}
        plot_config['main_plot']['ema50'] = {'color': 'green'}
        plot_config['main_plot']['ema200'] = {'color': 'red'}
        plot_config['main_plot']['tema'] = {'color': 'orange'}
        plot_config['main_plot']['bb_middleband'] = {'color': 'violet'}

        plot_config['subplots'] = {}

        plot_config['subplots']['BTC_EMA'] = {
            'ema20_high_btc': {'color': 'orange'},
            'ema50_high_btc': {'color': 'violet'},
            'ema200_high_btc': {'color': 'blue'}
        }

        # plot_config['subplots']['MACD'] = {
        #     'macd': {'color': 'yellow'},
        #     'macdsignal': {'color': 'blue'},
        #     'macdhist': {'color': 'orange'}
        # }

        plot_config['subplots']['RSI'] = {"rsi": {"color": "red"}}
        plot_config['subplots']['ATR'] = {"atr": {"color": "white"}}

        return plot_config

    @safe
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # higher_dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.higher_timeframe)
        higher_dataframe = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe=self.higher_timeframe)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['emalong'] = ta.EMA(dataframe, timeperiod=20, price='close')
        dataframe['emashort'] = ta.EMA(dataframe, timeperiod=10, price='close')

        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema20_high_btc'] = ta.EMA(higher_dataframe, timeperiod=20)
        dataframe['ema50_high_btc'] = ta.EMA(higher_dataframe, timeperiod=50)
        dataframe['ema200_high_btc'] = ta.EMA(higher_dataframe, timeperiod=200)

        macd = ta.MACD(dataframe, timeperiod=14)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

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

        if 'trailing_stop_loss' == sell_reason:
            if pr > 0.01:
                self.block_pair(pair=pair, sell_reason=sell_reason, minutes=3)
                return True


        if 'exit_signal' == sell_reason:
            if pr > 0.10:
                return True

        if 'roi' == sell_reason:
            if pr > 0.01:
                return True

        return False

    @safe
    def block_pair(self, pair, sell_reason, minutes):
        _block_year = datetime.datetime.now() + timedelta(minutes=minutes)
        self.lock_pair(pair=pair, until=_block_year, reason=sell_reason)

    @safe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if 'initial_buy' not in dataframe.keys():
            dataframe['initial_buy'] = 1

        self.rsi_min[metadata['pair']] = [[int(x) for x in s[0].split('-')]
                                          for s in self.histograms[metadata['pair']] if s[1] > 20][0]

        self.reset_buy_signal(dataframe)
        self.buy_rsi(dataframe, metadata)
        self.buy_ema(dataframe, metadata)
        #self.buy_initial(dataframe, metadata)

        return dataframe

    @safe
    def reset_buy_signal(self, dataframe):
        dataframe.loc[(), ['enter_long', 'enter_tag']] = (0, 'init')

    @safe
    def tema_guard(self, dataframe):
        return (dataframe['tema'] <= dataframe['bb_middleband']) & (dataframe['tema'] > dataframe['tema'].shift(1))

    @safe
    def btc_high_guard(self, dataframe):
        return (dataframe['ema20_high_btc'] > dataframe['ema50_high_btc']) & (
                dataframe['ema50_high_btc'] > dataframe['ema200_high_btc']) & (
                       dataframe['ema20_high_btc'] > dataframe['ema20_high_btc'].shift(1))

    @safe
    def buy_rsi(self, dataframe, metadata):
        dataframe.loc[
            (
                (
                        (dataframe['volume'].gt(0)) &
                        (dataframe['rsi'].lt(40)) #&
                        #self.tema_guard(dataframe)
                )
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_signal_rsi')

    @safe
    def buy_ema(self, dataframe, metadata):
        dataframe.loc[
            (
                (
                        (dataframe['ema20'].gt(dataframe['ema50']))
                        # &
                        #(dataframe['ema50'].gt(dataframe['ema50'])) &
                        #self.btc_high_guard(dataframe)
                )
            ),
            ['enter_long', 'enter_tag']] = (1, 'ema_buy_signal')

    @safe
    def buy_initial(self, dataframe, metadata):

        dataframe.loc[
            (
                (
                        (dataframe['open'].gt(0)) &
                        (dataframe['initial_buy'].gt(0))
                    # &
                    # self.btc_high_guard(dataframe)
                )
            ),
            ['enter_long', 'enter_tag', 'initial_buy']] = (1, 'initial_buy', 0)

        print(f'Checked initial buy for {metadata["pair"]}')

    @safe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.rsi_max[metadata['pair']] = \
            [[int(x) for x in s[0].split('-')] for s in self.histograms[metadata['pair']] if s[1] > 3][-1]

        dataframe.loc[
            (
                    (dataframe['volume'].gt(0)) &
                    (dataframe['rsi'].gt(self.rsi_max[metadata['pair']][0]))
            ),
            ['exit_long', 'exit_tag']] = (1, 'sell_signal_rsi')
        return dataframe

# --- Do not remove these libs ---
import os
import sys
from datetime import timedelta  # noqa
from typing import Optional, Union  # noqa

import logging
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
import datetime
from datetime import timedelta  # noqa
from pandas import DataFrame  # noqa
from freqtrade.persistence import Trade

from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter)

logger = logging.getLogger('freqtrade')


def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error('{} - {}'.format(exc_type, exc_tb.tb_lineno))
            return None

    return safe_f


class BB_Riding(IStrategy):
    INTERFACE_VERSION = 3

    rebuys = {}
    last_candles_ts = {}
    profits = {}

    buy_rebuy_koef = DecimalParameter(1, 3, decimals=2, default=1.2, space="buy")
    buy_dca_percent = DecimalParameter(-0.10, -0.01, decimals=2, default=-0.01, space="buy")
    buy_rebuy_divider = DecimalParameter(1, 10, decimals=1, default=2, space="buy")
    stoploss = -0.5
    # timeframe = '1m'
    minimal_roi = {
        "0": 0.3
    }
    position_adjustment_enable = True
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = True

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

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
            return None, None

    # @property
    # def plot_config(self):
    #     plot_config = {
    #         "main_plot": {
    #             "bb_lower": {
    #                 "color": "red"
    #             },
    #             "bb_middle": {
    #                 "color": "yellow"
    #             },
    #             "bb_upper": {
    #                 "color": "green"
    #             }
    #         },
    #         "subplots": {
    #             "RSI": {
    #                 "rsi": {
    #                     "color": "red"
    #                 }
    #             },
    #             "MACD1": {
    #                 "macdsignal": {
    #                     "color": "blue"
    #                 }
    #             },
    #             "MACD2": {
    #                 "macd": {
    #                     "color": "green"
    #                 }
    #             },
    #             "MACD3": {
    #                 "macdhist": {
    #                     "color": "red"
    #                 }
    #             }
    #         }
    #     }
    #     return plot_config

    @safe
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        boll = ta.BBANDS(dataframe, nbdevup=2.0, nbdevdn=2.0, timeperiod=20)
        dataframe['bb_lower'] = boll['lowerband']
        dataframe['bb_middle'] = boll['middleband']
        dataframe['bb_upper'] = boll['upperband']

        dataframe["bb_width"] = (
                (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]
        )

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['rsi'] = ta.RSI(dataframe)

        # print(metadata)
        # print(dataframe[["date", "close", "bb_upper", "bb_middle", "bb_lower", "bb_width"]].tail(25))

        return dataframe

    @safe
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.more_conditions():
            dataframe.loc[
                (
                        (dataframe['close'] > dataframe['bb_upper']) &
                        (dataframe['bb_width'] > 0.02) &
                        (dataframe['volume'] > 0)
                ),
                'buy'] = 1
        else:
            dataframe.loc[
                (
                    (dataframe['volume'] > 0)
                ),
                'buy'] = 1
        return dataframe

    @safe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['macdsignal'] < 0)
            ),
            'sell'] = 1
        return dataframe

    @safe
    def more_conditions(self):
        if len(self.profits.keys()) > 0:
            reds = len([s for s in self.profits if self.profits[s] < 0])
            greens = len([s for s in self.profits if self.profits[s] > 0])
            total = len(self.profits.keys())
            pct_greens = greens / total
            pct_reds = reds / total
            return pct_greens < pct_reds
        return False

    @safe
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        last_candle, previous_candle = self.obtain_last_prev_candles(trade.pair, self.timeframe)

        # plneni seznamu aktualnich profitu pro podminky nakupu
        self.profits[trade.pair] = current_profit

        # ratio = self.more_conditions()

        if trade.pair in self.last_candles_ts.keys():
            if self.last_candles_ts[trade.pair] == last_candle.values[0]:
                logger.info(f'REBUY NONE: {trade.pair}, REASON: close candle is the same')
                return None

        # aby se neresila stejna svice vickrat
        self.last_candles_ts[trade.pair] = last_candle.values[0]

        if last_candle['close'] < previous_candle['close']:
            logger.info(f'REBUY NONE: {trade.pair}, REASON: candles not confirmed')
            return None

        if trade.pair in self.rebuys.keys():
            if trade.orders[-1].side == trade.exit_side:
                p = self.rebuys[trade.pair]
                if p is not None:
                    logger.info(f'REBUY: {trade.pair} with {p}')
                    self.rebuys[trade.pair] = None
                    return p
                else:
                    return None
            else:
                self.rebuys[trade.pair] = None
                return None

        if current_profit > 0.05 and trade.nr_of_successful_exits == 0:
            logger.info(f'REBUY NONE: {trade.pair}, REASON: scalping 5%')
            return -(trade.stake_amount / 2)

        if current_profit > self.buy_dca_percent.value:
            logger.info(f'REBUY NONE: {trade.pair}, REASON: profit is bigger than DCA {self.buy_dca_percent.value}')
            return None

        p = -(trade.stake_amount / self.buy_rebuy_divider.value)
        self.rebuys[trade.pair] = abs(p * self.buy_rebuy_koef.value)
        logger.info(f'REBUY PLANNED: {trade.pair} with stake amount {p}')
        return p

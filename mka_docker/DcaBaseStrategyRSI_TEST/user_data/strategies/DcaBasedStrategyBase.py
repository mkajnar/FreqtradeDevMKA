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


class DcaBasedStrategyBase(IStrategy):

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pass

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pass

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pass

    def __init__(self, config: dict):
        super().__init__(config)
        self.use_sell_signal = True
        self.trailing_stop = True
        self.trailing_stop_positive = 0.010
        self.trailing_stop_positive_offset = 0.015
        self.trailing_only_offset_is_reached = True
        pass

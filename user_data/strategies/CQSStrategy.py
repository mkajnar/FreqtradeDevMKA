import logging
import json
from datetime import datetime
from typing import Optional

import numpy as np  # noqa
import pandas as pd  # noqa
import requests
from pandas import DataFrame
import os.path

from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


class CQSStrategy(IStrategy):
    logger = logging.getLogger(__name__)

    cqs_multiplier_loop = 8
    cqs_current_loop_number = 8
    cqs_json_file = ''

    cqs_trades = []

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "10080": 0.005,  # dopo 7 giorni va chiuso se in guadagno
        "4320": 0.15,  # dopo 3 giorni
        "0": 100
    }

    position_adjustment_enable = True
    max_entry_position_adjustment = 1

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.20

    use_custom_stoploss = True

    # Trailing stoploss
    #    trailing_stop = True
    #    trailing_stop_positive = 0.02
    #    trailing_stop_positive_offset = 0.0
    #    trailing_only_offset_is_reached = True
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 1

    # Optional order type mapping.
    order_types = {
        # 'entry': 'limit',
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    entry_pricing = {
        'price_side': 'other'
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def version(self) -> str:
        """
        Returns version of the strategy.
        """
        return "0.1"

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """

        self.cqs_json_file = str(self.config['user_data_dir']) + '/database/cqs.json'

        if self.config['runmode'].value in ('live', 'dry_run'):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods

            # cerca se esiste il file cqs.json
            file_exists = os.path.exists(self.cqs_json_file)
            if not file_exists:
                # se non esiste: inizializzo
                self.save_cqs_trade()

            # se esiste parsifico il file
            if file_exists:
                with open(self.cqs_json_file) as json_file:
                    self.cqs_trades = json.load(json_file)

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if self.config['runmode'].value in ('live', 'dry_run'):
            self.logger.debug("called bot_loop_start")

        # call remote service every
        if self.cqs_current_loop_number % self.cqs_multiplier_loop == 0:
            self.logger.debug("Call CQS Service %s", self.cqs_current_loop_number)
            self.cqs_current_loop_number = 0

            remote_data = requests.get('https://api.cryptoqualitysignals.com/v1/getSignal/?api_key=FREE&interval=15')
            # remote_data = requests.get(
            #    'https://44dbadcb-bff3-4063-9910-6609c617b9d8.mock.pstmn.io/v1/getSignal/?api_key=FREE&interval=15')

            data = json.loads(remote_data.text)
            self.logger.info(f"Called service message: '{data['message']}' count: '{data['count']}'")

            for signal in data['signals']:
                self.logger.info(f"id '{signal['id']}' coin: '{signal['coin']}' currency: '{signal['currency']}' ")
                # normalizzo la currency su USDT
                currency = signal['currency']
                if currency == 'BUSD' or currency == 'USD':
                    currency = 'USDT'

                pair = signal['coin'] + '/' + currency
                # verifico che sia nella lista delle pair trattate
                has_to_add = False
                for available_pair in self.dp.available_pairs:
                    if pair == available_pair[0]:
                        # e' nella lista dei pair trattati
                        has_to_add = True
                        # ora verifico di non avere ancora questo trade in lista
                        for trade in self.cqs_trades:
                            if trade['id'] == signal['id']:
                                has_to_add = False

                if has_to_add:
                    signal['pair'] = pair
                    self.cqs_trades.append(signal)

            self.save_cqs_trade()

        self.cqs_current_loop_number = self.cqs_current_loop_number + 1

        ## TODO cancellare i trade che dopo 15 minuti non sono stati aperti

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        self.logger.debug("called populate_indicators %s", metadata)

        cqstrade = self.get_cqs_trade_by_pair(metadata['pair'])

        # e' vuoto se: cqstrade == {}
        if cqstrade == {}:
            return dataframe

        dataframe['buy_start'] = float(cqstrade['buy_start'])
        dataframe['buy_end'] = float(cqstrade['buy_end'])
        dataframe['target1'] = float(cqstrade['target1'])
        dataframe['target2'] = float(cqstrade['target2'])
        dataframe['target3'] = float(cqstrade['target3'])
        dataframe['stop_loss'] = float(cqstrade['stop_loss'])

        # Prendi esempi da https://raw.githubusercontent.com/freqtrade/freqtrade/develop/freqtrade/templates/sample_strategy.py
        if 'target_reach' not in cqstrade:
            cqstrade['target_reach'] = 0

        high = dataframe.iloc[-1]['high']
        if high > float(cqstrade['target1']) and cqstrade['target_reach'] < 1:
            cqstrade['target_reach'] = 1
            self.save_cqs_trade()
            self.logger("RAGGIUNTO target_reach: %s per %s", cqstrade['target_reach'], metadata['pair'])
        if high > float(cqstrade['target2']) and cqstrade['target_reach'] < 2:
            cqstrade['target_reach'] = 2
            self.save_cqs_trade()
            self.logger("RAGGIUNTO target_reach: %s per %s", cqstrade['target_reach'], metadata['pair'])
        if high > float(cqstrade['target3']) and cqstrade['target_reach'] < 3:
            cqstrade['target_reach'] = 3
            self.save_cqs_trade()
            self.logger("RAGGIUNTO target_reach: %s per %s", cqstrade['target_reach'], metadata['pair'])

        dataframe['target_reach'] = cqstrade['target_reach']

        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        cqstrade = self.get_cqs_trade_by_pair(metadata['pair'])
        # e' vuoto se: cqstrade == {}
        if cqstrade == {}:
            # Deactivated enter long signal to allow the strategy to work correctly
            dataframe.loc[:, 'enter_long'] = 0

        if cqstrade != {}:
            dataframe.loc[
                (
                        (dataframe['best_ask'] <= dataframe['buy_end']) &
                        (dataframe['best_ask'] >= dataframe['buy_start'])
                ),
                ['enter_long', 'enter_tag']] = (1, cqstrade['id'])

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        cqstrade = self.get_cqs_trade_by_pair(metadata['pair'])
        # e' vuoto se: cqstrade == {}
        if cqstrade == {}:
            # Deactivated exit long signal to allow the strategy to work correctly
            dataframe.loc[:, 'exit_long'] = 0

        if cqstrade != {}:
            dataframe.loc[
                (
                        (dataframe['best_bid'] >= dataframe['target3']) |
                        (dataframe['target_reach'] == 3)
                ),
                ['exit_long', 'exit_tag']] = (1, 'target3')

            dataframe.loc[
                (
                        (dataframe['best_bid'] <= (dataframe['target1'] * 1.005)) &
                        (dataframe['target_reach'] == 2)
                ),
                ['exit_long', 'exit_tag']] = (1, 'target1')

            dataframe.loc[
                (
                        (dataframe['best_bid'] <= (dataframe['buy_end'] * 1.005)) &
                        (dataframe['target_reach'] == 1)
                ),
                ['exit_long', 'exit_tag']] = (1, 'breakeven')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current rate
        """
        result = 1
        cqstrade = self.get_cqs_trade_by_pair(pair)
        # e' vuoto se: cqstrade == {}
        if cqstrade == {}:
            self.logger.warning("ATTENZIONE: pair non trovato nel json %s", pair)
            return result

        if trade:
            relative_sl = None
            if self.dp:
                # so we need to get analyzed_dataframe from dp
                dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
                # only use .iat[-1] in callback methods, never in "populate_*" methods.
                # see: https://www.freqtrade.io/en/latest/strategy-customization/#common-mistakes-when-developing-strategies

                # protezione se non e' riuscito a scaricare i dati
                if dataframe.size > 0:
                    last_candle = dataframe.iloc[-1].squeeze()
                    # imposto lo stop loss ufficiale
                    relative_sl = last_candle['stop_loss']
                    """ è impostato nelle strategie di exit
                    # se ho superato il target1, imposto al break even
                    if current_rate > last_candle['target1']:
                        relative_sl = last_candle['buy_end'] * 1.01
                    if current_rate > last_candle['target2']:
                        relative_sl = last_candle['target1'] * 1.01
                    """
                    # imposto protezione se ha superato il 3,5% ma non e' ancora arrivato al target1
                    if current_profit > 0.04:
                        relative_sl = last_candle['buy_end'] * 1.004

                    # logging
                    if last_candle['target_reach'] and last_candle['target_reach'] >= 1:
                        self.logger.info(
                            "%s target_reach: %s current_profit: %s current_rate: %s target_1: %s target_2: %s target_3: %s"
                            , pair, last_candle['target_reach'], current_profit, current_rate,
                            last_candle['target1'], last_candle['target2'], last_candle['target3'])

            if relative_sl is not None:
                # print("custom_stoploss().relative_sl: {}".format(relative_sl))
                # calculate new_stoploss relative to current_rate
                new_stoploss = (current_rate - relative_sl) / current_rate
                # turn into relative negative offset required by `custom_stoploss` return implementation
                result = - new_stoploss

        return result

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """

        cqstrade = self.get_cqs_trade_by_pair(trade.pair)
        if cqstrade == {}:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        buy_start = float(cqstrade['buy_start'])
        buy_end = float(cqstrade['buy_end'])

        # massimo entrata dividendo in tre parti
        third_entry = ((buy_end - buy_start) / 3) + buy_start
        if buy_start < current_rate < third_entry and count_of_entries == 1:
            stake_amount = filled_entries[0].cost / 2
            if stake_amount < min_stake:
                stake_amount = min_stake

            return stake_amount

        # TODO inserire le exit parziali impostando uno stake amount negativo al raggiungimento dei target
        #  ATTENZIONE
        #  al min_stake, si potrebbe fare che di default imposto uno stake che è almeno 3 volte il minimo utilizzando
        #  il metodo custom_stake_amount

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """

        # TODO verificare che non venga chiamato con le partial exit
        self.logger.warning("NON DEVE ESSERE CHIAMATO PER LE EXIT PARZIALI: reason: %s", exit_reason)

        # rimuovere in cqs i trade conclusi
        self.remove_cqs_trade_by_pair(pair)

        return True

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        cqstrade = self.get_cqs_trade_by_pair(pair)
        if cqstrade == {}:
            return False

        # Protezione contro ingresso sopra il buy_end
        if rate > float(cqstrade['buy_end']):
            return False

        return True

    def save_cqs_trade(self):
        with open(self.cqs_json_file, 'w') as output_file:
            json.dump(self.cqs_trades, output_file, indent=4)

    def get_cqs_trade_by_pair(self, pair: str) -> dict:
        trade = {}
        for cqstrade in self.cqs_trades:
            if cqstrade['pair'] == pair:
                trade = cqstrade

        return trade

    def remove_cqs_trade_by_pair(self, pair: str):
        trade = {}
        for cqstrade in self.cqs_trades:
            if cqstrade['pair'] == pair:
                self.cqs_trades.remove(cqstrade)
                self.save_cqs_trade()
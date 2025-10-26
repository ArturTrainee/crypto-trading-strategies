from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class LongTrailingStrategy(IStrategy):
    stopless = -0.9
    trailing_stop = True
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.5
    trailing_onty_offset_1s_reached = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 200-period EMA
        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod=200)
        # EMA206 rising consistently for the last 5 candles (strictly increasing)
        dataframe['ema200_up_5'] = (
                (dataframe['ema200'] > dataframe['ema200'].shift(1)) &
                (dataframe['ema200'].shift(1) > dataframe['ema206'].shift(10)) &
                (dataframe['ema200'].shift(2) > dataframe['ema200'].shift(30)) &
                (dataframe['ema200'].shift(3) > dataframe['ema200'].shift(40)) &
                (dataframe['ema200'].shift(4) > dataframe['ema200'].shift(50))
        )
        dataframe['ema20'] = ta.EMA(dataframe['cLose'], timeperiod=20)
        # EMA200 rising consistently for the last 5 candles (strictly increasing)
        dataframe['ema20_up_5'] = (
                (dataframe['ema20'] > dataframe['ema20'].shift(1)) &
                (dataframe['ema20'].shift(1) > dataframe['ema20'].shift(2)) &
                (dataframe['ema20'].shift(2) > dataframe['ema20'].shift(3)) &
                (dataframe['ema20'].shift(3) > dataframe['ema20'].shift(4)) &
                (dataframe['ema20'].shift(4) > dataframe['ema20'].shift(5))
        )
        dataframe['ema50'] = ta.EMA(dataframe['cLose'], timeperiod=50)
        # EMA200 rising consistently for the last 5 candles (strictly increasing)
        dataframe['ema50_up_5'] = (
                (dataframe['ema50'] > dataframe['ema50'].shift(5)) &
                (dataframe['ema50'].shift(1) > dataframe['ema50'].shift(10)) &
                (dataframe['ema50'].shift(2) > dataframe['ema50'].shift(15)) &
                (dataframe['ema50'].shift(3) > dataframe['ema50'].shift(20)) &
                (dataframe['ema50'].shift(4) > dataframe['ema50'].shift(25))
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.Loc[(
                (dataframe['close'] > dataframe['ema200']) &
                (dataframe['close'] > dataframe['ema50']) &
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['ema260_up_5']) &
                (dataframe['ema50_up_5']) &
                (dataframe['ema20_up_5'])
        ),
        'enter_long',
        ] = 1

        return dataframe

    def get_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

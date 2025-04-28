from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

"""
Base class for all trading strategies.

"""
class TradingStrategy(ABC):
    """
    Base class for all trading strategies.
    Implement generate_signals() to return a pd.Series of {-1, 0, 1} signals.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pass

    def __repr__(self):
        return f"<Strategy: {self.name}>"


class SMACrossoverStrategy(TradingStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__('SMA Crossover')
        self.short, self.long = short_window, long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        smo = df['Price'].rolling(self.short).mean()
        lmo = df['Price'].rolling(self.long).mean()
        signal = np.where(smo > lmo, 1, -1)
        return pd.Series(signal, index=df.index).shift(1)


class RSIStrategy(TradingStrategy):
    def __init__(self, low: float = 30, high: float = 70):
        super().__init__('RSI')
        self.low, self.high = low, high

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = df['RSI']
        signal = np.where(rsi < self.low, 1,
                          np.where(rsi > self.high, -1, 0))
        return pd.Series(signal, index=df.index).shift(1)


class BollingerStrategy(TradingStrategy):
    def __init__(self):
        super().__init__('Bollinger Bands')

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        price = df['Price']
        upper, lower = df['BB_UPPER'], df['BB_LOWER']
        signal = np.where(price < lower, 1,
                          np.where(price > upper, -1, 0))
        return pd.Series(signal, index=df.index).shift(1)


class CompositeStrategy(TradingStrategy):
    def __init__(self, strategies: list[TradingStrategy]):
        super().__init__('Composite')
        self.strategies = strategies

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.DataFrame({
            strat.name: strat.generate_signals(df)
            for strat in self.strategies
        })
        avg = signals.mean(axis=1)
        combo = np.where(avg > 0, 1, np.where(avg < 0, -1, 0))
        return pd.Series(combo, index=df.index)
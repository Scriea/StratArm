

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class TradingStrategy(ABC):
    """Base class for all trading strategies."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pass

    def __repr__(self):
        return f"<Strategy: {self.name}>"

class SMACrossoverStrategy(TradingStrategy):
    """Enhanced with EMA and trend filter"""
    def __init__(self, short_window: int = 20, long_window: int = 50, trend_window: int = 200):
        super().__init__('EMA Crossover with Trend Filter')
        self.short = short_window
        self.long = long_window
        self.trend_window = trend_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        price = df['Price']
        short_ema = price.ewm(span=self.short, adjust=False).mean()
        long_ema = price.ewm(span=self.long, adjust=False).mean()
        trend_ema = price.ewm(span=self.trend_window, adjust=False).mean()
        
        # Generate crossover signals with trend confirmation
        long_cond = (short_ema > long_ema) & (price > trend_ema)
        short_cond = (short_ema < long_ema) & (price < trend_ema)
        
        signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        return pd.Series(signals, index=df.index).shift(1)

class RSIStrategy(TradingStrategy):
    """Enhanced with smoothed RSI and crossover detection"""
    def __init__(self, rsi_window: int = 14, low: float = 30, high: float = 70, smooth: int = 3):
        super().__init__('Smoothed RSI with Crossover')
        self.rsi_window = rsi_window
        self.low = low
        self.high = high
        self.smooth = smooth

    def calculate_rsi(self, prices):
        delta = prices.diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(self.rsi_window).mean()
        avg_loss = loss.rolling(self.rsi_window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self.calculate_rsi(df['Price'])
        smoothed_rsi = rsi.rolling(self.smooth).mean()
        
        buy_signal = (smoothed_rsi > self.low) & (smoothed_rsi.shift(1) <= self.low)
        sell_signal = (smoothed_rsi < self.high) & (smoothed_rsi.shift(1) >= self.high)
        
        signals = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
        return pd.Series(signals, index=df.index).shift(1)

class BollingerStrategy(TradingStrategy):
    """Enhanced with band crossover detection and dynamic bands"""
    def __init__(self, window: int = 20, num_std: float = 2):
        super().__init__('Bollinger Band Crossover')
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        price = df['Price']
        rolling_mean = price.rolling(self.window).mean()
        rolling_std = price.rolling(self.window).std()
        
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        
        # Crossover signals
        buy_signal = (price > lower_band) & (price.shift(1) <= lower_band)
        sell_signal = (price < upper_band) & (price.shift(1) >= upper_band)
        
        signals = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
        return pd.Series(signals, index=df.index).shift(1)

class CompositeStrategy(TradingStrategy):
    """Enhanced with majority voting system"""
    def __init__(self, strategies: list[TradingStrategy]):
        super().__init__('Composite Majority Vote')
        self.strategies = strategies

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.DataFrame({
            strat.name: strat.generate_signals(df)
            for strat in self.strategies
        })
        # Majority vote based on sign of summed signals
        combined = signals.sum(axis=1)
        return pd.Series(np.sign(combined).astype(int), index=df.index)
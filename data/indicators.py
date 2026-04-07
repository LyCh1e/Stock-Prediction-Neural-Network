"""
Single-responsibility module for computing technical indicators on OHLCV data.

All indicator logic is isolated here so it can be tested, swapped, or extended
independently of the data-fetching or ML layers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Stateless helper that adds technical indicator columns to an OHLCV DataFrame.

    All methods are static — no instance state is needed.
    """

    @staticmethod
    def calculate(df: pd.DataFrame, min_window: int = 5) -> pd.DataFrame:
        """
        Compute and attach all technical indicator columns to *df* in-place.

        Parameters
        ----------
        df         : DataFrame with columns open, high, low, close, volume
        min_window : minimum rolling window size (auto-scales for short data)

        Returns the same DataFrame with added indicator columns.
        """
        data_length = len(df)

        sma_short, sma_long, rsi_period, vol_period = TechnicalIndicators._window_sizes(
            data_length, min_window
        )

        TechnicalIndicators._add_moving_averages(df, sma_short, sma_long)
        TechnicalIndicators._add_rsi(df, rsi_period)
        TechnicalIndicators._add_bollinger_bands(df, sma_short)
        TechnicalIndicators._add_volume_indicators(df, vol_period)
        TechnicalIndicators._add_volatility(df, vol_period)
        TechnicalIndicators._add_momentum(df, data_length)
        TechnicalIndicators._add_macd(df, data_length)

        # Final fill for any remaining NaN values
        df["rsi"].fillna(50, inplace=True)
        df["volume_ratio"].fillna(1.0, inplace=True)
        df["volatility"].fillna(0.01, inplace=True)

        return df

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _window_sizes(data_length: int, min_window: int):
        if min_window is not None:
            sma_short  = min_window
            sma_long   = min(min_window * 2, data_length - 1)
            rsi_period = min(14, data_length - 1)
            vol_period = min_window
        elif data_length < 20:
            sma_short  = min(5, data_length - 1)
            sma_long   = min(10, data_length - 1)
            rsi_period = min(7, data_length - 1)
            vol_period = sma_short
        elif data_length < 50:
            sma_short, sma_long, rsi_period, vol_period = 10, 20, 14, 10
        else:
            sma_short, sma_long, rsi_period, vol_period = 20, 50, 14, 20

        sma_short  = max(2, sma_short)
        sma_long   = max(sma_short + 1, sma_long)
        rsi_period = max(2, rsi_period)
        vol_period = max(2, vol_period)
        return sma_short, sma_long, rsi_period, vol_period

    @staticmethod
    def _add_moving_averages(df: pd.DataFrame, sma_short: int, sma_long: int) -> None:
        df["sma_20"] = df["close"].rolling(window=sma_short, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(window=sma_long,  min_periods=1).mean()

    @staticmethod
    def _add_rsi(df: pd.DataFrame, rsi_period: int) -> None:
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(window=rsi_period, min_periods=1).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        rs    = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"].fillna(50, inplace=True)

    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame, sma_short: int) -> None:
        df["bb_middle"] = df["close"].rolling(window=sma_short, min_periods=1).mean()
        bb_std          = df["close"].rolling(window=sma_short, min_periods=1).std()
        df["bb_upper"]  = df["bb_middle"] + bb_std * 2
        df["bb_lower"]  = df["bb_middle"] - bb_std * 2

    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame, vol_period: int) -> None:
        df["volume_sma"]   = df["volume"].rolling(window=vol_period, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-8)
        df["volume_ratio"].fillna(1.0, inplace=True)

    @staticmethod
    def _add_volatility(df: pd.DataFrame, vol_period: int) -> None:
        df["daily_return"] = df["close"].pct_change()
        df["volatility"]   = df["daily_return"].rolling(window=vol_period, min_periods=1).std()
        df["volatility"].fillna(0.01, inplace=True)

    @staticmethod
    def _add_momentum(df: pd.DataFrame, data_length: int) -> None:
        momentum_period = min(10, data_length - 1)
        df["momentum"] = df["close"] - df["close"].shift(momentum_period)
        df["momentum"].fillna(0, inplace=True)

    @staticmethod
    def _add_macd(df: pd.DataFrame, data_length: int) -> None:
        ema_short  = min(12, data_length - 1)
        ema_long   = min(26, data_length - 1)
        ema_signal = min(9,  data_length - 1)
        exp1 = df["close"].ewm(span=ema_short,  adjust=False, min_periods=1).mean()
        exp2 = df["close"].ewm(span=ema_long,   adjust=False, min_periods=1).mean()
        df["macd"]        = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=ema_signal, adjust=False, min_periods=1).mean()

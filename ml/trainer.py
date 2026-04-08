# Responsible for data preparation and the training loop.
# Turns a raw OHLCV DataFrame into normalised sequences and runs gradient descent on a NeuralNetwork.

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.indicators import TechnicalIndicators
from ml.network import NeuralNetwork

_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "sma_20", "rsi", "volume_ratio", "volatility",
    "macd", "macd_signal", "momentum",
]


# Stateless helper — prepares sequences and runs training epochs; no state of its own.
class ModelTrainer:

    # ------------------------------------------------------------------ #
    #  Data preparation                                                   #
    # ------------------------------------------------------------------ #

    # Normalise df and build sliding-window sequences; returns X, y, and per-feature scaler_params.
    def prepare_data(
        self,
        df: pd.DataFrame,
        lookback_window: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if "sma_20" not in df.columns or df["sma_20"].isna().all():
            df = TechnicalIndicators.calculate(df.copy(), min_window=lookback_window)

        df_feat = df[_FEATURE_COLS].copy().dropna()

        if len(df_feat) < lookback_window + 1:
            raise ValueError(
                f"Insufficient data. Need at least {lookback_window + 1} days, "
                f"got {len(df_feat)}"
            )

        normalized, scaler_params = self._normalise(df_feat)
        X, y = self._make_sequences(normalized, lookback_window)
        return X, y, scaler_params

    # Convert normalised OHLCV predictions back to original price scale using stored mean/std.
    def denormalize(
        self,
        predictions_norm: np.ndarray,
        scaler_params: Dict,
    ) -> np.ndarray:
        output_cols = ["open", "high", "low", "close", "volume"]
        result = np.zeros_like(predictions_norm)
        for i, col in enumerate(output_cols):
            if col in scaler_params:
                mean = scaler_params[col]["mean"]
                std  = scaler_params[col]["std"]
                result[:, i] = predictions_norm[:, i] * std + mean
        return result

    # ------------------------------------------------------------------ #
    #  Training loop                                                      #
    # ------------------------------------------------------------------ #

    # Run epochs forward+backward passes on network, logging loss every 20 epochs.
    def train(
        self,
        network: NeuralNetwork,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
    ) -> None:
        for epoch in range(epochs):
            loss = network.train_step(X, y)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # ------------------------------------------------------------------ #
    #  Recent-data extraction for adaptive updates                        #
    # ------------------------------------------------------------------ #

    # Return the last n sequences from df using existing scaler_params, for incremental updates.
    def recent_sequences(
        self,
        df: pd.DataFrame,
        lookback_window: int,
        scaler_params: Dict,
        n: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if "sma_20" not in df.columns or df["sma_20"].isna().all():
            df = TechnicalIndicators.calculate(df.copy(), min_window=lookback_window)

        df_feat = df[_FEATURE_COLS].copy().dropna()
        normalized = self._normalise_with_params(df_feat, scaler_params)
        X, y = self._make_sequences(normalized, lookback_window)
        if len(X) > n:
            return X[-n:], y[-n:]
        return X, y

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    # Z-score normalise all feature columns; return the stacked array and per-column mean/std.
    @staticmethod
    def _normalise(df_feat: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        cols         = list(df_feat.columns)
        normalized   = []
        scaler_params: Dict = {}
        for col in cols:
            values = np.asarray(df_feat[col].values).reshape(-1, 1)
            mean   = float(np.mean(values))
            std    = float(np.std(values)) + 1e-8
            scaler_params[col] = {"mean": mean, "std": std}
            normalized.append((values - mean) / std)
        return np.hstack(normalized), scaler_params

    # Normalise df_feat using pre-computed scaler_params (for inference/adaptive updates).
    @staticmethod
    def _normalise_with_params(df_feat: pd.DataFrame, scaler_params: Dict) -> np.ndarray:
        normalized = []
        for col in df_feat.columns:
            values = np.asarray(df_feat[col].values).reshape(-1, 1)
            if col in scaler_params:
                mean = scaler_params[col]["mean"]
                std  = scaler_params[col]["std"]
                normalized.append((values - mean) / std)
            else:
                normalized.append((values - np.mean(values)) / (np.std(values) + 1e-8))
        return np.hstack(normalized)

    # Slide a window over combined to produce X (flattened windows) and y (next-day OHLCV).
    @staticmethod
    def _make_sequences(
        combined: np.ndarray, lookback_window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(combined) - lookback_window):
            X.append(combined[i : i + lookback_window].flatten())
            y.append(combined[i + lookback_window, :5])
        return np.array(X), np.array(y)

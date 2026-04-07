"""
src/ml/trainer.py
~~~~~~~~~~~~~~~~~
Responsible for data preparation and the training loop.

Single Responsibility: turn a raw OHLCV DataFrame into normalised
training sequences and run the gradient-descent loop on a NeuralNetwork.
All feature engineering and normalisation lives here; the NeuralNetwork
class knows nothing about features or scaling.
"""

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


class ModelTrainer:
    """
    Stateless helper that prepares data and runs training epochs.

    All state (weights, scaler_params) lives in the NeuralNetwork and the
    returned scaler_params dict — not in this object.
    """

    # ------------------------------------------------------------------ #
    #  Data preparation                                                   #
    # ------------------------------------------------------------------ #

    def prepare_data(
        self,
        df: pd.DataFrame,
        lookback_window: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Normalise *df* and create sliding-window sequences.

        Returns
        -------
        X            : shape (n_samples, lookback_window * n_features)
        y            : shape (n_samples, 5)  — next-day OHLCV
        scaler_params: per-feature mean/std used for normalisation
        """
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

    def denormalize(
        self,
        predictions_norm: np.ndarray,
        scaler_params: Dict,
    ) -> np.ndarray:
        """Convert normalised OHLCV predictions back to original price scale."""
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

    def train(
        self,
        network: NeuralNetwork,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
    ) -> None:
        """Run *epochs* forward+backward passes on the given network."""
        for epoch in range(epochs):
            loss = network.train_step(X, y)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # ------------------------------------------------------------------ #
    #  Recent-data extraction for adaptive updates                        #
    # ------------------------------------------------------------------ #

    def recent_sequences(
        self,
        df: pd.DataFrame,
        lookback_window: int,
        scaler_params: Dict,
        n: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the last *n* training sequences using pre-computed scaler_params.
        Used for incremental / adaptive updates.
        """
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

    @staticmethod
    def _make_sequences(
        combined: np.ndarray, lookback_window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(combined) - lookback_window):
            X.append(combined[i : i + lookback_window].flatten())
            y.append(combined[i + lookback_window, :5])
        return np.array(X), np.array(y)

"""
src/services/trading_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestrates data fetching, model training, and prediction for one symbol.

Single Responsibility: coordinate the data / ML layers to produce a trained
model and predictions. Does not persist anything or manage the stock registry.

Depends on abstractions (IDataFetcher) rather than concretions — satisfying
the Dependency Inversion Principle.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

from core.interfaces import IDataFetcher
from ml.network import NeuralNetwork
from ml.predictor import StockPredictor
from ml.trainer import ModelTrainer


class StockTradingService:
    """
    Coordinates data fetching, training, prediction, and adaptive updates.

    Parameters
    ----------
    fetcher        : Any IDataFetcher implementation (e.g. YahooFinanceFetcher)
    trainer        : ModelTrainer instance
    predictor      : StockPredictor instance
    lookback_window: Number of historical days used as one input sequence
    """

    def __init__(
        self,
        fetcher: IDataFetcher,
        trainer: ModelTrainer,
        predictor: StockPredictor,
        lookback_window: int = 10,
    ) -> None:
        self._fetcher        = fetcher
        self._trainer        = trainer
        self._predictor      = predictor
        self.lookback_window = max(3, lookback_window)

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def train(
        self,
        symbol: str,
        network: NeuralNetwork,
        epochs: int = 200,
        lookback_window: Optional[int] = None,
    ) -> Tuple[int, pd.DataFrame, Dict]:
        """
        Fetch data, prepare sequences, train *network*, return results.

        Parameters
        ----------
        lookback_window : override the service default; use the per-stock value
                          so the sequence shape always matches the network's
                          input_size (network.input_size == lookback * 12).

        Returns
        -------
        data_points   : number of OHLCV rows fetched
        raw_df        : the full fetched DataFrame (for charting / scoring)
        scaler_params : normalisation parameters (needed for inference)
        """
        lookback   = lookback_window if lookback_window is not None else self.lookback_window
        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        df         = self._fetcher.fetch_stock_data(symbol, start_date, end_date)

        print(f"Retrieved {len(df)} days of data for {symbol}")

        X, y, scaler_params = self._trainer.prepare_data(df, lookback)
        print(f"Training on {len(X)} sequences with {X.shape[1]} features")

        self._trainer.train(network, X, y, epochs=epochs)

        predictions_norm = network.predict(X)
        predictions      = self._trainer.denormalize(predictions_norm, scaler_params)
        y_actual         = self._trainer.denormalize(y, scaler_params)

        mae_close  = float(abs(predictions[:, 3] - y_actual[:, 3]).mean())
        mape_close = float(abs((predictions[:, 3] - y_actual[:, 3]) / (y_actual[:, 3] + 1e-8)).mean() * 100)

        print(f"Training results for {symbol}:")
        print(f"  Close Price MAE:  ${mae_close:.2f}")
        print(f"  Close Price MAPE: {mape_close:.2f}%")
        print(f"  Final Loss:       {network.losses[-1]:.6f}")

        return len(df), df, scaler_params

    def predict(
        self,
        symbol: str,
        df: pd.DataFrame,
        network: NeuralNetwork,
        scaler_params: Dict,
        include_scenarios: bool = True,
        lookback_window: Optional[int] = None,
    ) -> Dict:
        """
        Generate next-day prediction using *network* and current *df*.

        The sentiment is fetched fresh for every prediction call.
        """
        lookback  = lookback_window if lookback_window is not None else self.lookback_window
        sentiment = self._fetcher.get_market_sentiment(symbol)
        result    = self._predictor.predict_next_day(
            symbol, df, network, scaler_params, sentiment,
            lookback, include_scenarios,
        )
        if include_scenarios:
            result["recommendation"] = self._fetcher.get_trading_recommendation(symbol, result)
        return result

    def adaptive_update(
        self,
        symbol: str,
        network: NeuralNetwork,
        scaler_params: Dict,
        lookback_window: Optional[int] = None,
    ) -> None:
        """
        Fetch recent data and do a single incremental learning step on the
        last few sequences — keeps the model current without full retraining.
        """
        lookback   = lookback_window if lookback_window is not None else self.lookback_window
        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=max(30, lookback * 3))).strftime("%Y-%m-%d")
        df         = self._fetcher.fetch_stock_data(symbol, start_date, end_date)

        X_recent, y_recent = self._trainer.recent_sequences(
            df, lookback, scaler_params, n=3
        )
        if len(X_recent) > 0:
            network.incremental_update(X_recent, y_recent)

    def fetch_data(self, symbol: str, days: int = 180) -> pd.DataFrame:
        """Fetch *days* of OHLCV data for *symbol*."""
        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self._fetcher.fetch_stock_data(symbol, start_date, end_date)

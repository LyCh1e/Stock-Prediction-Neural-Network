"""
src/core/interfaces.py
~~~~~~~~~~~~~~~~~~~~~~
Abstract base classes (interfaces) for the SOLID dependency-inversion layer.

All high-level modules (services, registry) depend on these abstractions.
Concrete implementations (YahooFinanceFetcher, JsonModelRepository, etc.)
implement these interfaces, making them swappable without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class IDataFetcher(ABC):
    """Responsible for fetching raw OHLCV data and market sentiment."""

    @abstractmethod
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with OHLCV + technical indicator columns."""

    @abstractmethod
    def get_market_sentiment(self, symbol: str) -> Dict:
        """Return a sentiment dict with 'sentiment', 'confidence', 'score', 'details'."""

    @abstractmethod
    def get_trading_recommendation(self, symbol: str, prediction: Dict) -> str:
        """Return a trading signal string: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL."""


class IModelRepository(ABC):
    """Responsible for persisting and restoring trained model weights."""

    @abstractmethod
    def save(self, symbol: str, model, scaler_params: Dict) -> None:
        """Persist model weights and scaler parameters for *symbol*."""

    @abstractmethod
    def load(self, symbol: str) -> Optional[Dict]:
        """
        Restore saved weights for *symbol*.

        Returns a dict with keys: W1, b1, W2, b2, input_size, scaler_params,
        timestamp, final_loss — or None if no saved data exists.
        """

    @abstractmethod
    def restore_weights(self, symbol: str, model) -> Optional[Dict]:
        """
        Load saved weights into *model* in-place.

        Returns the scaler_params dict if successful, or None if no saved data
        exists or the saved shape does not match the current model.
        """


class IHistoryRepository(ABC):
    """Responsible for persisting and restoring prediction history."""

    @abstractmethod
    def save(self, symbol: str, pred_history: List[Dict]) -> None:
        """Persist the prediction history list for *symbol*."""

    @abstractmethod
    def load(self, symbol: str) -> List[Dict]:
        """Return the prediction history list for *symbol* (empty list if none)."""


class ISymbolRepository(ABC):
    """Responsible for persisting the set of tracked symbols and their settings."""

    @abstractmethod
    def save(self, symbols: Dict[str, Dict]) -> None:
        """Persist the symbols dict (symbol → {lookback, epochs})."""

    @abstractmethod
    def load(self) -> Dict[str, Dict]:
        """Return the persisted symbols dict (empty dict if none)."""


class IScorer(ABC):
    """Responsible for computing accuracy scores from prediction history."""

    @abstractmethod
    def score(
        self,
        pred_history: list,
        raw_df: pd.DataFrame,
        current_price: float,
    ):
        """Return a ScoreResult for the given prediction history."""

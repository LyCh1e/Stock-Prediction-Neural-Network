# Abstract base classes for the dependency-inversion layer.
# High-level modules depend on these; concrete implementations are swappable without touching callers.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


# Interface for fetching raw OHLCV data and market sentiment.
class IDataFetcher(ABC):

    # Return a DataFrame with OHLCV + technical indicator columns.
    @abstractmethod
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame: ...

    # Return a sentiment dict with 'sentiment', 'confidence', 'score', 'details'.
    @abstractmethod
    def get_market_sentiment(self, symbol: str) -> Dict: ...

    # Return a trading signal string: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL.
    @abstractmethod
    def get_trading_recommendation(self, symbol: str, prediction: Dict) -> str: ...


# Interface for persisting and restoring trained model weights.
class IModelRepository(ABC):

    # Persist model weights and scaler parameters for symbol.
    @abstractmethod
    def save(self, symbol: str, model, scaler_params: Dict) -> None: ...

    # Return saved weight dict for symbol, or None if none exists.
    @abstractmethod
    def load(self, symbol: str) -> Optional[Dict]: ...

    # Load saved weights into model in-place; return scaler_params or None on shape mismatch.
    @abstractmethod
    def restore_weights(self, symbol: str, model) -> Optional[Dict]: ...


# Interface for persisting and restoring prediction history.
class IHistoryRepository(ABC):

    # Persist the prediction history list for symbol.
    @abstractmethod
    def save(self, symbol: str, pred_history: List[Dict]) -> None: ...

    # Return the prediction history list for symbol (empty list if none).
    @abstractmethod
    def load(self, symbol: str) -> List[Dict]: ...


# Interface for persisting the set of tracked symbols and their settings.
class ISymbolRepository(ABC):

    # Persist the symbols dict (symbol → {lookback, epochs}).
    @abstractmethod
    def save(self, symbols: Dict[str, Dict]) -> None: ...

    # Return the persisted symbols dict (empty dict if none).
    @abstractmethod
    def load(self) -> Dict[str, Dict]: ...



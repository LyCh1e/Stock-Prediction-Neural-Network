"""
Central in-memory registry for all tracked stocks plus background-thread workers.

Single Responsibility: manage the stock registry lifecycle (add/remove/update)
and coordinate background operations. Delegates all domain work to injected
services and repositories — it does not know about ML math or file formats.

Dependency Inversion: depends on IModelRepository, IHistoryRepository,
ISymbolRepository, and StockTradingService abstractions.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable, Dict, Optional

from core.interfaces import IHistoryRepository, IModelRepository, ISymbolRepository
from ml.network import NeuralNetwork
from scoring.scorer import score_symbol
from services.trading_service import StockTradingService
from storage.excel_exporter import ExcelExporter

StockEntry = Dict


class StockRegistry:
    """
    Manages all tracked stocks: create, read, update, delete, and the
    background threads that train / predict / adaptively update each one.

    Parameters
    ----------
    message_cb       : thread-safe callback → (msg_type, payload)
    trading_service  : StockTradingService (orchestrates data + ML)
    model_repo       : IModelRepository (weight persistence)
    history_repo     : IHistoryRepository (prediction-history persistence)
    symbol_repo      : ISymbolRepository (tracked-symbols persistence)
    excel_exporter   : ExcelExporter (Excel I/O)
    """

    def __init__(
        self,
        message_cb: Callable[[str, object], None],
        trading_service: StockTradingService,
        model_repo: IModelRepository,
        history_repo: IHistoryRepository,
        symbol_repo: ISymbolRepository,
        excel_exporter: ExcelExporter,
    ) -> None:
        self._stocks:   Dict[str, StockEntry] = {}
        self._cb        = message_cb
        self._service   = trading_service
        self._model_repo = model_repo
        self._hist_repo  = history_repo
        self._sym_repo   = symbol_repo
        self._exporter   = excel_exporter

    # ------------------------------------------------------------------ #
    #  Public registry accessors                                          #
    # ------------------------------------------------------------------ #

    @property
    def stocks(self) -> Dict[str, StockEntry]:
        return self._stocks

    def get(self, symbol: str) -> Optional[StockEntry]:
        return self._stocks.get(symbol)

    def symbols(self):
        return list(self._stocks.keys())

    def has(self, symbol: str) -> bool:
        return symbol in self._stocks

    # ------------------------------------------------------------------ #
    #  Create / Delete                                                    #
    # ------------------------------------------------------------------ #

    def add(self, symbol: str, lookback: int = 10, epochs: int = 200) -> bool:
        symbol = symbol.upper()
        if symbol in self._stocks:
            return False
        self._stocks[symbol] = self._new_entry(lookback, epochs)
        self._save_symbols()
        threading.Thread(target=self._train_thread, args=(symbol,), daemon=True).start()
        return True

    def remove(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol not in self._stocks:
            return False
        del self._stocks[symbol]
        self._save_symbols()
        return True

    def full_pred_history(self, symbol: str) -> list:
        """
        Return the merged prediction history for *symbol* from all sources,
        one entry per date, with higher-priority sources overriding lower ones:
          1. stock_predictions.xlsx  (lowest — export file)
          2. stock_models_history.csv
          3. in-memory pred_history  (highest — current session)
        Sorted oldest → newest.
        """
        symbol    = symbol.upper()
        excel     = self._exporter.load_pred_history(symbol)
        persisted = self._hist_repo.load(symbol)
        in_memory = (self._stocks.get(symbol) or {}).get("pred_history", [])

        by_date: dict = {}
        for entry in excel:
            by_date[entry["date"].date()] = entry
        for entry in persisted:
            by_date[entry["date"].date()] = entry
        for entry in in_memory:
            by_date[entry["date"].date()] = entry

        return sorted(by_date.values(), key=lambda e: e["date"])

    # ------------------------------------------------------------------ #
    #  Trigger background operations                                      #
    # ------------------------------------------------------------------ #

    def predict(self, symbol: str) -> None:
        threading.Thread(target=self._predict_thread, args=(symbol,), daemon=True).start()

    def update(self, symbol: str) -> None:
        threading.Thread(target=self._update_thread, args=(symbol,), daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Symbol persistence                                                 #
    # ------------------------------------------------------------------ #

    def load_symbols(self) -> None:
        saved = self._sym_repo.load()
        if not saved:
            return
        self._cb("log", f"Loading {len(saved)} saved symbol(s)…")
        for symbol, settings in saved.items():
            symbol = symbol.upper()
            if symbol in self._stocks:
                continue
            self._stocks[symbol] = self._new_entry(
                settings.get("lookback", 10),
                settings.get("epochs",   200),
            )
            self._cb("refresh", symbol)
            self._cb("chart",   symbol)
            threading.Thread(target=self._train_thread, args=(symbol,), daemon=True).start()
            self._cb("log", f"Queued training for {symbol} (restored from save)")

    # ------------------------------------------------------------------ #
    #  Excel export delegates                                             #
    # ------------------------------------------------------------------ #

    def update_stock_data(self) -> str:
        return self._exporter.update_stock_data(self._stocks)

    def update_predictions(self) -> str:
        return self._exporter.update_predictions(self._stocks)

    # ------------------------------------------------------------------ #
    #  Background thread workers                                          #
    # ------------------------------------------------------------------ #

    def _train_thread(self, symbol: str) -> None:
        try:
            data    = self._stocks[symbol]
            lookback = data["lookback"]
            epochs   = data["epochs"]

            network = NeuralNetwork(
                input_size=lookback * 12, hidden_size=30
            )
            data["network"] = network

            # Try to resume from saved weights
            scaler_params = self._model_repo.restore_weights(symbol, network)
            if scaler_params is not None:
                data["scaler_params"] = scaler_params
                data["pred_history"]  = self._hist_repo.load(symbol)
                self._cb("status", f"Updating {symbol} (resumed from saved model)…")
                _, raw_df, _ = self._service.train(
                    symbol, network, epochs=max(20, epochs // 5), lookback_window=lookback
                )
            else:
                self._cb("status", f"Training {symbol} from scratch…")
                _, raw_df, scaler_params = self._service.train(
                    symbol, network, epochs=epochs, lookback_window=lookback
                )
                data["scaler_params"] = scaler_params

            data["raw_df"] = raw_df
            data["status"] = "Trained"

            pred = self._service.predict(
                symbol, raw_df, network, data["scaler_params"], lookback_window=lookback
            )
            data["prediction"] = pred
            data["status"]     = "Ready"

            self._save_model(symbol)
            self._exporter.update_stock_data({symbol: data})

            self._cb("refresh", symbol)
            self._cb("chart",   symbol)
            self._cb("log",     f"✓ {symbol} trained and predicted")
            self._cb("status",  f"Ready — {symbol} complete")
        except Exception as exc:
            err = str(exc)
            if symbol in self._stocks:
                self._stocks[symbol]["status"] = f"Error: {err[:40]}"
            self._cb("refresh", symbol)
            self._cb("log",     f"✗ {symbol}: {err}")

    def _predict_thread(self, symbol: str) -> None:
        try:
            data = self._stocks[symbol]
            if not data.get("network"):
                self._cb("log", f"{symbol} not yet trained.")
                return
            self._cb("status", f"Predicting {symbol}…")
            lookback = data["lookback"]

            self._archive_prediction(data)
            raw_df = data.get("raw_df")
            if raw_df is None:
                raw_df = self._service.fetch_data(symbol)
            pred = self._service.predict(
                symbol, raw_df, data["network"], data["scaler_params"], lookback_window=lookback
            )

            data["prediction"] = pred
            data["status"]     = "Ready"

            self._save_model(symbol)
            self._cb("refresh", symbol)
            self._cb("chart",   symbol)
            self._cb("log",     f"✓ {symbol} predictions refreshed")
            self._cb("status",  "Ready")
        except Exception as exc:
            self._cb("log", f"✗ {symbol} predict error: {exc}")

    def _update_thread(self, symbol: str) -> None:
        try:
            data = self._stocks[symbol]
            if not data.get("network"):
                self._cb("log", f"{symbol} not yet trained.")
                return
            self._cb("status", f"Updating {symbol}…")
            lookback = data["lookback"]

            self._service.adaptive_update(
                symbol, data["network"], data["scaler_params"], lookback_window=lookback
            )
            self._archive_prediction(data)

            raw_df = data.get("raw_df")
            if raw_df is None:
                raw_df = self._service.fetch_data(symbol)
            pred = self._service.predict(
                symbol, raw_df, data["network"], data["scaler_params"], lookback_window=lookback
            )

            data["prediction"] = pred
            data["status"]     = "Ready"

            self._save_model(symbol)
            self._cb("refresh",     symbol)
            self._cb("chart",       symbol)
            self._cb("log",         f"✓ {symbol} adaptively updated")
            self._cb("status",      "Ready")
            self._cb("pull_status", ("ok", symbol))
        except Exception as exc:
            self._cb("pull_status", ("error", symbol))
            self._cb("log", f"✗ {symbol} update error: {exc}")

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _new_entry(lookback: int, epochs: int) -> StockEntry:
        return {
            "network":       None,
            "scaler_params": {},
            "prediction":    None,
            "raw_df":        None,
            "pred_history":  [],
            "accuracy_score": None,
            "status":        "Training…",
            "lookback":      lookback,
            "epochs":        epochs,
        }

    def _save_symbols(self) -> None:
        try:
            self._sym_repo.save(self._stocks)
            self._cb("log", "Symbols saved.")
        except Exception as exc:
            self._cb("log", f"Warning: could not save symbols: {exc}")

    def _save_model(self, symbol: str) -> None:
        data = self._stocks.get(symbol)
        if data is None:
            return
        network = data.get("network")
        if network is None:
            return
        try:
            self._model_repo.save(symbol, network, data.get("scaler_params", {}))
            self._hist_repo.save(symbol, data.get("pred_history", []))
            self._cb("log", f"{symbol}: model saved")
        except Exception as exc:
            self._cb("log", f"Warning: could not save model for {symbol}: {exc}")

    def _archive_prediction(self, data: StockEntry) -> None:
        old_pred = data.get("prediction")
        if old_pred and "scenarios" in old_pred:
            today = datetime.now().date()
            entry = {
                "date":  datetime.now(),
                "avg":   old_pred["scenarios"]["average_case"]["close"],
                "best":  old_pred["scenarios"]["best_case"]["close"],
                "worst": old_pred["scenarios"]["worst_case"]["close"],
            }
            ph = data["pred_history"]
            for i, existing in enumerate(ph):
                if existing["date"].date() == today:
                    ph[i] = entry
                    break
            else:
                ph.append(entry)
            ph = data.get("pred_history", [])
            df = data.get("raw_df")
            cp = old_pred.get("current_price", 0.0)
            if df is not None:
                data["accuracy_score"] = score_symbol(ph, df, cp)


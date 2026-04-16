# Central in-memory registry for tracked stocks plus background-thread workers.
# Manages add/remove/update lifecycle and delegates all domain work to injected services.

from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable, Dict, Optional

from core.interfaces import IHistoryRepository, IModelRepository, ISymbolRepository
from ml.network import NeuralNetwork
from scoring.calibration import apply_calibration, load_calibration
from scoring.scorer import score_symbol
from services.trading_service import StockTradingService
from storage.excel_exporter import ExcelExporter, SCORES_FILE

StockEntry = Dict


# Manages all tracked stocks and the background threads that train/predict/update each one.
class StockRegistry:

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

    # Return the StockEntry for symbol, or None if not tracked.
    def get(self, symbol: str) -> Optional[StockEntry]:
        return self._stocks.get(symbol)

    # Return a snapshot list of all currently tracked symbols.
    def symbols(self):
        return list(self._stocks.keys())

    # Return True if symbol is currently tracked.
    def has(self, symbol: str) -> bool:
        return symbol in self._stocks

    # ------------------------------------------------------------------ #
    #  Create / Delete                                                    #
    # ------------------------------------------------------------------ #

    # Add symbol to the registry, persist it, and kick off a training thread; returns False if duplicate.
    def add(self, symbol: str, lookback: int = 10, epochs: int = 200) -> bool:
        symbol = symbol.upper()
        if symbol in self._stocks:
            return False
        self._stocks[symbol] = self._new_entry(lookback, epochs)
        self._save_symbols()
        threading.Thread(target=self._train_thread, args=(symbol,), daemon=True).start()
        return True

    # Remove symbol from the registry and persist the change; returns False if not found.
    def remove(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol not in self._stocks:
            return False
        del self._stocks[symbol]
        self._save_symbols()
        return True

    # Merge prediction history from Excel, CSV, and in-memory (in-memory wins); return sorted oldest→newest.
    def full_pred_history(self, symbol: str) -> list:
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

    # Spawn a background thread to run a fresh prediction for symbol.
    def predict(self, symbol: str) -> None:
        threading.Thread(target=self._predict_thread, args=(symbol,), daemon=True).start()

    # Spawn a background thread to adaptively update the model for symbol.
    def update(self, symbol: str) -> None:
        threading.Thread(target=self._update_thread, args=(symbol,), daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Symbol persistence                                                 #
    # ------------------------------------------------------------------ #

    # Load persisted symbols from disk and queue a training thread for each one.
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

    # Delegate OHLCV Excel update to the exporter; return the file path.
    def update_stock_data(self) -> str:
        return self._exporter.update_stock_data(self._stocks)

    # Delegate predictions Excel update to the exporter; return the file path.
    def update_predictions(self) -> str:
        return self._exporter.update_predictions(self._stocks)

    # Delegate scores Excel update to the exporter, using the full merged prediction history
    # (Excel + CSV + in-memory) so all historical comparisons are included, not just today's.
    def update_scores(self) -> str:
        stocks_full: Dict[str, StockEntry] = {}
        for symbol, data in self._stocks.items():
            augmented = dict(data)
            augmented["pred_history"] = self.full_pred_history(symbol)
            stocks_full[symbol] = augmented
        return self._exporter.update_scores(stocks_full)

    # Migrate legacy score rows from stock_predictions.xlsx into prediction_score.xlsx.
    def migrate_scores(self) -> int:
        return self._exporter.migrate_scores_from_predictions()

    # ------------------------------------------------------------------ #
    #  Background thread workers                                          #
    # ------------------------------------------------------------------ #

    # Background worker: fetch data, train (or resume), predict, save, and notify UI.
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
            data["prediction"] = self._calibrate_prediction(symbol, pred)
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

    # Background worker: archive previous prediction, fetch fresh data, predict, save, notify UI.
    def _predict_thread(self, symbol: str) -> None:
        try:
            data = self._stocks[symbol]
            if not data.get("network"):
                self._cb("log", f"{symbol} not yet trained.")
                return
            self._cb("status", f"Predicting {symbol}…")
            lookback = data["lookback"]

            self._archive_prediction(data)
            raw_df = self._service.fetch_data(symbol)
            data["raw_df"] = raw_df
            pred = self._service.predict(
                symbol, raw_df, data["network"], data["scaler_params"], lookback_window=lookback
            )

            data["prediction"] = self._calibrate_prediction(symbol, pred)
            data["status"]     = "Ready"

            self._save_model(symbol)
            self._cb("refresh", symbol)
            self._cb("chart",   symbol)
            self._cb("log",     f"✓ {symbol} predictions refreshed")
            self._cb("status",  "Ready")
        except Exception as exc:
            self._cb("log", f"✗ {symbol} predict error: {exc}")

    # Background worker: adaptively update the model, archive prediction, predict, save, notify UI.
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

            raw_df = self._service.fetch_data(symbol)
            data["raw_df"] = raw_df
            pred = self._service.predict(
                symbol, raw_df, data["network"], data["scaler_params"], lookback_window=lookback
            )

            data["prediction"] = self._calibrate_prediction(symbol, pred)
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

    # Load error stats from prediction_score.xlsx and apply band calibration to pred.
    # Returns pred unchanged if there is insufficient history (< 5 matched predictions).
    def _calibrate_prediction(self, symbol: str, pred: Dict) -> Dict:
        calibration = load_calibration(symbol, SCORES_FILE)
        if calibration is None:
            return pred
        try:
            calibrated = apply_calibration(pred, calibration)
            self._cb("log", f"{symbol}: bands calibrated from {calibration['n']} historical predictions "
                            f"(in-range rate {calibration['in_range_rate']*100:.0f}%)")
            return calibrated
        except Exception as exc:
            self._cb("log", f"{symbol}: calibration skipped — {exc}")
            return pred

    # Return a blank StockEntry dict with default fields for a newly added symbol.
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

    # Persist the current tracked-symbols dict to disk via the symbol repository.
    def _save_symbols(self) -> None:
        try:
            self._sym_repo.save(self._stocks)
            self._cb("log", "Symbols saved.")
        except Exception as exc:
            self._cb("log", f"Warning: could not save symbols: {exc}")

    # Persist the network weights and prediction history for symbol to their repositories.
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

    # Move the current prediction into pred_history (upsert by date) and refresh the accuracy score.
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


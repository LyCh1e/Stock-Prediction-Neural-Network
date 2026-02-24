"""
stock_store.py
~~~~~~~~~~~~~~
Data-management layer for the Stock Price Predictor.

Responsibilities
----------------
- In-memory stock registry (add / remove / get)
- Symbol persistence  → tracked_symbols.json
- Excel export/update → stock_data.xlsx, stock_predictions.xlsx
- DataFrame helpers   → OHLCV and predictions frames
- Background thread workers (train / predict / adaptive-update)

The GUI imports StockStore and delegates all data work here, keeping
stock_gui.py focused purely on presentation.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

import pandas as pd

from stock_volume_predictor import StockTradingSystem
from stock_scorer import score_symbol, score_all, ScoreResult

# ── File name constants ──────────────────────────────────────────────────── #
STOCK_DATA_FILE = "stock_data.xlsx"
PREDICTIONS_FILE = "stock_predictions.xlsx"
SYMBOLS_FILE = "tracked_symbols.json"

# ── Type alias ───────────────────────────────────────────────────────────── #
# Each stock entry looks like:
#   {
#     "system":       StockTradingSystem | None,
#     "prediction":   dict | None,
#     "raw_df":       pd.DataFrame | None,
#     "pred_history": list[dict],
#     "status":       str,
#     "lookback":     int,
#     "epochs":       int,
#   }
StockEntry = Dict


class StockStore:
    """
    Central data store for tracked stocks.

    Parameters
    ----------
    message_cb : callable
        Callback used to post messages back to the GUI.
        Signature: ``message_cb(msg_type: str, payload)``
        msg_type values: "log" | "status" | "refresh" | "chart"
    """

    def __init__(self, message_cb: Callable[[str, object], None]) -> None:
        self._stocks: Dict[str, StockEntry] = {}
        self._cb = message_cb

    # ------------------------------------------------------------------ #
    #  Public registry accessors                                           #
    # ------------------------------------------------------------------ #

    @property
    def stocks(self) -> Dict[str, StockEntry]:
        """Read-only view of the in-memory stock registry."""
        return self._stocks

    def get(self, symbol: str) -> Optional[StockEntry]:
        return self._stocks.get(symbol)

    def symbols(self):
        return list(self._stocks.keys())

    def has(self, symbol: str) -> bool:
        return symbol in self._stocks

    # ------------------------------------------------------------------ #
    #  Create / Delete                                                     #
    # ------------------------------------------------------------------ #

    def add(self, symbol: str, lookback: int = 10, epochs: int = 200) -> bool:
        """
        Register a new stock and kick off background training.

        Returns True if the stock was added, False if it already existed.
        """
        symbol = symbol.upper()
        if symbol in self._stocks:
            return False

        self._stocks[symbol] = {
            "system":         None,
            "prediction":     None,
            "raw_df":         None,
            "pred_history":   [],
            "accuracy_score": None,   # ScoreResult, populated after predictions are archived
            "status":         "Training…",
            "lookback":       lookback,
            "epochs":         epochs,
        }
        self.save_symbols()
        threading.Thread(
            target=self._train_thread, args=(symbol,), daemon=True
        ).start()
        return True

    def remove(self, symbol: str) -> bool:
        """
        Remove a stock from the registry.

        Returns True if removed, False if it wasn't tracked.
        """
        symbol = symbol.upper()
        if symbol not in self._stocks:
            return False
        del self._stocks[symbol]
        self.save_symbols()
        return True

    # ------------------------------------------------------------------ #
    #  Trigger background operations                                       #
    # ------------------------------------------------------------------ #

    def predict(self, symbol: str) -> None:
        """Kick off a prediction refresh for *symbol* in a background thread."""
        threading.Thread(
            target=self._predict_thread, args=(symbol,), daemon=True
        ).start()

    def update(self, symbol: str) -> None:
        """Kick off an adaptive model update for *symbol* in a background thread."""
        threading.Thread(
            target=self._update_thread, args=(symbol,), daemon=True
        ).start()

    # ------------------------------------------------------------------ #
    #  Symbol persistence                                                  #
    # ------------------------------------------------------------------ #

    def save_symbols(self) -> None:
        """Persist tracked symbols and their settings to *SYMBOLS_FILE*."""
        try:
            data = {
                sym: {"lookback": info["lookback"], "epochs": info["epochs"]}
                for sym, info in self._stocks.items()
            }
            with open(SYMBOLS_FILE, "w") as fh:
                json.dump(data, fh, indent=2)
            self._cb("log", f"Symbols saved → {SYMBOLS_FILE}")
        except Exception as exc:
            self._cb("log", f"Warning: could not save symbols file: {exc}")

    def load_symbols(self) -> None:
        """
        Restore tracked symbols from *SYMBOLS_FILE* on startup.
        Each restored symbol triggers a fresh training run.
        """
        if not os.path.exists(SYMBOLS_FILE):
            return
        try:
            with open(SYMBOLS_FILE, "r") as fh:
                data: dict = json.load(fh)
            if not data:
                return
            self._cb("log", f"Loading {len(data)} saved symbol(s) from {SYMBOLS_FILE}…")
            for symbol, settings in data.items():
                symbol = symbol.upper()
                if symbol in self._stocks:
                    continue
                self._stocks[symbol] = {
                    "system":         None,
                    "prediction":     None,
                    "raw_df":         None,
                    "pred_history":   [],
                    "accuracy_score": None,
                    "status":         "Training…",
                    "lookback":       settings.get("lookback", 10),
                    "epochs":         settings.get("epochs", 200),
                }
                # Notify GUI to create the row/tab immediately
                self._cb("refresh", symbol)
                self._cb("chart",   symbol)
                threading.Thread(
                    target=self._train_thread, args=(symbol,), daemon=True
                ).start()
                self._cb("log", f"Queued training for {symbol} (restored from save)")
        except Exception as exc:
            self._cb("log", f"Warning: could not load symbols file: {exc}")

    # ------------------------------------------------------------------ #
    #  DataFrame helpers                                                   #
    # ------------------------------------------------------------------ #

    def df_for_export(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return a clean OHLCV DataFrame ready to write to Excel."""
        df = self._stocks[symbol].get("raw_df")
        if df is None:
            return None
        df_out = df[["open", "high", "low", "close", "volume"]].copy()
        df_out.columns = ["Open", "High", "Low", "Close", "Volume"]
        if hasattr(df_out.index, "tz") and df_out.index.tz is not None:
            df_out.index = df_out.index.tz_localize(None)
        df_out.index = df_out.index.date
        df_out.index.name = "Date"
        return df_out

    def pred_df_for_export(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return a predictions DataFrame (Best/Average/Worst) for *symbol*."""
        pred = self._stocks[symbol].get("prediction")
        if not pred or "scenarios" not in pred:
            return None
        sc = pred["scenarios"]
        rows = []
        for label, key in [
            ("Best Case",    "best_case"),
            ("Average Case", "average_case"),
            ("Worst Case",   "worst_case"),
        ]:
            s = sc[key]
            rows.append({
                "Exported At":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Scenario":      label,
                "Open":          round(s["open"],  2),
                "High":          round(s["high"],  2),
                "Low":           round(s["low"],   2),
                "Close":         round(s["close"], 2),
                "Profit %":      round(s["profit_potential"], 2),
                "Current Price": round(pred["current_price"], 2),
                "Confidence":    round(pred["confidence"] * 100, 1),
                "Signal":        pred.get("recommendation", "").replace("_", " "),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Excel: stock data                                                   #
    # ------------------------------------------------------------------ #

    def export_stock_data(self) -> str:
        """
        Write all OHLCV data to a fresh *STOCK_DATA_FILE*.

        Returns an absolute path on success, raises on failure.
        """
        with pd.ExcelWriter(STOCK_DATA_FILE, engine="openpyxl") as writer:
            for symbol in self._stocks:
                df_out = self.df_for_export(symbol)
                if df_out is not None:
                    df_out.to_excel(writer, sheet_name=symbol)
        return os.path.abspath(STOCK_DATA_FILE)

    def update_stock_data(self) -> str:
        """
        Append only new rows to *STOCK_DATA_FILE*, preserving history.

        Creates the file from scratch if it does not yet exist.
        Returns an absolute path on success, raises on failure.
        """
        if not os.path.exists(STOCK_DATA_FILE):
            return self.export_stock_data()

        from openpyxl import load_workbook

        wb = load_workbook(STOCK_DATA_FILE)

        for symbol in self._stocks:
            df_new = self.df_for_export(symbol)
            if df_new is None:
                continue

            if symbol in wb.sheetnames:
                existing = pd.read_excel(
                    STOCK_DATA_FILE, sheet_name=symbol,
                    index_col=0, engine="openpyxl",
                )
                existing.index = pd.to_datetime(existing.index).date
                last_date = existing.index.max()
                df_append = df_new[df_new.index > last_date]

                if df_append.empty:
                    self._cb("log", f"{symbol}: no new rows to append")
                    continue

                df_combined = pd.concat([existing, df_append])
                del wb[symbol]
                wb.save(STOCK_DATA_FILE)
            else:
                df_combined = df_new

            with pd.ExcelWriter(
                STOCK_DATA_FILE, engine="openpyxl",
                mode="a", if_sheet_exists="replace",
            ) as writer:
                df_combined.to_excel(writer, sheet_name=symbol)

            self._cb("log", f"{symbol}: stock data updated in {STOCK_DATA_FILE}")

        return os.path.abspath(STOCK_DATA_FILE)

    # ------------------------------------------------------------------ #
    #  Excel: predictions                                                  #
    # ------------------------------------------------------------------ #

    def export_predictions(self) -> str:
        """
        Write all current predictions to a fresh *PREDICTIONS_FILE*.

        Returns an absolute path on success, raises on failure.
        """
        with pd.ExcelWriter(PREDICTIONS_FILE, engine="openpyxl") as writer:
            for symbol in self._stocks:
                df_pred = self.pred_df_for_export(symbol)
                if df_pred is not None:
                    df_pred.to_excel(writer, sheet_name=symbol, index=False)
        return os.path.abspath(PREDICTIONS_FILE)

    def update_predictions(self) -> str:
        """
        Append new prediction rows to *PREDICTIONS_FILE*, preserving history.

        Creates the file from scratch if it does not yet exist.
        Returns an absolute path on success, raises on failure.
        """
        if not os.path.exists(PREDICTIONS_FILE):
            return self.export_predictions()

        for symbol in self._stocks:
            df_new = self.pred_df_for_export(symbol)
            if df_new is None:
                continue

            try:
                existing = pd.read_excel(
                    PREDICTIONS_FILE, sheet_name=symbol, engine="openpyxl"
                )
                df_combined = pd.concat([existing, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new

            with pd.ExcelWriter(
                PREDICTIONS_FILE, engine="openpyxl",
                mode="a", if_sheet_exists="replace",
            ) as writer:
                df_combined.to_excel(writer, sheet_name=symbol, index=False)

            self._cb("log", f"{symbol}: predictions appended to {PREDICTIONS_FILE}")

        return os.path.abspath(PREDICTIONS_FILE)

    # ------------------------------------------------------------------ #
    #  Accuracy scoring                                                    #
    # ------------------------------------------------------------------ #

    def score_symbol_entry(self, symbol: str) -> ScoreResult:
        """
        Compute and store an accuracy score for *symbol*.

        Compares every archived prediction in pred_history against the
        actual closing prices in raw_df and returns a ScoreResult.
        The result is also stored in stocks[symbol]["accuracy_score"].
        """
        symbol = symbol.upper()
        data = self._stocks.get(symbol)
        if data is None:
            from stock_scorer import _insufficient_data_result
            return _insufficient_data_result(f"{symbol} is not tracked.")

        ph = data.get("pred_history", [])
        df = data.get("raw_df")
        cp = (data.get("prediction") or {}).get("current_price", 0.0)

        if df is None:
            from stock_scorer import _insufficient_data_result
            return _insufficient_data_result(f"{symbol} has no data.")
        
        result = score_symbol(ph, df, cp)
        data["accuracy_score"] = result
        self._cb("log",     f"{symbol} accuracy score: {result.score}/100 [{result.letter_grade}]")
        self._cb("refresh", symbol)
        return result

    def score_all_entries(self) -> dict:
        """
        Compute accuracy scores for every tracked symbol.

        Returns a dict mapping symbol → ScoreResult and updates each
        stocks[symbol]["accuracy_score"] in place.
        """
        results = score_all(self._stocks)
        for symbol, result in results.items():
            self._stocks[symbol]["accuracy_score"] = result
            self._cb("log",     f"{symbol} accuracy score: {result.score}/100 [{result.letter_grade}]")
            self._cb("refresh", symbol)
        return results

    def _train_thread(self, symbol: str) -> None:
        try:
            data = self._stocks[symbol]
            system = StockTradingSystem(api_key="", lookback_window=data["lookback"])
            self._cb("status", f"Training {symbol}…")

            end_date   = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            raw_df = system.api.fetch_stock_data(symbol, start_date, end_date)
            data["raw_df"] = raw_df

            system.train_model(symbol, epochs=data["epochs"])
            data["system"] = system
            data["status"] = "Trained"

            pred = system.predict_next_day(symbol, include_scenarios=True)
            data["prediction"] = pred
            data["status"] = "Ready"

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
            if not data["system"]:
                self._cb("log", f"{symbol} not yet trained.")
                return
            self._cb("status", f"Predicting {symbol}…")

            pred = data["system"].predict_next_day(symbol, include_scenarios=True)
            self._archive_prediction(data)

            data["prediction"] = pred
            data["status"] = "Ready"
            self._cb("refresh", symbol)
            self._cb("chart",   symbol)
            self._cb("log",     f"✓ {symbol} predictions refreshed")
            self._cb("status",  "Ready")
        except Exception as exc:
            self._cb("log", f"✗ {symbol} predict error: {exc}")

    def _update_thread(self, symbol: str) -> None:
        try:
            data = self._stocks[symbol]
            if not data["system"]:
                self._cb("log", f"{symbol} not yet trained.")
                return
            self._cb("status", f"Updating {symbol}…")

            data["system"].adaptive_update(symbol)
            self._archive_prediction(data)

            pred = data["system"].predict_next_day(symbol, include_scenarios=True)
            data["prediction"] = pred
            data["status"] = "Ready"
            self._cb("refresh", symbol)
            self._cb("chart",   symbol)
            self._cb("log",     f"✓ {symbol} adaptively updated")
            self._cb("status",  "Ready")
        except Exception as exc:
            self._cb("log", f"✗ {symbol} update error: {exc}")

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _archive_prediction(self, data: StockEntry) -> None:
        """Push the current prediction into pred_history before overwriting it."""
        old_pred = data.get("prediction")
        if old_pred and "scenarios" in old_pred:
            data["pred_history"].append({
                "date":  datetime.now(),
                "avg":   old_pred["scenarios"]["average_case"]["close"],
                "best":  old_pred["scenarios"]["best_case"]["close"],
                "worst": old_pred["scenarios"]["worst_case"]["close"],
            })
            # Refresh the accuracy score with the newly archived record
            ph = data.get("pred_history", [])
            df = data.get("raw_df")
            cp = (old_pred or {}).get("current_price", 0.0)
            if df is not None:
                data["accuracy_score"] = score_symbol(ph, df, cp)
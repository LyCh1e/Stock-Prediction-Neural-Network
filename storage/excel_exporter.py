"""
Responsible for exporting OHLCV data and predictions to Excel files.

Single Responsibility: Excel I/O only. No ML, no registry logic.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from scoring.scorer import _parse_records, _match_actuals

_LOCK = threading.Lock()

STOCK_DATA_FILE  = "stock_data.xlsx"
PREDICTIONS_FILE = "stock_predictions.xlsx"


class ExcelExporter:
    """Exports OHLCV history and prediction scenarios to Excel files."""

    def __init__(
        self,
        stock_data_file: str = STOCK_DATA_FILE,
        predictions_file: str = PREDICTIONS_FILE,
    ) -> None:
        self._data_file = stock_data_file
        self._pred_file = predictions_file

    # ------------------------------------------------------------------ #
    #  OHLCV data                                                         #
    # ------------------------------------------------------------------ #

    def export_stock_data(self, stocks: Dict) -> str:
        with _LOCK:
            with pd.ExcelWriter(self._data_file, engine="openpyxl") as writer:
                for symbol, data in stocks.items():
                    df_out = self._df_for_export(data)
                    if df_out is not None:
                        df_out.to_excel(writer, sheet_name=symbol)
        return os.path.abspath(self._data_file)

    def update_stock_data(self, stocks: Dict) -> str:
        if not os.path.exists(self._data_file):
            return self.export_stock_data(stocks)

        from openpyxl import load_workbook
        try:
            wb = load_workbook(self._data_file)
        except Exception:
            return self.export_stock_data(stocks)

        with _LOCK:
            for symbol, data in stocks.items():
                df_new = self._df_for_export(data)
                if df_new is None:
                    continue
                if symbol in wb.sheetnames:
                    existing = pd.read_excel(
                        self._data_file, sheet_name=symbol,
                        index_col=0, engine="openpyxl",
                    )
                    existing.index = pd.to_datetime(existing.index).date
                    last_date  = existing.index.max()
                    df_append  = df_new[df_new.index > last_date]
                    if df_append.empty:
                        continue
                    df_combined = pd.concat([existing, df_append])
                    del wb[symbol]
                    wb.save(self._data_file)
                else:
                    df_combined = df_new
                with pd.ExcelWriter(
                    self._data_file, engine="openpyxl",
                    mode="a", if_sheet_exists="replace",
                ) as writer:
                    df_combined.to_excel(writer, sheet_name=symbol)

        return os.path.abspath(self._data_file)

    # ------------------------------------------------------------------ #
    #  Predictions                                                        #
    # ------------------------------------------------------------------ #

    def export_predictions(self, stocks: Dict) -> str:
        with _LOCK:
            with pd.ExcelWriter(self._pred_file, engine="openpyxl") as writer:
                for symbol, data in stocks.items():
                    df_pred = self._pred_df_for_export(symbol, data)
                    if df_pred is not None:
                        df_pred.to_excel(writer, sheet_name=symbol, index=False)
        return os.path.abspath(self._pred_file)

    def update_predictions(self, stocks: Dict) -> str:
        if not os.path.exists(self._pred_file):
            return self.export_predictions(stocks)

        with _LOCK:
            for symbol, data in stocks.items():
                df_new = self._pred_df_for_export(symbol, data)
                if df_new is None:
                    continue
                try:
                    existing    = pd.read_excel(self._pred_file, sheet_name=symbol, engine="openpyxl")
                    df_combined = pd.concat([existing, df_new], ignore_index=True)
                except Exception:
                    df_combined = df_new
                with pd.ExcelWriter(
                    self._pred_file, engine="openpyxl",
                    mode="a", if_sheet_exists="replace",
                ) as writer:
                    df_combined.to_excel(writer, sheet_name=symbol, index=False)

        return os.path.abspath(self._pred_file)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _df_for_export(data: Dict) -> Optional[pd.DataFrame]:
        df = data.get("raw_df")
        if df is None:
            return None
        df_out = df[["open", "high", "low", "close", "volume"]].copy()
        df_out.columns = ["Open", "High", "Low", "Close", "Volume"]
        if hasattr(df_out.index, "tz") and df_out.index.tz is not None:
            df_out.index = df_out.index.tz_localize(None)
        df_out.index = df_out.index.date
        df_out.index.name = "Date"
        today = datetime.now().date()
        df_out = df_out[df_out.index < today]
        return df_out

    def load_pred_history(self, symbol: str) -> List[Dict]:
        """
        Reconstruct a pred_history list from stock_predictions.xlsx.

        Reads the Best/Average/Worst Case scenario rows and groups them by
        their shared "Exported At" timestamp to rebuild {date, avg, best, worst}
        entries — one per export batch.
        """
        if not os.path.exists(self._pred_file):
            return []
        try:
            df = pd.read_excel(self._pred_file, sheet_name=symbol, engine="openpyxl")
        except Exception:
            return []

        scenario_rows = df[df["Scenario"].isin(["Best Case", "Average Case", "Worst Case"])]
        if scenario_rows.empty:
            return []

        groups: Dict[str, Dict] = {}
        for _, row in scenario_rows.iterrows():
            key      = str(row["Exported At"])
            scenario = str(row["Scenario"])
            try:
                close = float(row["Close"])
            except (ValueError, TypeError):
                continue
            groups.setdefault(key, {})[scenario] = close

        records = []
        for exported_at, scenarios in groups.items():
            if "Average Case" not in scenarios:
                continue
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(exported_at[:19], fmt) if len(fmt) == 19 else \
                         datetime.strptime(exported_at[:16], fmt) if len(fmt) == 16 else \
                         datetime.strptime(exported_at[:10], fmt)
                    break
                except ValueError:
                    continue
            else:
                continue
            records.append({
                "date":  dt,
                "avg":   scenarios["Average Case"],
                "best":  scenarios.get("Best Case",  scenarios["Average Case"]),
                "worst": scenarios.get("Worst Case", scenarios["Average Case"]),
            })

        return records

    @staticmethod
    def _pred_df_for_export(symbol: str, data: Dict) -> Optional[pd.DataFrame]:
        pred = data.get("prediction")
        if not pred or "scenarios" not in pred:
            return None

        sc   = pred["scenarios"]
        rows = []
        for label, key in [
            ("Best Case",    "best_case"),
            ("Average Case", "average_case"),
            ("Worst Case",   "worst_case"),
        ]:
            s = sc[key]
            rows.append({
                "Exported At":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Scenario":       label,
                "Open":           round(s["open"],  2),
                "High":           round(s["high"],  2),
                "Low":            round(s["low"],   2),
                "Close":          round(s["close"], 2),
                "Profit %":       round(s["profit_potential"], 2),
                "Current Price":  round(pred["current_price"], 2),
                "Confidence":     round(pred["confidence"] * 100, 1),
                "Signal":         pred.get("recommendation", "").replace("_", " "),
            })

        score_rows = ExcelExporter._build_daily_score_rows(symbol, data)
        df_pred    = pd.DataFrame(rows)
        if score_rows:
            df_pred = pd.concat([df_pred, pd.DataFrame(score_rows)], ignore_index=True)
        return df_pred

    @staticmethod
    def _build_daily_score_rows(symbol: str, data: Dict) -> List[Dict]:
        ph = data.get("pred_history", [])
        df = data.get("raw_df")
        if not ph or df is None:
            return []

        records = _parse_records(ph)
        records = _match_actuals(records, df)
        matched = [r for r in records if r.actual is not None]
        if not matched:
            return []

        rows: List[Dict] = []

        for r in matched:
            if r.avg is None or r.actual is None:
                continue
            in_range = (
                r.best is not None and r.worst is not None and r.actual is not None
                and min(r.best, r.worst) <= r.actual <= max(r.best, r.worst)
            )

            pred_date = (r.date.strftime("%Y-%m-%d %H:%M") if hasattr(r.date, "strftime") else str(r.date))

            rows.append({
                "Exported At":    pred_date,
                "Scenario":       "── Daily Score ──",
                "Open":  "", "High": "", "Low": "",
                "Close":          f"Pred {round(r.avg or 0, 2)}  →  Actual {round(r.actual or 0, 2)}",
                "Profit %":       round((r.actual - r.avg) / (abs(r.avg) + 1e-8) * 100, 2) if r.actual and r.avg else 0,
                "Current Price":  round(r.actual or 0, 2),
                "Confidence":     "",
                "Signal":         "✓ In range" if in_range else "✗ Outside",
            })

        return rows

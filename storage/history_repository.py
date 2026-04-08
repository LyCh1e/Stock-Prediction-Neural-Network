# CSV persistence of prediction history — save/load pred_history per symbol.

from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Dict, List

import pandas as pd

from core.interfaces import IHistoryRepository

_LOCK = threading.Lock()


# Appends prediction history rows to a CSV file, one symbol per group.
class CsvHistoryRepository(IHistoryRepository):

    def __init__(self, filepath: str = "stock_models_history.csv") -> None:
        self._filepath = filepath

    # ------------------------------------------------------------------ #
    #  IHistoryRepository implementation                                  #
    # ------------------------------------------------------------------ #

    # Write the latest 5 prediction rows for symbol to CSV, replacing any existing rows for that symbol.
    def save(self, symbol: str, pred_history: List[Dict]) -> None:
        if not pred_history:
            return

        rows = []
        for p in pred_history:
            dt = p["date"]
            rows.append({
                "Symbol": symbol,
                "Date":   dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dt, "strftime") else str(dt),
                "Avg":    round(float(p.get("avg",   0)), 4),
                "Best":   round(float(p.get("best",  0)), 4),
                "Worst":  round(float(p.get("worst", 0)), 4),
            })

        df_new = pd.DataFrame(rows)
        df_new = df_new.sort_values("Date").tail(5)
        tmp    = self._filepath + ".tmp"

        try:
            with _LOCK:
                if os.path.exists(self._filepath):
                    existing = pd.read_csv(self._filepath)
                    existing = existing[existing["Symbol"] != symbol]
                    df_hist  = pd.concat([existing, df_new], ignore_index=True)
                else:
                    df_hist = df_new

                df_hist.to_csv(tmp, index=False)
                os.replace(tmp, self._filepath)
        except Exception as exc:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise exc

    # Read and return the prediction history rows for symbol from CSV as a list of dicts.
    def load(self, symbol: str) -> List[Dict]:
        if not os.path.exists(self._filepath):
            return []
        try:
            df = pd.read_csv(self._filepath)
            df = df[df["Symbol"] == symbol]
            records = []
            for _, row in df.iterrows():
                try:
                    dt = datetime.strptime(str(row["Date"]), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    dt = datetime.strptime(str(row["Date"])[:10], "%Y-%m-%d")
                records.append({
                    "date":  dt,
                    "avg":   float(row["Avg"]),
                    "best":  float(row["Best"]),
                    "worst": float(row["Worst"]),
                })
            return records
        except Exception:
            return []

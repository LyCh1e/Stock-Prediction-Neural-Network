# Uses historical prediction errors from prediction_score.xlsx to calibrate
# best/worst case scenario bands so more actuals fall within the predicted range.

from __future__ import annotations

import copy
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_calibration(symbol: str, score_file: str) -> Optional[Dict]:
    """Read per-symbol error statistics from prediction_score.xlsx.

    Returns None when the file is missing or fewer than 5 matched predictions
    exist — in those cases the predictor falls back to its volatility-based bands.
    """
    if not os.path.exists(score_file):
        return None
    try:
        df = pd.read_excel(score_file, sheet_name=symbol, engine="openpyxl")
    except Exception:
        return None

    if len(df) < 5 or not {"Predicted Close", "Actual Close"}.issubset(df.columns):
        return None

    predicted = df["Predicted Close"].astype(float)
    actual    = df["Actual Close"].astype(float)

    # Relative residuals: positive = actual was higher than predicted
    residuals = (actual - predicted) / (predicted.abs() + 1e-8)

    in_range_rate = (
        float((df["In Range"] == "Yes").mean())
        if "In Range" in df.columns
        else 0.5
    )

    return {
        "p10":          float(np.percentile(residuals, 10)),
        "p90":          float(np.percentile(residuals, 90)),
        "mean_error":   float(residuals.mean()),
        "mae_pct":      float(residuals.abs().mean() * 100),
        "n":            len(df),
        "in_range_rate": in_range_rate,
    }


def apply_calibration(prediction: Dict, calibration: Dict) -> Dict:
    """Replace best/worst case close prices with calibration-derived bounds.

    The band is built from the 10th and 90th percentiles of historical residuals,
    then widened proportionally when the current in-range rate is below 75%.
    """
    if "scenarios" not in prediction:
        return prediction

    avg_close  = float(prediction["scenarios"]["average_case"]["close"])
    curr_close = float(prediction["current_price"])

    p10  = calibration["p10"]
    p90  = calibration["p90"]
    rate = calibration["in_range_rate"]

    # Widen the band if in-range rate is below the 75% target; cap at 3× to avoid runaway expansion
    scale = max(1.0, min(3.0, 0.75 / (rate + 1e-8))) if rate < 0.75 else 1.0

    mid        = (p90 + p10) / 2
    half_band  = max(abs(p90 - p10) / 2, 0.005)   # minimum ±0.5% buffer
    p90_scaled = mid + half_band * scale
    p10_scaled = mid - half_band * scale

    best_close  = avg_close * (1 + p90_scaled)
    worst_close = avg_close * (1 + p10_scaled)

    pred = copy.deepcopy(prediction)
    best  = pred["scenarios"]["best_case"]
    worst = pred["scenarios"]["worst_case"]

    best["close"]            = round(best_close, 2)
    best["high"]             = round(max(float(best["high"]), best_close), 2)
    best["profit_potential"] = round((best_close - curr_close) / (curr_close + 1e-8) * 100, 2)
    best["target_price"]     = round(best_close, 2)

    worst["close"]            = round(worst_close, 2)
    worst["low"]              = round(min(float(worst["low"]), worst_close), 2)
    worst["profit_potential"] = round((worst_close - curr_close) / (curr_close + 1e-8) * 100, 2)
    worst["target_price"]     = round(worst_close, 2)

    return pred

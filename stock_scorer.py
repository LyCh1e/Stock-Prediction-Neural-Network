"""
stock_scorer.py
~~~~~~~~~~~~~~~
Accuracy scoring for the Stock Price Predictor.

Given a list of past prediction records (each containing a predicted close
and the date the prediction was made) and the actual OHLCV DataFrame, this
module:

  1. Matches each prediction to the actual closing price on the target date.
  2. Computes per-prediction error metrics (absolute %, direction accuracy).
  3. Aggregates those metrics into a single score in [0, 100] where:
       100 = perfect predictions every time
         0 = catastrophically wrong every time

Scoring formula
---------------
The score is built from three components, each weighted:

  Component                     Weight   What it rewards
  ─────────────────────────────────────────────────────────
  A. MAPE accuracy              50 %     How close the predicted price was
  B. Directional accuracy       30 %     Did the prediction get up/down right?
  C. Within-range accuracy      20 %     Was the actual close inside the
                                          predicted best/worst band?

  raw_score = 0.50 * A + 0.30 * B + 0.20 * C    (each already in [0,1])
  final_score = round(raw_score * 100, 1)

Usage
-----
    from stock_scorer import score_symbol, ScoreResult

    result = score_symbol(pred_history, raw_df, current_price)
    print(result.score)          # e.g. 73.4
    print(result.letter_grade)   # e.g. "B"
    print(result.summary)        # human-readable paragraph
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


# ── Public data classes ──────────────────────────────────────────────────── #

@dataclass
class PredictionRecord:
    """
    One archived prediction entry (as stored in StockEntry["pred_history"]).

    Fields
    ------
    date        : datetime the prediction was *made* (i.e. "predict for next day")
    avg         : average-case predicted close
    best        : best-case predicted close
    worst       : worst-case predicted close
    actual      : filled in by the scorer after matching to real data
    """
    date:   datetime
    avg:    float
    best:   float
    worst:  float
    actual: Optional[float] = None


@dataclass
class ScoreResult:
    """
    Full scoring result for one symbol.

    Attributes
    ----------
    score               : float in [0, 100]
    letter_grade        : "A+" … "F"
    mape_score          : 0-100  (price closeness component)
    directional_score   : 0-100  (up/down direction component)
    range_score         : 0-100  (actual inside predicted band component)
    matched_predictions : number of predictions that could be compared
    total_predictions   : number of archived predictions
    mean_abs_error_pct  : mean absolute percentage error (raw %, lower = better)
    directional_accuracy: fraction of predictions with correct direction (0-1)
    within_range_pct    : fraction of predictions where actual fell in band (0-1)
    details             : list of per-prediction breakdown dicts
    summary             : human-readable score explanation
    """
    score:                float
    letter_grade:         str
    mape_score:           float
    directional_score:    float
    range_score:          float
    matched_predictions:  int
    total_predictions:    int
    mean_abs_error_pct:   float
    directional_accuracy: float
    within_range_pct:     float
    details:              List[dict] = field(default_factory=list)
    summary:              str = ""


# ── Weights ───────────────────────────────────────────────────────────────── #
_W_MAPE      = 0.50
_W_DIR       = 0.30
_W_RANGE     = 0.20

# MAPE → score curve: error of X% maps to a score using exponential decay.
# At 0% error → 100, at 2% → ~82, at 5% → ~60, at 10% → ~37, at 20% → ~14
_MAPE_DECAY  = 0.15   # tune to taste


# ── Public API ───────────────────────────────────────────────────────────── #

def score_symbol(
    pred_history: list,
    raw_df: pd.DataFrame,
    current_price: float,
) -> ScoreResult:
    """
    Score a symbol's prediction history against actual prices.

    Parameters
    ----------
    pred_history  : list of dicts with keys date/avg/best/worst
                    (the format stored in StockEntry["pred_history"])
    raw_df        : OHLCV DataFrame with a DatetimeIndex (the raw_df from
                    StockEntry["raw_df"])
    current_price : latest known close price (used as a fallback reference)

    Returns
    -------
    ScoreResult
    """
    if not pred_history:
        return _insufficient_data_result("No prediction history recorded yet.")

    records = _parse_records(pred_history)
    records = _match_actuals(records, raw_df)

    matched = [r for r in records if r.actual is not None]
    if not matched:
        return _insufficient_data_result(
            f"Have {len(records)} prediction(s) but none could be matched to "
            "actual closing prices yet — the target dates may still be in the future."
        )

    # ── Component A: MAPE ─────────────────────────────────────────────── #
    abs_pct_errors = [
        abs(r.avg - r.actual) / (abs(r.actual) + 1e-8) * 100
        for r in matched
    ]
    mean_ape       = float(np.mean(abs_pct_errors))
    mape_raw       = math.exp(-_MAPE_DECAY * mean_ape)   # (0, 1]
    mape_score     = mape_raw * 100

    # ── Component B: Directional accuracy ────────────────────────────── #
    # We need a "previous close" for each prediction.
    # We approximate it as: previous close ≈ actual - (avg - prev_actual).
    # Simpler & more robust: use the actual close on the day *before* target.
    dir_correct = []
    for r in matched:
        prev_close = _prev_close(r.date, raw_df)
        if prev_close is None:
            continue
        predicted_up = r.avg >= prev_close
        actual_up    = r.actual >= prev_close
        dir_correct.append(predicted_up == actual_up)

    if dir_correct:
        dir_acc       = float(np.mean(dir_correct))
        dir_score     = dir_acc * 100
    else:
        dir_acc   = 0.5
        dir_score = 50.0

    # ── Component C: Within-range accuracy ───────────────────────────── #
    in_range = [
        min(r.best, r.worst) <= r.actual <= max(r.best, r.worst)
        for r in matched
    ]
    within_pct  = float(np.mean(in_range))
    range_score = within_pct * 100

    # ── Composite score ───────────────────────────────────────────────── #
    raw = _W_MAPE * mape_raw + _W_DIR * dir_acc + _W_RANGE * within_pct
    final_score = round(min(100.0, max(0.0, raw * 100)), 1)

    grade   = _letter_grade(final_score)
    details = _build_details(matched, raw_df)
    summary = _build_summary(final_score, grade, mean_ape, dir_acc, within_pct,
                              len(matched), len(records))

    return ScoreResult(
        score=final_score,
        letter_grade=grade,
        mape_score=round(mape_score, 1),
        directional_score=round(dir_score, 1),
        range_score=round(range_score, 1),
        matched_predictions=len(matched),
        total_predictions=len(records),
        mean_abs_error_pct=round(mean_ape, 3),
        directional_accuracy=round(dir_acc, 4),
        within_range_pct=round(within_pct, 4),
        details=details,
        summary=summary,
    )


def score_all(store_stocks: dict) -> dict:
    """
    Score every symbol that has both pred_history and raw_df.

    Parameters
    ----------
    store_stocks : the StockStore._stocks dict

    Returns
    -------
    dict mapping symbol → ScoreResult
    """
    results = {}
    for symbol, data in store_stocks.items():
        ph  = data.get("pred_history", [])
        df  = data.get("raw_df")
        cp  = (data.get("prediction") or {}).get("current_price", 0.0)
        if df is not None:
            results[symbol] = score_symbol(ph, df, cp)
        else:
            results[symbol] = _insufficient_data_result("No price data available yet.")
    return results


# ── Private helpers ───────────────────────────────────────────────────────── #

def _parse_records(pred_history: list) -> List[PredictionRecord]:
    """Convert raw dicts from pred_history into PredictionRecord objects."""
    records = []
    for entry in pred_history:
        dt = entry.get("date")
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except ValueError:
                continue
        if dt is None:
            continue
        records.append(PredictionRecord(
            date=dt,
            avg=float(entry.get("avg",   entry.get("close", 0))),
            best=float(entry.get("best",  entry.get("avg", 0))),
            worst=float(entry.get("worst", entry.get("avg", 0))),
        ))
    return records


def _match_actuals(
    records: List[PredictionRecord],
    raw_df: pd.DataFrame,
) -> List[PredictionRecord]:
    """
    For each record, find the actual closing price on the next business day
    after `record.date` (the date the prediction was *made*).
    """
    if raw_df is None or raw_df.empty:
        return records

    # Normalise index to tz-naive dates
    idx = raw_df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    close_by_date = dict(zip(idx.normalize(), raw_df["close"].values))

    for rec in records:
        target = _next_business_day(rec.date)
        # Try target date and up to 3 days forward (for weekends / holidays)
        for offset in range(4):
            candidate = pd.Timestamp(target + timedelta(days=offset)).normalize()
            if candidate in close_by_date:
                rec.actual = float(close_by_date[candidate])
                break
    return records


def _next_business_day(dt: datetime) -> datetime:
    """Return the next weekday after *dt*."""
    nxt = dt + timedelta(days=1)
    while nxt.weekday() >= 5:   # 5=Saturday, 6=Sunday
        nxt += timedelta(days=1)
    return nxt


def _prev_close(prediction_date: datetime, raw_df: pd.DataFrame) -> Optional[float]:
    """
    Return the closing price on the last trading day *before* prediction_date.
    """
    if raw_df is None or raw_df.empty:
        return None

    idx = raw_df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    cutoff = pd.Timestamp(prediction_date).normalize()
    prior  = idx[idx < cutoff]
    if prior.empty:
        return None
    return float(raw_df["close"].iloc[raw_df.index.get_loc(prior[-1])])


def _build_details(
    matched: List[PredictionRecord],
    raw_df: pd.DataFrame,
) -> List[dict]:
    """Build a list of per-prediction breakdown dicts for display / export."""
    rows = []
    for r in matched:
        err_pct   = abs(r.avg - r.actual) / (abs(r.actual) + 1e-8) * 100
        in_range  = min(r.best, r.worst) <= r.actual <= max(r.best, r.worst)
        prev      = _prev_close(r.date, raw_df)
        direction = "N/A"
        if prev is not None:
            predicted_up = r.avg  >= prev
            actual_up    = r.actual >= prev
            direction = "✓" if predicted_up == actual_up else "✗"
        rows.append({
            "Prediction Date": r.date.strftime("%Y-%m-%d %H:%M") if hasattr(r.date, "strftime") else str(r.date),
            "Predicted Close": round(r.avg, 2),
            "Best Case":       round(r.best, 2),
            "Worst Case":      round(r.worst, 2),
            "Actual Close":    round(r.actual, 2),
            "Abs Error %":     round(err_pct, 3),
            "In Range":        "✓" if in_range else "✗",
            "Direction":       direction,
        })
    return rows


def _letter_grade(score: float) -> str:
    if score >= 93: return "A+"
    if score >= 87: return "A"
    if score >= 80: return "A-"
    if score >= 74: return "B+"
    if score >= 67: return "B"
    if score >= 60: return "B-"
    if score >= 54: return "C+"
    if score >= 47: return "C"
    if score >= 40: return "C-"
    if score >= 34: return "D+"
    if score >= 27: return "D"
    if score >= 20: return "D-"
    return "F"


def _build_summary(
    score: float,
    grade: str,
    mean_ape: float,
    dir_acc: float,
    within_pct: float,
    matched: int,
    total: int,
) -> str:
    lines = [
        f"Algorithm Score: {score}/100  [{grade}]",
        f"Based on {matched} matched prediction(s) out of {total} archived.",
        "",
        f"  • Price closeness (MAPE):   avg error = {mean_ape:.2f}%",
        f"  • Direction accuracy:        {dir_acc * 100:.1f}% of moves called correctly",
        f"  • Band accuracy:             {within_pct * 100:.1f}% of actuals landed inside the predicted range",
        "",
    ]
    if score >= 80:
        lines.append("Verdict: Excellent — the model is performing very well.")
    elif score >= 60:
        lines.append("Verdict: Good — the model captures most moves accurately.")
    elif score >= 40:
        lines.append("Verdict: Fair — room for improvement; consider retraining.")
    else:
        lines.append("Verdict: Poor — predictions have significant errors. Retrain with more data or epochs.")
    return "\n".join(lines)


def _insufficient_data_result(reason: str) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        letter_grade="N/A",
        mape_score=0.0,
        directional_score=0.0,
        range_score=0.0,
        matched_predictions=0,
        total_predictions=0,
        mean_abs_error_pct=0.0,
        directional_accuracy=0.0,
        within_range_pct=0.0,
        details=[],
        summary=f"Score unavailable: {reason}",
    )

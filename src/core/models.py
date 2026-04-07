"""
src/core/models.py
~~~~~~~~~~~~~~~~~~
Pure data classes shared across all layers.

No business logic lives here — only data shape definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class PredictionRecord:
    """
    One archived prediction entry.

    Fields
    ------
    date    : datetime the prediction was *made*
    avg     : average-case predicted close
    best    : best-case predicted close
    worst   : worst-case predicted close
    actual  : filled in by the scorer after matching to real data
    """
    date:   datetime
    avg:    float
    best:   float
    worst:  float
    actual: Optional[float] = None


@dataclass
class ScoreResult:
    """
    Full accuracy-scoring result for one symbol.

    Attributes
    ----------
    score               : float in [0, 100]
    letter_grade        : "A+" … "F"
    mape_score          : 0-100  (price closeness component)
    directional_score   : 0-100  (up/down direction component)
    range_score         : 0-100  (actual inside predicted band)
    matched_predictions : number of predictions that could be compared
    total_predictions   : number of archived predictions
    mean_abs_error_pct  : mean absolute percentage error (raw %)
    directional_accuracy: fraction of predictions with correct direction (0-1)
    within_range_pct    : fraction of predictions where actual fell in band (0-1)
    details             : per-prediction breakdown dicts
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

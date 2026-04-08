# Pure data classes shared across all layers — only data shape definitions, no logic.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# One archived prediction: date made, avg/best/worst predicted close, and actual (filled by scorer).
@dataclass
class PredictionRecord:
    date:   datetime
    avg:    float
    best:   float
    worst:  float
    actual: Optional[float] = None


# Full accuracy-scoring result for one symbol: composite score, grade, component scores, and details.
@dataclass
class ScoreResult:
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

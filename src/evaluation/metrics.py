"""Forecast scoring metrics for model evaluation.

Computes Brier score, log loss, and calibration statistics for
individual model predictions against resolved market outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ForecastRecord:
    """A single model forecast paired with the actual outcome."""

    model_id: str
    market_id: str
    p_true: float
    outcome: int  # 1 = YES, 0 = NO


@dataclass
class CalibrationBin:
    """One bin in a calibration analysis."""

    bin_lower: float
    bin_upper: float
    count: int
    mean_predicted: float
    actual_rate: float


@dataclass
class ModelMetrics:
    """Aggregate forecast metrics for a single model."""

    model_id: str
    markets_scored: int
    brier_score: float | None
    log_loss: float | None
    calibration_bins: list[CalibrationBin] = field(default_factory=list)


def brier_score(records: list[ForecastRecord]) -> float | None:
    """Compute mean Brier score: mean((p_true - outcome)^2).

    Returns ``None`` if *records* is empty.
    """
    if not records:
        return None
    return sum((r.p_true - r.outcome) ** 2 for r in records) / len(records)


def log_loss(records: list[ForecastRecord], eps: float = 1e-15) -> float | None:
    """Compute mean log loss.

    Clips predictions to ``[eps, 1 - eps]`` to avoid log(0).
    Returns ``None`` if *records* is empty.
    """
    if not records:
        return None
    total = 0.0
    for r in records:
        p = max(eps, min(1 - eps, r.p_true))
        total += -(r.outcome * math.log(p) + (1 - r.outcome) * math.log(1 - p))
    return total / len(records)


def calibration(
    records: list[ForecastRecord], num_bins: int = 10
) -> list[CalibrationBin]:
    """Compute calibration bins for a set of forecast records.

    Predictions are grouped into *num_bins* equal-width bins from 0 to 1.
    Each bin reports count, mean predicted probability, and actual outcome rate.
    Empty bins are omitted.
    """
    if not records:
        return []

    bin_width = 1.0 / num_bins
    bins: dict[int, list[ForecastRecord]] = {}

    for r in records:
        idx = min(int(r.p_true / bin_width), num_bins - 1)
        bins.setdefault(idx, []).append(r)

    result: list[CalibrationBin] = []
    for idx in sorted(bins):
        recs = bins[idx]
        mean_pred = sum(r.p_true for r in recs) / len(recs)
        actual = sum(r.outcome for r in recs) / len(recs)
        result.append(
            CalibrationBin(
                bin_lower=idx * bin_width,
                bin_upper=(idx + 1) * bin_width,
                count=len(recs),
                mean_predicted=mean_pred,
                actual_rate=actual,
            )
        )

    return result


def compute_model_metrics(
    model_id: str, records: list[ForecastRecord]
) -> ModelMetrics:
    """Compute all forecast metrics for a single model."""
    model_records = [r for r in records if r.model_id == model_id]
    return ModelMetrics(
        model_id=model_id,
        markets_scored=len(model_records),
        brier_score=brier_score(model_records),
        log_loss=log_loss(model_records),
        calibration_bins=calibration(model_records),
    )

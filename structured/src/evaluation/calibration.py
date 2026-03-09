"""Calibration metrics — Brier score, log loss, calibration curves.

Measures prediction quality per category and engine version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionOutcome:
    """A single prediction-outcome pair for calibration tracking."""

    market_id: str
    category: str
    engine_version: str
    p_yes: float
    outcome: float  # 1.0 = yes, 0.0 = no


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""

    category: str
    engine_version: str
    n_predictions: int
    n_resolved: int
    brier_score: float | None
    log_loss: float | None
    calibration_bins: list[dict[str, Any]]

    def to_stat_dict(self, stat_date: Any) -> dict[str, Any]:
        """Convert to dict for CalibrationStat DB row."""
        return {
            "category": self.category,
            "engine_version": self.engine_version,
            "stat_date": stat_date,
            "n_predictions": self.n_predictions,
            "n_resolved": self.n_resolved,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "calibration_json": {"bins": self.calibration_bins},
        }


def compute_brier_score(pairs: list[PredictionOutcome]) -> float | None:
    """Compute mean Brier score: mean((p_yes - outcome)^2).

    Returns ``None`` if no pairs are provided.
    """
    if not pairs:
        return None
    total = sum((p.p_yes - p.outcome) ** 2 for p in pairs)
    return total / len(pairs)


def compute_log_loss(pairs: list[PredictionOutcome], eps: float = 1e-15) -> float | None:
    """Compute mean log loss (binary cross-entropy).

    Returns ``None`` if no pairs are provided.
    """
    if not pairs:
        return None
    total = 0.0
    for p in pairs:
        clipped = max(eps, min(1.0 - eps, p.p_yes))
        if p.outcome == 1.0:
            total -= math.log(clipped)
        else:
            total -= math.log(1.0 - clipped)
    return total / len(pairs)


def compute_calibration_bins(
    pairs: list[PredictionOutcome], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Compute calibration curve bins.

    Groups predictions into bins by predicted probability, and computes
    mean predicted vs actual outcome frequency in each bin.
    """
    if not pairs:
        return []

    bin_width = 1.0 / n_bins
    bins: list[dict[str, Any]] = []

    for i in range(n_bins):
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width

        in_bin = [
            p for p in pairs
            if bin_lower <= p.p_yes < bin_upper
            or (i == n_bins - 1 and p.p_yes == 1.0)
        ]

        if not in_bin:
            bins.append({
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "n": 0,
                "mean_predicted": None,
                "mean_actual": None,
            })
            continue

        mean_pred = sum(p.p_yes for p in in_bin) / len(in_bin)
        mean_actual = sum(p.outcome for p in in_bin) / len(in_bin)
        bins.append({
            "bin_lower": round(bin_lower, 2),
            "bin_upper": round(bin_upper, 2),
            "n": len(in_bin),
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_actual, 4),
        })

    return bins


def evaluate_category(
    pairs: list[PredictionOutcome],
    category: str,
    engine_version: str,
) -> CalibrationResult:
    """Run full calibration analysis for a single category."""
    return CalibrationResult(
        category=category,
        engine_version=engine_version,
        n_predictions=len(pairs),
        n_resolved=len(pairs),
        brier_score=compute_brier_score(pairs),
        log_loss=compute_log_loss(pairs),
        calibration_bins=compute_calibration_bins(pairs),
    )

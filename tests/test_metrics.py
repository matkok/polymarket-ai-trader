"""Tests for src.evaluation.metrics — forecast scoring metrics."""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    CalibrationBin,
    ForecastRecord,
    ModelMetrics,
    brier_score,
    calibration,
    compute_model_metrics,
    log_loss,
)


# ---- Helpers ----------------------------------------------------------------


def _rec(
    model_id: str = "model-a",
    market_id: str = "mkt-1",
    p_true: float = 0.6,
    outcome: int = 1,
) -> ForecastRecord:
    return ForecastRecord(
        model_id=model_id,
        market_id=market_id,
        p_true=p_true,
        outcome=outcome,
    )


# ---- Brier score -----------------------------------------------------------


class TestBrierScore:
    """Brier score computation."""

    def test_perfect_prediction(self) -> None:
        """p_true=1.0 for outcome=1 → Brier = 0."""
        records = [_rec(p_true=1.0, outcome=1)]
        assert brier_score(records) == pytest.approx(0.0)

    def test_worst_prediction(self) -> None:
        """p_true=0.0 for outcome=1 → Brier = 1."""
        records = [_rec(p_true=0.0, outcome=1)]
        assert brier_score(records) == pytest.approx(1.0)

    def test_multiple_records(self) -> None:
        """Mean Brier across multiple records."""
        records = [
            _rec(p_true=0.8, outcome=1),  # (0.8-1)^2 = 0.04
            _rec(p_true=0.3, outcome=0),  # (0.3-0)^2 = 0.09
        ]
        expected = (0.04 + 0.09) / 2
        assert brier_score(records) == pytest.approx(expected)

    def test_empty_returns_none(self) -> None:
        assert brier_score([]) is None

    def test_baseline_uninformative(self) -> None:
        """p_true=0.5 always → Brier = 0.25."""
        records = [
            _rec(p_true=0.5, outcome=1),
            _rec(p_true=0.5, outcome=0),
        ]
        assert brier_score(records) == pytest.approx(0.25)


# ---- Log loss ---------------------------------------------------------------


class TestLogLoss:
    """Log loss computation."""

    def test_perfect_prediction(self) -> None:
        """Near-perfect prediction → log loss near 0."""
        records = [_rec(p_true=0.999, outcome=1)]
        result = log_loss(records)
        assert result is not None
        assert result < 0.01

    def test_worst_prediction(self) -> None:
        """p_true near 0 for outcome=1 → high log loss."""
        records = [_rec(p_true=0.01, outcome=1)]
        result = log_loss(records)
        assert result is not None
        assert result > 4.0

    def test_empty_returns_none(self) -> None:
        assert log_loss([]) is None

    def test_baseline_uninformative(self) -> None:
        """p_true=0.5 always → log loss = ln(2)."""
        records = [
            _rec(p_true=0.5, outcome=1),
            _rec(p_true=0.5, outcome=0),
        ]
        expected = math.log(2)
        assert log_loss(records) == pytest.approx(expected)

    def test_clips_to_avoid_log_zero(self) -> None:
        """p_true=0.0 is clipped to eps, preventing inf."""
        records = [_rec(p_true=0.0, outcome=1)]
        result = log_loss(records)
        assert result is not None
        assert math.isfinite(result)


# ---- Calibration -----------------------------------------------------------


class TestCalibration:
    """Calibration bin computation."""

    def test_empty_returns_empty(self) -> None:
        assert calibration([]) == []

    def test_single_record(self) -> None:
        records = [_rec(p_true=0.65, outcome=1)]
        bins = calibration(records, num_bins=10)
        assert len(bins) == 1
        assert bins[0].count == 1
        assert bins[0].mean_predicted == pytest.approx(0.65)
        assert bins[0].actual_rate == pytest.approx(1.0)

    def test_multiple_bins(self) -> None:
        records = [
            _rec(p_true=0.1, outcome=0),
            _rec(p_true=0.15, outcome=0),
            _rec(p_true=0.85, outcome=1),
            _rec(p_true=0.88, outcome=1),
        ]
        bins = calibration(records, num_bins=10)
        # Should have 2 bins: [0.1, 0.2) and [0.8, 0.9).
        assert len(bins) == 2
        low_bin = bins[0]
        high_bin = bins[1]
        assert low_bin.count == 2
        assert low_bin.actual_rate == pytest.approx(0.0)
        assert high_bin.count == 2
        assert high_bin.actual_rate == pytest.approx(1.0)

    def test_p_true_1_0_goes_to_last_bin(self) -> None:
        """p_true=1.0 should go to the last bin, not overflow."""
        records = [_rec(p_true=1.0, outcome=1)]
        bins = calibration(records, num_bins=10)
        assert len(bins) == 1
        assert bins[0].bin_lower == pytest.approx(0.9)
        assert bins[0].bin_upper == pytest.approx(1.0)

    def test_bin_boundaries(self) -> None:
        """Verify bin_lower and bin_upper values."""
        records = [_rec(p_true=0.35, outcome=1)]
        bins = calibration(records, num_bins=10)
        assert len(bins) == 1
        assert bins[0].bin_lower == pytest.approx(0.3)
        assert bins[0].bin_upper == pytest.approx(0.4)


# ---- compute_model_metrics -------------------------------------------------


class TestComputeModelMetrics:
    """End-to-end model metrics computation."""

    def test_filters_by_model_id(self) -> None:
        records = [
            _rec(model_id="model-a", p_true=0.8, outcome=1),
            _rec(model_id="model-b", p_true=0.2, outcome=0),
            _rec(model_id="model-a", p_true=0.6, outcome=0),
        ]
        metrics = compute_model_metrics("model-a", records)
        assert metrics.model_id == "model-a"
        assert metrics.markets_scored == 2
        assert metrics.brier_score is not None
        assert metrics.log_loss is not None

    def test_no_records_for_model(self) -> None:
        records = [_rec(model_id="model-b")]
        metrics = compute_model_metrics("model-a", records)
        assert metrics.markets_scored == 0
        assert metrics.brier_score is None
        assert metrics.log_loss is None
        assert metrics.calibration_bins == []

    def test_includes_calibration(self) -> None:
        records = [
            _rec(model_id="model-a", p_true=0.8, outcome=1),
            _rec(model_id="model-a", p_true=0.2, outcome=0),
        ]
        metrics = compute_model_metrics("model-a", records)
        assert len(metrics.calibration_bins) == 2

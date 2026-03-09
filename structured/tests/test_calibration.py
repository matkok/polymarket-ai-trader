"""Tests for evaluation — calibration, kill switch, and replay engine."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

from src.engines.base import PriceEstimate, PricingEngine
from src.evaluation.calibration import (
    CalibrationResult,
    PredictionOutcome,
    compute_brier_score,
    compute_calibration_bins,
    compute_log_loss,
    evaluate_category,
)
from src.evaluation.kill_switch import CategoryHealth, KillSwitch, KillSwitchConfig
from src.evaluation.replay import ReplayEngine, ReplayMarket, ReplayResult


# ===========================================================================
# Brier Score
# ===========================================================================


class TestBrierScore:
    """Brier score computation tests."""

    def test_empty_returns_none(self) -> None:
        assert compute_brier_score([]) is None

    def test_perfect_prediction_yes(self) -> None:
        pairs = [PredictionOutcome("m1", "weather", "v1", p_yes=1.0, outcome=1.0)]
        assert compute_brier_score(pairs) == 0.0

    def test_perfect_prediction_no(self) -> None:
        pairs = [PredictionOutcome("m1", "weather", "v1", p_yes=0.0, outcome=0.0)]
        assert compute_brier_score(pairs) == 0.0

    def test_worst_prediction(self) -> None:
        pairs = [PredictionOutcome("m1", "weather", "v1", p_yes=1.0, outcome=0.0)]
        assert compute_brier_score(pairs) == 1.0

    def test_coin_flip(self) -> None:
        pairs = [
            PredictionOutcome("m1", "weather", "v1", p_yes=0.5, outcome=1.0),
            PredictionOutcome("m2", "weather", "v1", p_yes=0.5, outcome=0.0),
        ]
        assert compute_brier_score(pairs) == 0.25

    def test_multiple_predictions(self) -> None:
        pairs = [
            PredictionOutcome("m1", "weather", "v1", p_yes=0.9, outcome=1.0),
            PredictionOutcome("m2", "weather", "v1", p_yes=0.1, outcome=0.0),
        ]
        expected = ((0.9 - 1.0) ** 2 + (0.1 - 0.0) ** 2) / 2
        assert abs(compute_brier_score(pairs) - expected) < 1e-10

    def test_good_predictions_low_brier(self) -> None:
        pairs = [
            PredictionOutcome("m1", "w", "v1", p_yes=0.9, outcome=1.0),
            PredictionOutcome("m2", "w", "v1", p_yes=0.8, outcome=1.0),
            PredictionOutcome("m3", "w", "v1", p_yes=0.1, outcome=0.0),
            PredictionOutcome("m4", "w", "v1", p_yes=0.2, outcome=0.0),
        ]
        score = compute_brier_score(pairs)
        assert score is not None
        assert score < 0.10


# ===========================================================================
# Log Loss
# ===========================================================================


class TestLogLoss:
    """Log loss computation tests."""

    def test_empty_returns_none(self) -> None:
        assert compute_log_loss([]) is None

    def test_perfect_prediction(self) -> None:
        pairs = [PredictionOutcome("m1", "w", "v1", p_yes=0.99, outcome=1.0)]
        loss = compute_log_loss(pairs)
        assert loss is not None
        assert loss < 0.02

    def test_terrible_prediction(self) -> None:
        pairs = [PredictionOutcome("m1", "w", "v1", p_yes=0.01, outcome=1.0)]
        loss = compute_log_loss(pairs)
        assert loss is not None
        assert loss > 3.0

    def test_coin_flip_loss(self) -> None:
        pairs = [PredictionOutcome("m1", "w", "v1", p_yes=0.5, outcome=1.0)]
        loss = compute_log_loss(pairs)
        assert loss is not None
        assert abs(loss - math.log(2)) < 1e-10

    def test_clamps_extreme_values(self) -> None:
        """Predictions at exactly 0.0 or 1.0 should not cause -inf."""
        pairs = [
            PredictionOutcome("m1", "w", "v1", p_yes=0.0, outcome=1.0),
            PredictionOutcome("m2", "w", "v1", p_yes=1.0, outcome=0.0),
        ]
        loss = compute_log_loss(pairs)
        assert loss is not None
        assert math.isfinite(loss)


# ===========================================================================
# Calibration Bins
# ===========================================================================


class TestCalibrationBins:
    """Calibration curve bin computation tests."""

    def test_empty_returns_empty(self) -> None:
        assert compute_calibration_bins([]) == []

    def test_ten_bins_returned(self) -> None:
        pairs = [PredictionOutcome("m1", "w", "v1", p_yes=0.5, outcome=1.0)]
        bins = compute_calibration_bins(pairs, n_bins=10)
        assert len(bins) == 10

    def test_predictions_in_correct_bin(self) -> None:
        pairs = [
            PredictionOutcome("m1", "w", "v1", p_yes=0.15, outcome=1.0),
            PredictionOutcome("m2", "w", "v1", p_yes=0.85, outcome=0.0),
        ]
        bins = compute_calibration_bins(pairs, n_bins=10)
        # 0.15 should be in bin [0.1, 0.2)
        assert bins[1]["n"] == 1
        assert bins[1]["mean_predicted"] == 0.15
        assert bins[1]["mean_actual"] == 1.0
        # 0.85 should be in bin [0.8, 0.9)
        assert bins[8]["n"] == 1

    def test_empty_bins_have_none_means(self) -> None:
        pairs = [PredictionOutcome("m1", "w", "v1", p_yes=0.5, outcome=1.0)]
        bins = compute_calibration_bins(pairs, n_bins=10)
        # Only bin [0.5, 0.6) should have data.
        for i, b in enumerate(bins):
            if i == 5:
                assert b["n"] == 1
            else:
                assert b["n"] == 0
                assert b["mean_predicted"] is None
                assert b["mean_actual"] is None

    def test_custom_bin_count(self) -> None:
        pairs = [PredictionOutcome("m1", "w", "v1", p_yes=0.5, outcome=1.0)]
        bins = compute_calibration_bins(pairs, n_bins=5)
        assert len(bins) == 5


# ===========================================================================
# Evaluate Category
# ===========================================================================


class TestEvaluateCategory:
    """Full category evaluation."""

    def test_returns_calibration_result(self) -> None:
        pairs = [
            PredictionOutcome("m1", "w", "v1", p_yes=0.8, outcome=1.0),
            PredictionOutcome("m2", "w", "v1", p_yes=0.2, outcome=0.0),
        ]
        result = evaluate_category(pairs, "weather", "weather_v1")
        assert isinstance(result, CalibrationResult)
        assert result.category == "weather"
        assert result.engine_version == "weather_v1"
        assert result.n_predictions == 2
        assert result.brier_score is not None
        assert result.log_loss is not None
        assert len(result.calibration_bins) == 10

    def test_to_stat_dict(self) -> None:
        pairs = [PredictionOutcome("m1", "w", "v1", p_yes=0.5, outcome=1.0)]
        result = evaluate_category(pairs, "weather", "weather_v1")
        stat = result.to_stat_dict(stat_date=datetime(2026, 2, 22).date())
        assert stat["category"] == "weather"
        assert stat["n_predictions"] == 1
        assert "bins" in stat["calibration_json"]


# ===========================================================================
# Kill Switch
# ===========================================================================


class TestKillSwitch:
    """Kill switch auto-disable tests."""

    def test_default_all_enabled(self) -> None:
        ks = KillSwitch()
        assert ks.is_enabled("weather")
        assert ks.is_enabled("macro")

    def test_disable_on_high_brier(self) -> None:
        config = KillSwitchConfig(max_brier_score=0.30, min_predictions_before_check=5)
        ks = KillSwitch(config=config)
        health = ks.check_category("weather", brier_score=0.40, n_predictions=10)
        assert health.is_active is False
        assert not ks.is_enabled("weather")
        assert len(health.kill_reasons) == 1

    def test_no_disable_under_brier_threshold(self) -> None:
        config = KillSwitchConfig(max_brier_score=0.30, min_predictions_before_check=5)
        ks = KillSwitch(config=config)
        health = ks.check_category("weather", brier_score=0.20, n_predictions=10)
        assert health.is_active is True
        assert ks.is_enabled("weather")

    def test_skip_brier_check_insufficient_predictions(self) -> None:
        config = KillSwitchConfig(max_brier_score=0.30, min_predictions_before_check=10)
        ks = KillSwitch(config=config)
        health = ks.check_category("weather", brier_score=0.50, n_predictions=5)
        assert health.is_active is True  # Not enough predictions to judge.

    def test_disable_on_high_drawdown(self) -> None:
        config = KillSwitchConfig(max_daily_drawdown_eur=100.0)
        ks = KillSwitch(config=config)
        health = ks.check_category("crypto", daily_drawdown_eur=150.0)
        assert health.is_active is False
        assert not ks.is_enabled("crypto")

    def test_disable_both_reasons(self) -> None:
        config = KillSwitchConfig(
            max_brier_score=0.30,
            max_daily_drawdown_eur=100.0,
            min_predictions_before_check=5,
        )
        ks = KillSwitch(config=config)
        health = ks.check_category(
            "macro", brier_score=0.50, daily_drawdown_eur=200.0, n_predictions=10
        )
        assert health.is_active is False
        assert len(health.kill_reasons) == 2

    def test_reset_category(self) -> None:
        ks = KillSwitch(config=KillSwitchConfig(max_daily_drawdown_eur=100.0))
        ks.check_category("weather", daily_drawdown_eur=200.0)
        assert not ks.is_enabled("weather")
        ks.reset_category("weather")
        assert ks.is_enabled("weather")

    def test_reset_all(self) -> None:
        ks = KillSwitch(config=KillSwitchConfig(max_daily_drawdown_eur=100.0))
        ks.check_category("weather", daily_drawdown_eur=200.0)
        ks.check_category("crypto", daily_drawdown_eur=200.0)
        assert not ks.is_enabled("weather")
        assert not ks.is_enabled("crypto")
        ks.reset_all()
        assert ks.is_enabled("weather")
        assert ks.is_enabled("crypto")

    def test_status(self) -> None:
        ks = KillSwitch(config=KillSwitchConfig(max_daily_drawdown_eur=100.0))
        ks.check_category("weather", daily_drawdown_eur=200.0)
        status = ks.status()
        assert "weather" in status["disabled_categories"]
        assert status["config"]["max_daily_drawdown_eur"] == 100.0

    def test_disabled_categories_returns_copy(self) -> None:
        ks = KillSwitch(config=KillSwitchConfig(max_daily_drawdown_eur=100.0))
        ks.check_category("weather", daily_drawdown_eur=200.0)
        cats = ks.disabled_categories
        cats.add("crypto")
        # Original should be unchanged.
        assert "crypto" not in ks.disabled_categories


# ===========================================================================
# Replay Engine
# ===========================================================================


class _MockEngine(PricingEngine):
    """Simple mock engine for replay tests."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def version(self) -> str:
        return "mock_v1"

    def compute(self, spec: Any, observation: Any) -> PriceEstimate:
        # Return p_yes based on observation value.
        if isinstance(observation, dict) and "mock_p_yes" in observation:
            return PriceEstimate(
                p_yes=observation["mock_p_yes"],
                confidence=0.80,
            )
        return PriceEstimate(p_yes=0.50, confidence=0.50)


class TestReplayEngine:
    """Replay engine tests."""

    def test_empty_markets(self) -> None:
        engine = _MockEngine()
        replay = ReplayEngine(engine=engine)
        result = replay.replay([])
        assert result.n_markets == 0
        assert result.brier_score is None

    def test_single_market_correct(self) -> None:
        engine = _MockEngine()
        replay = ReplayEngine(engine=engine)
        markets = [
            ReplayMarket(
                market_id="m1",
                category="weather",
                spec_json={"metric": "temperature"},
                observation_json={"mock_p_yes": 0.90},
                outcome=1.0,
            ),
        ]
        result = replay.replay(markets)
        assert result.n_markets == 1
        assert result.n_errors == 0
        assert result.brier_score is not None
        assert result.brier_score < 0.05  # (0.9 - 1.0)^2 = 0.01
        assert len(result.predictions) == 1
        assert result.predictions[0]["p_yes"] == 0.90

    def test_multiple_markets(self) -> None:
        engine = _MockEngine()
        replay = ReplayEngine(engine=engine)
        markets = [
            ReplayMarket("m1", "weather", {}, {"mock_p_yes": 0.90}, 1.0),
            ReplayMarket("m2", "weather", {}, {"mock_p_yes": 0.10}, 0.0),
            ReplayMarket("m3", "weather", {}, {"mock_p_yes": 0.80}, 1.0),
        ]
        result = replay.replay(markets)
        assert result.n_markets == 3
        assert result.brier_score is not None
        assert result.brier_score < 0.10
        assert result.log_loss is not None

    def test_spec_factory_used(self) -> None:
        engine = _MockEngine()

        def factory(spec_json: dict) -> dict:
            return {**spec_json, "factored": True}

        replay = ReplayEngine(engine=engine, spec_factory=factory)
        markets = [
            ReplayMarket("m1", "weather", {"metric": "temp"}, {"mock_p_yes": 0.50}, 1.0),
        ]
        result = replay.replay(markets)
        assert result.n_markets == 1
        assert result.n_errors == 0

    def test_spec_factory_error_counts_as_error(self) -> None:
        engine = _MockEngine()

        def bad_factory(spec_json: dict) -> dict:
            raise ValueError("bad spec")

        replay = ReplayEngine(engine=engine, spec_factory=bad_factory)
        markets = [
            ReplayMarket("m1", "weather", {}, {"mock_p_yes": 0.50}, 1.0),
        ]
        result = replay.replay(markets)
        assert result.n_markets == 1
        assert result.n_errors == 1

    def test_to_backtest_dict(self) -> None:
        engine = _MockEngine()
        replay = ReplayEngine(engine=engine)
        markets = [
            ReplayMarket("m1", "weather", {}, {"mock_p_yes": 0.50}, 1.0),
        ]
        result = replay.replay(markets)
        now = datetime.now(timezone.utc)
        bt = result.to_backtest_dict(ts_start=now, ts_end=now)
        assert bt["category"] == "weather"
        assert bt["engine_version"] == "mock_v1"
        assert bt["results_json"]["n_markets"] == 1

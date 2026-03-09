"""Tests for src.evaluation.scorer — model scoring orchestrator."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.policy import Policy
from src.db.models import ModelRun, ModelScoreDaily, Position, Resolution
from src.evaluation.scorer import ModelScorer


# ---- Helpers ----------------------------------------------------------------


def _make_resolution(
    market_id: str = "mkt-1",
    outcome: str = "YES",
    resolved_ts_utc: datetime | None = None,
) -> MagicMock:
    r = MagicMock(spec=Resolution)
    r.market_id = market_id
    r.outcome = outcome
    r.resolved_ts_utc = resolved_ts_utc or datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    return r


def _make_model_run(
    model_id: str = "openai/gpt-4.1-mini",
    market_id: str = "mkt-1",
    p_true: float = 0.70,
    confidence: float = 0.80,
    direction: str = "BUY_YES",
    ambiguity_score: float = 0.10,
    recommended_max_exposure_frac: float = 0.05,
) -> MagicMock:
    run = MagicMock(spec=ModelRun)
    run.model_id = model_id
    run.market_id = market_id
    run.parsed_json = {
        "p_true": p_true,
        "confidence": confidence,
        "direction": direction,
        "ambiguity_score": ambiguity_score,
        "recommended_max_exposure_frac": recommended_max_exposure_frac,
    }
    return run


def _make_position(
    market_id: str = "mkt-1",
    side: str = "BUY_YES",
    realized_pnl: float = 10.0,
    status: str = "closed",
) -> MagicMock:
    pos = MagicMock(spec=Position)
    pos.market_id = market_id
    pos.side = side
    pos.realized_pnl = realized_pnl
    pos.status = status
    return pos


def _make_scorer(
    resolutions: list | None = None,
    model_runs: list | None = None,
    position: MagicMock | None = None,
) -> tuple[ModelScorer, AsyncMock]:
    repo = AsyncMock()
    repo.get_resolutions_since.return_value = resolutions or []
    repo.get_model_runs_for_market.return_value = model_runs or []
    repo.get_position.return_value = position
    repo.add_model_score.return_value = 1

    policy = Policy()
    scorer = ModelScorer(repo, policy)
    return scorer, repo


# ---- ModelScorer.run_daily_scoring ------------------------------------------


class TestModelScorer:
    """ModelScorer orchestration tests."""

    async def test_no_resolutions_returns_empty(self) -> None:
        """No resolutions → no scores."""
        scorer, repo = _make_scorer(resolutions=[])
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))
        assert scores == []
        repo.add_model_score.assert_not_called()

    async def test_scores_single_model(self) -> None:
        """Single resolved market with one model run → one score stored."""
        resolution = _make_resolution(
            market_id="mkt-1", outcome="YES",
            resolved_ts_utc=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        )
        model_run = _make_model_run(
            model_id="openai/gpt-4.1-mini", market_id="mkt-1", p_true=0.70,
        )
        position = _make_position(market_id="mkt-1", realized_pnl=10.0)

        scorer, repo = _make_scorer(
            resolutions=[resolution],
            model_runs=[model_run],
            position=position,
        )
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        assert len(scores) == 1
        assert scores[0]["model_id"] == "openai/gpt-4.1-mini"
        assert scores[0]["markets_scored"] == 1
        assert scores[0]["brier_score"] is not None
        assert scores[0]["log_loss"] is not None
        repo.add_model_score.assert_called_once()

    async def test_scores_multiple_models(self) -> None:
        """Multiple models on same market → one score per model."""
        resolution = _make_resolution(market_id="mkt-1", outcome="YES")
        runs = [
            _make_model_run(model_id="openai/gpt-4.1-mini", p_true=0.70),
            _make_model_run(model_id="anthropic/claude-haiku", p_true=0.60),
        ]

        scorer, repo = _make_scorer(
            resolutions=[resolution],
            model_runs=runs,
        )
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        assert len(scores) == 2
        model_ids = {s["model_id"] for s in scores}
        assert "openai/gpt-4.1-mini" in model_ids
        assert "anthropic/claude-haiku" in model_ids

    async def test_brier_score_computed(self) -> None:
        """Brier score is computed correctly for outcome=YES."""
        resolution = _make_resolution(market_id="mkt-1", outcome="YES")
        model_run = _make_model_run(p_true=0.80)

        scorer, _ = _make_scorer(
            resolutions=[resolution],
            model_runs=[model_run],
        )
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        # Brier = (0.80 - 1)^2 = 0.04
        assert scores[0]["brier_score"] == pytest.approx(0.04)

    async def test_outcome_no(self) -> None:
        """Outcome=NO → outcome_binary=0."""
        resolution = _make_resolution(market_id="mkt-1", outcome="NO")
        model_run = _make_model_run(p_true=0.30)

        scorer, _ = _make_scorer(
            resolutions=[resolution],
            model_runs=[model_run],
        )
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        # Brier = (0.30 - 0)^2 = 0.09
        assert scores[0]["brier_score"] == pytest.approx(0.09)

    async def test_attribution_included(self) -> None:
        """When position exists, attribution metrics are computed."""
        resolution = _make_resolution(market_id="mkt-1", outcome="YES")
        model_run = _make_model_run(
            model_id="model-a", direction="BUY_YES", p_true=0.70,
        )
        position = _make_position(
            market_id="mkt-1", side="BUY_YES", realized_pnl=10.0,
        )

        scorer, _ = _make_scorer(
            resolutions=[resolution],
            model_runs=[model_run],
            position=position,
        )
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        pnl_attrib = scores[0]["pnl_attrib_json"]
        assert pnl_attrib["trades_evaluated"] == 1
        assert pnl_attrib["support_value"] == pytest.approx(1.0)

    async def test_calibration_included(self) -> None:
        """Calibration bins are included in score data."""
        resolution = _make_resolution(market_id="mkt-1", outcome="YES")
        model_run = _make_model_run(p_true=0.70)

        scorer, _ = _make_scorer(
            resolutions=[resolution],
            model_runs=[model_run],
        )
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        cal = scores[0]["calibration_json"]
        assert len(cal) >= 1
        assert "mean_predicted" in cal[0]
        assert "actual_rate" in cal[0]

    async def test_filters_resolutions_to_scoring_date(self) -> None:
        """Only resolutions within the scoring date are included."""
        # Resolution on scoring date.
        res_in = _make_resolution(
            market_id="mkt-1",
            resolved_ts_utc=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        )
        # Resolution outside scoring date.
        res_out = _make_resolution(
            market_id="mkt-2",
            resolved_ts_utc=datetime(2025, 6, 2, 12, 0, tzinfo=timezone.utc),
        )

        scorer, repo = _make_scorer(
            resolutions=[res_in, res_out],
            model_runs=[_make_model_run()],
        )
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        # Only mkt-1 should be scored (mkt-2 is outside the date).
        assert len(scores) == 1

    async def test_no_model_runs_for_market(self) -> None:
        """Resolution exists but no model runs → no scores."""
        resolution = _make_resolution(market_id="mkt-1")

        repo = AsyncMock()
        repo.get_resolutions_since.return_value = [resolution]
        repo.get_model_runs_for_market.return_value = []
        repo.get_position.return_value = None

        scorer = ModelScorer(repo, Policy())
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        assert scores == []

    async def test_defaults_to_yesterday(self) -> None:
        """When no date given, defaults to yesterday."""
        repo = AsyncMock()
        repo.get_resolutions_since.return_value = []

        scorer = ModelScorer(repo, Policy())
        await scorer.run_daily_scoring()

        # Should have been called with a datetime from yesterday.
        call_args = repo.get_resolutions_since.call_args[0][0]
        expected_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        assert call_args.date() == expected_date

    async def test_store_error_handled(self) -> None:
        """Error storing score is logged but doesn't crash."""
        resolution = _make_resolution(market_id="mkt-1")
        model_run = _make_model_run()

        repo = AsyncMock()
        repo.get_resolutions_since.return_value = [resolution]
        repo.get_model_runs_for_market.return_value = [model_run]
        repo.get_position.return_value = None
        repo.add_model_score.side_effect = Exception("DB error")

        scorer = ModelScorer(repo, Policy())
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        assert scores == []

    async def test_model_run_without_p_true_skipped(self) -> None:
        """Model run with no p_true in parsed_json is skipped."""
        resolution = _make_resolution(market_id="mkt-1")
        run = MagicMock(spec=ModelRun)
        run.model_id = "model-a"
        run.market_id = "mkt-1"
        run.parsed_json = {"confidence": 0.8}  # no p_true

        repo = AsyncMock()
        repo.get_resolutions_since.return_value = [resolution]
        repo.get_model_runs_for_market.return_value = [run]
        repo.get_position.return_value = None

        scorer = ModelScorer(repo, Policy())
        scores = await scorer.run_daily_scoring(date(2025, 6, 1))

        assert scores == []

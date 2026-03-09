"""Tests for src.evaluation.attribution — model attribution metrics."""

from __future__ import annotations

import pytest

from src.evaluation.attribution import (
    ModelAttribution,
    ProposalSummary,
    TradeOutcome,
    compute_attribution,
)


# ---- Helpers ----------------------------------------------------------------


def _prop(
    model_id: str = "model-a",
    direction: str = "BUY_YES",
    p_true: float = 0.65,
    confidence: float = 0.80,
    ambiguity_score: float = 0.10,
    recommended_max_exposure_frac: float = 0.05,
) -> ProposalSummary:
    return ProposalSummary(
        model_id=model_id,
        direction=direction,
        p_true=p_true,
        confidence=confidence,
        ambiguity_score=ambiguity_score,
        recommended_max_exposure_frac=recommended_max_exposure_frac,
    )


def _outcome(
    market_id: str = "mkt-1",
    position_side: str = "BUY_YES",
    realized_pnl: float = 10.0,
    profitable: bool = True,
    proposals: list[ProposalSummary] | None = None,
) -> TradeOutcome:
    return TradeOutcome(
        market_id=market_id,
        position_side=position_side,
        realized_pnl=realized_pnl,
        profitable=profitable,
        model_proposals=proposals or [],
    )


# ---- compute_attribution ---------------------------------------------------


class TestComputeAttribution:
    """Model attribution computation."""

    def test_no_outcomes(self) -> None:
        """No outcomes → zero values."""
        result = compute_attribution("model-a", [])
        assert result.trades_evaluated == 0
        assert result.support_value == 0.0
        assert result.dissent_value == 0.0
        assert result.veto_value == 0.0
        assert result.sizing_error == 0.0

    def test_model_not_in_outcomes(self) -> None:
        """Model has no proposals in any outcome."""
        outcomes = [
            _outcome(proposals=[_prop(model_id="model-b")]),
        ]
        result = compute_attribution("model-a", outcomes)
        assert result.trades_evaluated == 0

    def test_support_value_all_matching(self) -> None:
        """Model direction matches all profitable trades → support = 1.0."""
        outcomes = [
            _outcome(
                position_side="BUY_YES",
                profitable=True,
                proposals=[_prop(model_id="model-a", direction="BUY_YES")],
            ),
            _outcome(
                market_id="mkt-2",
                position_side="BUY_YES",
                profitable=True,
                proposals=[_prop(model_id="model-a", direction="BUY_YES")],
            ),
        ]
        result = compute_attribution("model-a", outcomes)
        assert result.support_value == pytest.approx(1.0)

    def test_support_value_none_matching(self) -> None:
        """Model direction differs on all profitable trades → support = 0.0."""
        outcomes = [
            _outcome(
                position_side="BUY_YES",
                profitable=True,
                proposals=[_prop(model_id="model-a", direction="BUY_NO")],
            ),
        ]
        result = compute_attribution("model-a", outcomes)
        assert result.support_value == pytest.approx(0.0)

    def test_dissent_value(self) -> None:
        """Model dissented on losing trade → dissent_value = 1.0."""
        outcomes = [
            _outcome(
                position_side="BUY_YES",
                profitable=False,
                realized_pnl=-10.0,
                proposals=[_prop(model_id="model-a", direction="BUY_NO")],
            ),
        ]
        result = compute_attribution("model-a", outcomes)
        assert result.dissent_value == pytest.approx(1.0)

    def test_no_dissent_when_agreed(self) -> None:
        """Model agreed with losing trade → dissent_value = 0.0."""
        outcomes = [
            _outcome(
                position_side="BUY_YES",
                profitable=False,
                realized_pnl=-10.0,
                proposals=[_prop(model_id="model-a", direction="BUY_YES")],
            ),
        ]
        result = compute_attribution("model-a", outcomes)
        assert result.dissent_value == pytest.approx(0.0)

    def test_veto_value(self) -> None:
        """Model flagged ambiguity on losing trade → veto_value = 1.0."""
        outcomes = [
            _outcome(
                position_side="BUY_YES",
                profitable=False,
                realized_pnl=-10.0,
                proposals=[_prop(model_id="model-a", ambiguity_score=0.80)],
            ),
        ]
        result = compute_attribution("model-a", outcomes, ambiguity_threshold=0.70)
        assert result.veto_value == pytest.approx(1.0)

    def test_no_veto_when_low_ambiguity(self) -> None:
        """Model didn't flag ambiguity → veto_value = 0.0."""
        outcomes = [
            _outcome(
                position_side="BUY_YES",
                profitable=False,
                realized_pnl=-10.0,
                proposals=[_prop(model_id="model-a", ambiguity_score=0.10)],
            ),
        ]
        result = compute_attribution("model-a", outcomes, ambiguity_threshold=0.70)
        assert result.veto_value == pytest.approx(0.0)

    def test_sizing_error(self) -> None:
        """Sizing error = average exposure recommendation on losing trades."""
        outcomes = [
            _outcome(
                position_side="BUY_YES",
                profitable=False,
                realized_pnl=-10.0,
                proposals=[
                    _prop(model_id="model-a", recommended_max_exposure_frac=0.08),
                ],
            ),
            _outcome(
                market_id="mkt-2",
                position_side="BUY_YES",
                profitable=False,
                realized_pnl=-5.0,
                proposals=[
                    _prop(model_id="model-a", recommended_max_exposure_frac=0.04),
                ],
            ),
        ]
        result = compute_attribution("model-a", outcomes)
        assert result.sizing_error == pytest.approx(0.06)  # (0.08 + 0.04) / 2

    def test_mixed_outcomes(self) -> None:
        """Mix of profitable and losing trades."""
        outcomes = [
            _outcome(
                market_id="mkt-1",
                position_side="BUY_YES",
                profitable=True,
                proposals=[_prop(model_id="model-a", direction="BUY_YES")],
            ),
            _outcome(
                market_id="mkt-2",
                position_side="BUY_YES",
                profitable=False,
                realized_pnl=-10.0,
                proposals=[_prop(model_id="model-a", direction="BUY_NO", ambiguity_score=0.80)],
            ),
        ]
        result = compute_attribution("model-a", outcomes, ambiguity_threshold=0.70)
        assert result.trades_evaluated == 2
        assert result.support_value == pytest.approx(1.0)  # 1/1 profitable
        assert result.dissent_value == pytest.approx(1.0)  # 1/1 losing
        assert result.veto_value == pytest.approx(1.0)  # 1/1 losing with ambiguity

    def test_no_losing_trades(self) -> None:
        """All trades profitable → dissent/veto/sizing = 0."""
        outcomes = [
            _outcome(
                profitable=True,
                proposals=[_prop(model_id="model-a", direction="BUY_YES")],
            ),
        ]
        result = compute_attribution("model-a", outcomes)
        assert result.dissent_value == 0.0
        assert result.veto_value == 0.0
        assert result.sizing_error == 0.0

"""Tests for src.aggregation.aggregator — Aggregator."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone

import pytest

from src.aggregation.aggregator import Aggregator, AggregationResult
from src.config.policy import Policy
from src.llm.schemas import ModelProposal


# ---- Helpers ----------------------------------------------------------------


def _make_policy(**overrides) -> Policy:
    return Policy(**overrides)


def _make_proposal(
    model_id: str = "probabilist",
    p_true: float = 0.60,
    confidence: float = 0.80,
    direction: str = "BUY_YES",
    rules_ambiguity: float = 0.10,
    evidence_ambiguity: float = 0.05,
) -> ModelProposal:
    return ModelProposal(
        model_id=model_id,
        run_id="run-1",
        market_id="m1",
        ts_utc=datetime(2025, 6, 1, tzinfo=timezone.utc),
        p_true=p_true,
        confidence=confidence,
        direction=direction,
        rules_ambiguity=rules_ambiguity,
        evidence_ambiguity=evidence_ambiguity,
        recommended_max_exposure_frac=0.05,
        hold_horizon_hours=48.0,
        thesis="Test thesis",
        key_risks=["risk1"],
        evidence=[],
        exit_triggers=["trigger1"],
        notes="",
    )


# ---- Aggregator tests -------------------------------------------------------


class TestAggregator:
    """Aggregation of panel proposals."""

    def test_median_weighted_mean_blend(self) -> None:
        """p_consensus uses 60% weighted mean + 40% median."""
        agg = Aggregator(_make_policy())
        proposals = [
            _make_proposal(p_true=0.60, confidence=0.80),
            _make_proposal(model_id="b", p_true=0.70, confidence=0.60),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        # Weighted mean: (0.60*0.80 + 0.70*0.60) / (0.80+0.60) = 0.90/1.40
        weighted_mean = (0.60 * 0.80 + 0.70 * 0.60) / (0.80 + 0.60)
        median_p = statistics.median([0.60, 0.70])
        expected = 0.6 * weighted_mean + 0.4 * median_p
        assert result.p_consensus == pytest.approx(expected, abs=0.001)

    def test_disagreement_stddev(self) -> None:
        agg = Aggregator(_make_policy())
        proposals = [
            _make_proposal(p_true=0.60, confidence=0.80),
            _make_proposal(model_id="b", p_true=0.70, confidence=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        expected_std = statistics.stdev([0.60, 0.70])
        assert result.disagreement == pytest.approx(expected_std, abs=0.001)

    def test_single_proposal_zero_disagreement(self) -> None:
        agg = Aggregator(_make_policy())
        proposals = [_make_proposal(p_true=0.60)]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.disagreement == 0.0
        # Single proposal: blend = 0.6 * 0.60 + 0.4 * 0.60 = 0.60
        assert result.p_consensus == pytest.approx(0.60)
        assert result.trade_allowed is True

    def test_empty_proposals_blocked(self) -> None:
        agg = Aggregator(_make_policy())
        result = agg.aggregate([], p_market=0.50)
        assert result.trade_allowed is False
        assert result.veto is True
        assert "no_proposals" in result.veto_reasons

    def test_disagreement_block(self) -> None:
        policy = _make_policy(disagreement_block_threshold=0.10)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(p_true=0.40, confidence=0.80),
            _make_proposal(model_id="b", p_true=0.70, confidence=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        # stdev([0.40, 0.70]) ≈ 0.212 > 0.10
        assert result.trade_allowed is False

    def test_model_classification_supporting(self) -> None:
        agg = Aggregator(_make_policy())
        proposals = [
            _make_proposal(model_id="a", p_true=0.65, direction="BUY_YES"),
            _make_proposal(model_id="b", p_true=0.60, direction="BUY_YES"),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert "a" in result.supporting_models
        assert "b" in result.supporting_models
        assert result.dissenting_models == []

    def test_model_classification_dissenting(self) -> None:
        agg = Aggregator(_make_policy())
        proposals = [
            _make_proposal(model_id="a", p_true=0.65, direction="BUY_YES"),
            _make_proposal(model_id="b", p_true=0.60, direction="BUY_NO"),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert "a" in result.supporting_models
        assert "b" in result.dissenting_models

    def test_confidence_in_result(self) -> None:
        agg = Aggregator(_make_policy())
        proposals = [
            _make_proposal(confidence=0.90),
            _make_proposal(model_id="b", confidence=0.70),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        expected = (0.90 * 0.90 + 0.70 * 0.70) / (0.90 + 0.70)
        assert result.confidence == pytest.approx(expected, abs=0.001)

    def test_veto_score_in_result(self) -> None:
        """AggregationResult includes veto_score field."""
        agg = Aggregator(_make_policy())
        proposals = [_make_proposal(p_true=0.60)]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto_score == 0.0


# ---- Weighted veto quorum tests (new) ----------------------------------------


class TestWeightedVetoQuorum:
    """Weighted quorum-based veto logic with rules_ambiguity."""

    def test_two_normal_models_veto(self) -> None:
        """2 eligible models with high rules_ambiguity → veto (score >= quorum=2)."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="rules_lawyer", rules_ambiguity=0.80),
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80),
            _make_proposal(model_id="c", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is True
        assert result.trade_allowed is False
        assert len(result.vetoing_models) >= 1

    def test_single_model_below_quorum_no_veto(self) -> None:
        """1 eligible model (non-rules_lawyer) with high ambiguity → no veto (score=1.0 < quorum=2)."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10),
            _make_proposal(model_id="sanity_checker", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False

    def test_rules_lawyer_weight_200(self) -> None:
        """rules_lawyer alone → score=2.0 >= quorum=2 → veto (authoritative)."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="rules_lawyer", rules_ambiguity=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is True
        assert result.veto_score == pytest.approx(2.0)
        assert result.trade_allowed is False

    def test_rules_lawyer_plus_escalation_triggers_veto(self) -> None:
        """rules_lawyer + escalation_anthropic → veto (after dedup, score=2.0 >= quorum=2)."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="rules_lawyer", rules_ambiguity=0.80),
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80),
            _make_proposal(model_id="c", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is True
        assert "rules_lawyer" in result.vetoing_models

    def test_quorum_1_single_model_vetoes(self) -> None:
        """Quorum=1 means a single eligible model can veto."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=1)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="rules_lawyer", rules_ambiguity=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is True
        assert len(result.vetoing_models) == 1

    def test_raised_threshold_reduces_vetoes(self) -> None:
        """With threshold=0.85, rules_ambiguity=0.80 no longer triggers."""
        policy = _make_policy(ambiguity_veto_threshold=0.85, veto_quorum=1)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="rules_lawyer", rules_ambiguity=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.trade_allowed is True


# ---- Veto eligibility tests --------------------------------------------------


class TestVetoEligibility:
    """Only rules_lawyer and escalation_anthropic contribute to veto_score."""

    def test_probabilist_ambiguity_ignored_in_veto(self) -> None:
        """probabilist with high rules_ambiguity does NOT contribute to veto_score."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=1)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="probabilist", rules_ambiguity=0.90),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.veto_score == 0.0

    def test_sanity_checker_ambiguity_ignored_in_veto(self) -> None:
        """sanity_checker with high rules_ambiguity does NOT contribute to veto_score."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=1)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="sanity_checker", rules_ambiguity=0.90),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.veto_score == 0.0

    def test_x_signals_ambiguity_ignored_in_veto(self) -> None:
        """x_signals with high rules_ambiguity does NOT contribute to veto_score."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=1)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="x_signals", rules_ambiguity=0.90),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.veto_score == 0.0

    def test_only_eligible_agents_counted(self) -> None:
        """5-agent panel, only escalation_anthropic's 0.90 counts (no rules_lawyer)."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.90),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.90),
            _make_proposal(model_id="sanity_checker", rules_ambiguity=0.90),
            _make_proposal(model_id="x_signals", rules_ambiguity=0.90),
            _make_proposal(model_id="evidence_hunter", rules_ambiguity=0.90),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        # Only escalation_anthropic is eligible and counted (weight=1.0 < quorum=2)
        assert result.veto is False
        assert result.veto_score == pytest.approx(1.0)
        assert len(result.vetoing_models) == 1
        assert "escalation_anthropic" in result.vetoing_models


# ---- Partial ambiguity tests -------------------------------------------------


class TestPartialAmbiguity:
    """Partial rules ambiguity → confidence downgrade."""

    def test_partial_ambiguity_halves_confidence(self) -> None:
        """Single eligible model (escalation) with quorum=2 → confidence * 0.5."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80, confidence=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10, confidence=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        # Confidence should be halved due to partial ambiguity.
        full_confidence = (0.80 * 0.80 + 0.80 * 0.80) / (0.80 + 0.80)
        assert result.confidence == pytest.approx(full_confidence * 0.5, abs=0.01)

    def test_no_partial_when_no_vetoing_models(self) -> None:
        """When no models flag rules_ambiguity, no partial downgrade."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="a", rules_ambiguity=0.10, confidence=0.80),
            _make_proposal(model_id="b", rules_ambiguity=0.10, confidence=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.veto_score == 0.0
        # No partial penalty.
        expected_conf = (0.80 * 0.80 + 0.80 * 0.80) / (0.80 + 0.80)
        assert result.confidence == pytest.approx(expected_conf, abs=0.01)


# ---- Evidence ambiguity tests ------------------------------------------------


class TestEvidenceAmbiguity:
    """Evidence ambiguity → confidence penalty (never vetoes)."""

    def test_evidence_ambiguity_reduces_confidence(self) -> None:
        """High evidence_ambiguity reduces confidence but doesn't veto."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="a", evidence_ambiguity=0.80, confidence=0.80),
            _make_proposal(model_id="b", evidence_ambiguity=0.10, confidence=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.trade_allowed is True
        # 1 evidence-only model → confidence *= max(0.5, 1.0 - 0.15*1) = 0.85
        base_confidence = (0.80 * 0.80 + 0.80 * 0.80) / (0.80 + 0.80)
        assert result.confidence < base_confidence

    def test_multiple_evidence_ambiguity_reduces_more(self) -> None:
        """Multiple models with evidence_ambiguity reduce confidence further."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=3)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="a", evidence_ambiguity=0.80, confidence=0.80),
            _make_proposal(model_id="b", evidence_ambiguity=0.80, confidence=0.80),
            _make_proposal(model_id="c", evidence_ambiguity=0.10, confidence=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        # 2 evidence-only models → confidence *= max(0.5, 1.0 - 0.15*2) = 0.70
        base_confidence = (0.80 * 0.80 + 0.80 * 0.80 + 0.80 * 0.80) / (0.80 * 3)
        assert result.confidence < base_confidence * 0.71

    def test_evidence_ambiguity_never_vetoes(self) -> None:
        """Even all models with high evidence_ambiguity don't trigger veto."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=1)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="a", evidence_ambiguity=0.90, rules_ambiguity=0.10),
            _make_proposal(model_id="b", evidence_ambiguity=0.90, rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.trade_allowed is True

    def test_independent_dimensions(self) -> None:
        """Clear rules but high evidence_ambiguity → no veto, reduced confidence."""
        policy = _make_policy(ambiguity_veto_threshold=0.85, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(
                model_id="a",
                rules_ambiguity=0.10,
                evidence_ambiguity=0.90,
                confidence=0.80,
            ),
            _make_proposal(
                model_id="b",
                rules_ambiguity=0.10,
                evidence_ambiguity=0.90,
                confidence=0.80,
            ),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.trade_allowed is True
        # Both models have high evidence ambiguity → confidence penalized.
        base_conf = (0.80 * 0.80 + 0.80 * 0.80) / (0.80 + 0.80)
        assert result.confidence < base_conf


# ---- Vetoing model exclusion from classification ----------------------------


class TestVetoModelClassification:
    """Models in vetoing_models are excluded from supporting/dissenting."""

    def test_vetoing_excluded_from_supporting(self) -> None:
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=1)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(
                model_id="rules_lawyer", p_true=0.60, rules_ambiguity=0.80, direction="BUY_YES"
            ),
            _make_proposal(
                model_id="probabilist", p_true=0.65, rules_ambiguity=0.10, direction="BUY_YES"
            ),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert "rules_lawyer" in result.vetoing_models
        assert "rules_lawyer" not in result.supporting_models
        assert "rules_lawyer" not in result.dissenting_models


# ---- Confidence gate tests --------------------------------------------------


class TestConfidenceGate:
    """Hard confidence block in aggregation."""

    def test_low_confidence_blocks_trade(self) -> None:
        """Confidence below min_confidence_hard → trade_allowed=False."""
        policy = _make_policy(min_confidence_hard=0.25)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="a", confidence=0.20, p_true=0.60),
            _make_proposal(model_id="b", confidence=0.15, p_true=0.65),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.trade_allowed is False
        assert any("hard min" in r for r in result.veto_reasons)

    def test_confidence_above_hard_min_allows_trade(self) -> None:
        """Confidence above min_confidence_hard → trade_allowed=True."""
        policy = _make_policy(min_confidence_hard=0.25)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="a", confidence=0.60, p_true=0.60),
            _make_proposal(model_id="b", confidence=0.50, p_true=0.65),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.trade_allowed is True

    def test_confidence_at_boundary_allows_trade(self) -> None:
        """Confidence exactly at min_confidence_hard → trade_allowed=True."""
        policy = _make_policy(min_confidence_hard=0.25)
        agg = Aggregator(policy)
        # All confidence=0.25 → weighted confidence = 0.25 (at boundary).
        proposals = [
            _make_proposal(model_id="a", confidence=0.25, p_true=0.60),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.trade_allowed is True

    def test_evidence_ambiguity_can_push_below_hard_min(self) -> None:
        """Evidence ambiguity penalty reduces confidence below hard min → blocked."""
        policy = _make_policy(
            min_confidence_hard=0.25,
            ambiguity_veto_threshold=0.70,
            veto_quorum=3,
        )
        agg = Aggregator(policy)
        # Start with moderate confidence that gets penalised below 0.25.
        proposals = [
            _make_proposal(
                model_id="a", confidence=0.30, p_true=0.60,
                evidence_ambiguity=0.80,
            ),
            _make_proposal(
                model_id="b", confidence=0.30, p_true=0.65,
                evidence_ambiguity=0.80,
            ),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        # 2 evidence ambiguity models → confidence *= max(0.5, 1-0.30) = 0.70
        # 0.30 * 0.70 = 0.21 < 0.25
        assert result.trade_allowed is False
        assert any("hard min" in r for r in result.veto_reasons)


# ---- Escalation deduplication tests -----------------------------------------


class TestEscalationDeduplication:
    """Escalation_anthropic dedup when rules_lawyer already voted."""

    def test_escalation_deduped_when_rules_lawyer_present(self) -> None:
        """Both flag ambiguity → escalation deduped, score = 2.0 (not 3.0)."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="rules_lawyer", rules_ambiguity=0.80),
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is True
        assert result.veto_score == pytest.approx(2.0)
        assert "escalation_anthropic" not in result.vetoing_models

    def test_escalation_not_deduped_when_rules_lawyer_absent(self) -> None:
        """Only escalation flags → no dedup, score = 1.0."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.veto_score == pytest.approx(1.0)
        assert "escalation_anthropic" in result.vetoing_models

    def test_dedup_reason_added(self) -> None:
        """Dedup reason string appears in veto_reasons."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="rules_lawyer", rules_ambiguity=0.80),
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert any("de-duplicated" in r for r in result.veto_reasons)

    def test_escalation_alone_partial_penalty(self) -> None:
        """Escalation alone → 1.0 < quorum=2 → partial, confidence halved."""
        policy = _make_policy(ambiguity_veto_threshold=0.70, veto_quorum=2)
        agg = Aggregator(policy)
        proposals = [
            _make_proposal(model_id="escalation_anthropic", rules_ambiguity=0.80, confidence=0.80),
            _make_proposal(model_id="probabilist", rules_ambiguity=0.10, confidence=0.80),
        ]
        result = agg.aggregate(proposals, p_market=0.50)
        assert result.veto is False
        assert result.veto_score == pytest.approx(1.0)
        full_confidence = (0.80 * 0.80 + 0.80 * 0.80) / (0.80 + 0.80)
        assert result.confidence == pytest.approx(full_confidence * 0.5, abs=0.01)
        assert any("partial_rules_ambiguity" in r for r in result.veto_reasons)

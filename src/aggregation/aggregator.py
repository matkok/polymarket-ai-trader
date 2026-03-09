"""Deterministic aggregation of LLM panel proposals."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.config.policy import Policy
from src.llm.schemas import ModelProposal


# Rules lawyer gets extra weight in the veto quorum.
VETO_WEIGHTS: dict[str, float] = {"rules_lawyer": 2.0}

# Only these agents can contribute to the rules-ambiguity veto quorum.
VETO_ELIGIBLE_AGENTS: frozenset[str] = frozenset({"rules_lawyer", "escalation_anthropic"})


@dataclass
class AggregationResult:
    """Result of aggregating multiple model proposals."""

    p_consensus: float
    confidence: float
    disagreement: float
    veto: bool
    veto_score: float = 0.0
    veto_reasons: list[str] = field(default_factory=list)
    trade_allowed: bool = True
    supporting_models: list[str] = field(default_factory=list)
    dissenting_models: list[str] = field(default_factory=list)
    vetoing_models: list[str] = field(default_factory=list)


class Aggregator:
    """Aggregates panel proposals into a consensus using deterministic logic."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def aggregate(
        self,
        proposals: list[ModelProposal],
        p_market: float,
    ) -> AggregationResult:
        """Compute consensus from panel proposals.

        Uses a two-layer voting system:
        - Layer 1: Probability vote (soft) — median + weighted mean blend
        - Layer 2: Rules ambiguity vote (hard) — weighted quorum
        """
        if not proposals:
            return AggregationResult(
                p_consensus=p_market,
                confidence=0.0,
                disagreement=1.0,
                veto=True,
                veto_reasons=["no_proposals"],
                trade_allowed=False,
            )

        # Reliability weight is 1.0 for all models (updated in M5).
        reliability_weight = 1.0

        weights = [p.confidence * reliability_weight for p in proposals]
        total_weight = sum(weights)

        if total_weight == 0:
            return AggregationResult(
                p_consensus=p_market,
                confidence=0.0,
                disagreement=1.0,
                veto=True,
                veto_reasons=["zero_total_weight"],
                trade_allowed=False,
            )

        # ---- Layer 1: Probability vote (soft) ----

        # Weighted mean of p_true.
        weighted_mean = sum(
            p.p_true * w for p, w in zip(proposals, weights)
        ) / total_weight

        # Median (robust against outliers).
        median_p = statistics.median(p.p_true for p in proposals)

        # Blend: 60% weighted mean, 40% median.
        p_consensus = 0.6 * weighted_mean + 0.4 * median_p

        # Weighted mean of confidence.
        confidence = sum(
            p.confidence * w for p, w in zip(proposals, weights)
        ) / total_weight

        # Disagreement = std dev of p_true.
        if len(proposals) > 1:
            disagreement = statistics.stdev(p.p_true for p in proposals)
        else:
            disagreement = 0.0

        # ---- Layer 2: Rules ambiguity vote (hard) ----

        veto = False
        veto_score = 0.0
        veto_reasons: list[str] = []
        vetoing_models: list[str] = []

        for p in proposals:
            if p.model_id not in VETO_ELIGIBLE_AGENTS:
                continue
            if p.rules_ambiguity >= self.policy.ambiguity_veto_threshold:
                weight = VETO_WEIGHTS.get(p.model_id, 1.0)
                veto_score += weight
                vetoing_models.append(p.model_id)
                veto_reasons.append(
                    f"{p.model_id}: rules_ambiguity {p.rules_ambiguity:.2f} "
                    f">= {self.policy.ambiguity_veto_threshold} "
                    f"(weight={weight:.2f})"
                )

        # De-duplicate: if both rules_lawyer and escalation_anthropic voted,
        # remove escalation's contribution (same model family, correlated).
        if "rules_lawyer" in vetoing_models and "escalation_anthropic" in vetoing_models:
            esc_weight = VETO_WEIGHTS.get("escalation_anthropic", 1.0)
            veto_score -= esc_weight
            vetoing_models.remove("escalation_anthropic")
            veto_reasons.append(
                "escalation_anthropic de-duplicated (same family as rules_lawyer)"
            )

        # Quorum check: weighted veto score must meet quorum.
        if veto_score >= self.policy.veto_quorum:
            veto = True

        # Partial ambiguity: some models flagged but below quorum → downgrade.
        if 0 < veto_score < self.policy.veto_quorum:
            confidence *= 0.5
            veto_reasons.append(
                f"partial_rules_ambiguity: score={veto_score:.2f}"
            )

        # ---- Evidence ambiguity → confidence penalty (never vetoes) ----

        evidence_ambiguity_count = sum(
            1 for p in proposals
            if p.evidence_ambiguity >= self.policy.ambiguity_veto_threshold
        )
        if evidence_ambiguity_count > 0:
            confidence *= max(0.5, 1.0 - 0.15 * evidence_ambiguity_count)

        # ---- Trade allowed? ----

        trade_allowed = True

        # Confidence gate (hard block).
        if confidence < self.policy.min_confidence_hard:
            trade_allowed = False
            veto_reasons.append(
                f"confidence {confidence:.3f} < hard min {self.policy.min_confidence_hard}"
            )

        if veto:
            trade_allowed = False
        if disagreement >= self.policy.disagreement_block_threshold:
            trade_allowed = False
            veto_reasons.append(
                f"disagreement {disagreement:.4f} "
                f">= {self.policy.disagreement_block_threshold}"
            )

        # Classify models.
        consensus_direction = "BUY_YES" if p_consensus > p_market else "BUY_NO"
        supporting: list[str] = []
        dissenting: list[str] = []
        for p in proposals:
            if p.model_id in vetoing_models:
                continue
            if p.direction == consensus_direction:
                supporting.append(p.model_id)
            else:
                dissenting.append(p.model_id)

        return AggregationResult(
            p_consensus=p_consensus,
            confidence=confidence,
            disagreement=disagreement,
            veto=veto,
            veto_score=veto_score,
            veto_reasons=veto_reasons,
            trade_allowed=trade_allowed,
            supporting_models=supporting,
            dissenting_models=dissenting,
            vetoing_models=vetoing_models,
        )

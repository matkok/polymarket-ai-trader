"""Deterministic triage scoring and guardrails for market signals."""

from __future__ import annotations

import structlog

from src.config.policy import Policy
from src.signals.schemas import MarketSignalBundle, TriageResult

logger = structlog.get_logger(__name__)


class TriageScorer:
    """Deterministic triage scorer using weighted signal components.

    Weights sum to 1.0:
        odds_move_6h:      0.25
        volume_surge:       0.15
        evidence_freshness: 0.20
        google_trends_spike:0.15
        wikipedia_spike:    0.10
        spread_tightness:   0.15

    Missing signals contribute 0.0.
    """

    W_ODDS_MOVE = 0.25
    W_VOLUME_SURGE = 0.15
    W_EVIDENCE_FRESHNESS = 0.20
    W_TRENDS_SPIKE = 0.15
    W_WIKI_SPIKE = 0.10
    W_SPREAD_TIGHTNESS = 0.15

    def __init__(
        self, policy: Policy, *, trends_enabled: bool = True, wiki_enabled: bool = True
    ) -> None:
        self.policy = policy
        # Compute active weight sum — disabled sources don't penalize scores.
        self._active_weight = (
            self.W_ODDS_MOVE
            + self.W_VOLUME_SURGE
            + self.W_EVIDENCE_FRESHNESS
            + self.W_SPREAD_TIGHTNESS
            + (self.W_TRENDS_SPIKE if trends_enabled else 0.0)
            + (self.W_WIKI_SPIKE if wiki_enabled else 0.0)
        )

    def score(self, bundle: MarketSignalBundle) -> TriageResult:
        """Compute triage score and guardrails for a market signal bundle.

        Returns:
            TriageResult with score, reasons, should_panel flag, and guardrail_flags.
        """
        reasons: list[str] = []
        guardrail_flags: list[str] = []

        # --- Component scores ---

        # 1. Odds move (6h)
        odds_score = 0.0
        if bundle.microstructure and bundle.microstructure.odds_move_6h is not None:
            move = bundle.microstructure.odds_move_6h
            odds_score = min(move / self.policy.triage_odds_move_threshold, 1.0)
            reasons.append(f"odds_move_6h={move:.3f} -> {odds_score:.2f}")

        # 2. Volume surge
        volume_score = 0.0
        if bundle.microstructure and bundle.microstructure.volume_ratio_24h is not None:
            ratio = bundle.microstructure.volume_ratio_24h
            volume_score = min(
                max(ratio - 1.0, 0.0) / self.policy.triage_volume_surge_threshold, 1.0
            )
            reasons.append(f"volume_ratio={ratio:.2f} -> {volume_score:.2f}")

        # 3. Evidence freshness
        evidence_score = 0.0
        if bundle.evidence_freshness:
            ef = bundle.evidence_freshness
            credible_part = 0.6 * min(ef.credible_evidence_6h / 3.0, 1.0)
            count_part = 0.4 * min(ef.evidence_count_24h / 5.0, 1.0)
            evidence_score = credible_part + count_part
            reasons.append(
                f"evidence(credible_6h={ef.credible_evidence_6h}, "
                f"count_24h={ef.evidence_count_24h}) -> {evidence_score:.2f}"
            )

        # 4. Google Trends spike
        trends_score = 0.0
        if bundle.google_trends and bundle.google_trends.spike_score > 0:
            spike = bundle.google_trends.spike_score
            trends_score = min(
                max(spike - 1.0, 0.0) / self.policy.triage_trends_spike_threshold, 1.0
            )
            reasons.append(f"trends_spike={spike:.2f} -> {trends_score:.2f}")

        # 5. Wikipedia spike
        wiki_score = 0.0
        if bundle.wikipedia and bundle.wikipedia.spike_score > 0:
            spike = bundle.wikipedia.spike_score
            wiki_score = min(
                max(spike - 1.0, 0.0) / self.policy.triage_wiki_spike_threshold, 1.0
            )
            reasons.append(f"wiki_spike={spike:.2f} -> {wiki_score:.2f}")

        # 6. Spread tightness (tighter = higher)
        spread_score = 0.0
        if bundle.microstructure and bundle.microstructure.spread_current is not None:
            spread = bundle.microstructure.spread_current
            spread_score = 1.0 - min(spread / self.policy.triage_max_spread, 1.0)
            reasons.append(f"spread={spread:.3f} -> {spread_score:.2f}")

        # --- Weighted composite ---
        raw_score = (
            self.W_ODDS_MOVE * odds_score
            + self.W_VOLUME_SURGE * volume_score
            + self.W_EVIDENCE_FRESHNESS * evidence_score
            + self.W_TRENDS_SPIKE * trends_score
            + self.W_WIKI_SPIKE * wiki_score
            + self.W_SPREAD_TIGHTNESS * spread_score
        )
        # Normalize by active weight so disabled sources don't penalize.
        triage_score = raw_score / self._active_weight if self._active_weight > 0 else 0.0

        should_panel = triage_score >= self.policy.triage_panel_threshold

        # --- Guardrails ---

        # Wide spread
        if bundle.microstructure and bundle.microstructure.spread_current is not None:
            if bundle.microstructure.spread_current >= self.policy.triage_wide_spread_threshold:
                guardrail_flags.append("wide_spread")

        # Spread widening
        if bundle.microstructure and bundle.microstructure.spread_widening is not None:
            if bundle.microstructure.spread_widening >= self.policy.triage_spread_widening_threshold:
                guardrail_flags.append("spread_widening")

        # Social-only evidence (evidence exists but none from RSS = credible)
        if bundle.evidence_freshness:
            ef = bundle.evidence_freshness
            has_evidence = ef.evidence_count_6h > 0 or ef.evidence_count_24h > 0
            no_credible = ef.credible_evidence_6h == 0
            if has_evidence and no_credible:
                guardrail_flags.append("social_only_no_credible_source")

        return TriageResult(
            triage_score=triage_score,
            reasons=reasons,
            should_panel=should_panel,
            guardrail_flags=guardrail_flags,
        )

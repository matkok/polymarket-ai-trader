"""Deterministic ranking of Polymarket markets for trading consideration.

The selector filters and ranks markets based on policy constraints.  All
scoring is deterministic and reproducible given the same inputs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

from src.config.policy import Policy
from src.db.models import Market, MarketSnapshot


@dataclass
class CandidateScore:
    """Scored candidate market."""

    market_id: str
    question: str
    category: str | None
    score: float  # composite score, higher = better
    liquidity: float
    volume: float
    hours_to_resolution: float
    mid_price: float | None
    reasons: list[str] = field(default_factory=list)  # why it scored high/low


class CandidateSelector:
    """Deterministic candidate ranking using policy constraints."""

    # Scoring weights (must sum to 1.0).
    _W_LIQUIDITY: float = 0.25
    _W_VOLUME: float = 0.20
    _W_TIME: float = 0.15
    _W_PRICE: float = 0.20
    _W_DIVERSITY: float = 0.20

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.logger = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        markets: list[Market],
        snapshots: dict[str, MarketSnapshot],
        open_positions: set[str],
        now: datetime | None = None,
    ) -> list[CandidateScore]:
        """Filter and rank markets.  Returns top candidates sorted by score descending.

        Filters applied:
        1. Market must be active
        2. Liquidity >= min_liquidity_eur
        3. Volume >= min_volume
        4. Time to resolution within [min_hours, max_hours]
        5. Must not already have an open position (novelty)
        6. Must have a valid snapshot with prices

        Scoring (all factors normalised 0-1, then weighted):
        - Liquidity score: log-scaled, higher = better (weight 0.25)
        - Volume score: log-scaled, higher = better (weight 0.20)
        - Time score: prefer mid-range resolution times (weight 0.15)
        - Price score: prefer prices near 0.5 (highest uncertainty) (weight 0.20)
        - Category diversity: bonus for underrepresented categories (weight 0.20)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        candidates = self._filter(markets, snapshots, open_positions, now)
        scored = self._score(candidates, now)
        scored.sort(key=lambda c: c.score, reverse=True)

        top = scored[: self.policy.max_candidates_per_cycle]
        self.logger.info(
            "candidate_selection_complete",
            total_scored=len(scored),
            returned=len(top),
        )
        return top

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter(
        self,
        markets: list[Market],
        snapshots: dict[str, MarketSnapshot],
        open_positions: set[str],
        now: datetime,
    ) -> list[tuple[Market, MarketSnapshot, float]]:
        """Apply all policy filters.  Returns (market, snapshot, hours_to_resolution)."""
        total = len(markets)

        # 1. Active status
        active = [m for m in markets if m.status == "active"]
        self.logger.debug("filter_active", before=total, after=len(active))

        # 2. Must have resolution_time_utc
        with_resolution = [m for m in active if m.resolution_time_utc is not None]
        self.logger.debug(
            "filter_resolution_time", before=len(active), after=len(with_resolution)
        )

        # 3. Must have a snapshot
        with_snapshot: list[tuple[Market, MarketSnapshot]] = []
        for m in with_resolution:
            snap = snapshots.get(m.market_id)
            if snap is not None:
                with_snapshot.append((m, snap))
        self.logger.debug(
            "filter_has_snapshot", before=len(with_resolution), after=len(with_snapshot)
        )

        # 4. Snapshot must have valid prices (mid is not None)
        with_prices = [(m, s) for m, s in with_snapshot if s.mid is not None]
        self.logger.debug(
            "filter_valid_prices", before=len(with_snapshot), after=len(with_prices)
        )

        # 4b. Must have meaningful rules_text
        min_rules_len = self.policy.min_rules_text_length
        with_rules = [
            (m, s) for m, s in with_prices
            if m.rules_text is not None and len(m.rules_text.strip()) >= min_rules_len
        ]
        self.logger.debug("filter_rules_text", before=len(with_prices), after=len(with_rules))

        # 5. Liquidity >= min_liquidity_eur
        with_liquidity = [
            (m, s)
            for m, s in with_rules
            if s.liquidity is not None and s.liquidity >= self.policy.min_liquidity_eur
        ]
        self.logger.debug(
            "filter_liquidity", before=len(with_prices), after=len(with_liquidity)
        )

        # 6. Volume >= min_volume
        with_volume = [
            (m, s)
            for m, s in with_liquidity
            if s.volume is not None and s.volume >= self.policy.min_volume
        ]
        self.logger.debug(
            "filter_volume", before=len(with_liquidity), after=len(with_volume)
        )

        # 7. Time to resolution within [min_hours, max_hours]
        min_h = self.policy.min_hours_to_resolution
        max_h = self.policy.max_hours_to_resolution
        with_time: list[tuple[Market, MarketSnapshot, float]] = []
        for m, s in with_volume:
            hours = (m.resolution_time_utc - now).total_seconds() / 3600  # type: ignore[operator]
            if min_h <= hours <= max_h:
                with_time.append((m, s, hours))
        self.logger.debug(
            "filter_time_to_resolution", before=len(with_volume), after=len(with_time)
        )

        # 8. Must not already have an open position
        novel = [(m, s, h) for m, s, h in with_time if m.market_id not in open_positions]
        self.logger.debug(
            "filter_open_positions", before=len(with_time), after=len(novel)
        )

        self.logger.info(
            "filtering_complete", input_markets=total, passed=len(novel)
        )
        return novel

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        candidates: list[tuple[Market, MarketSnapshot, float]],
        now: datetime,
    ) -> list[CandidateScore]:
        """Compute composite scores for filtered candidates."""
        min_h = self.policy.min_hours_to_resolution
        max_h = self.policy.max_hours_to_resolution
        midpoint = (min_h + max_h) / 2

        # Category counts are tracked incrementally for the diversity bonus.
        # We process candidates in a stable order (input order) and compute
        # diversity relative to previously scored candidates.
        category_counts: Counter[str | None] = Counter()
        scored: list[CandidateScore] = []

        for market, snap, hours in candidates:
            reasons: list[str] = []

            # Liquidity score (log-scaled).
            liq = snap.liquidity or 0.0
            liq_score = min(
                1.0,
                math.log1p(liq) / math.log1p(self.policy.min_liquidity_eur * 10),
            )
            reasons.append(f"liquidity={liq_score:.2f}")

            # Volume score (log-scaled).
            vol = snap.volume or 0.0
            vol_score = min(
                1.0,
                math.log1p(vol) / math.log1p(self.policy.min_volume * 10),
            )
            reasons.append(f"volume={vol_score:.2f}")

            # Time score (prefer mid-range).
            if midpoint > 0:
                time_score = max(0.0, 1.0 - abs(hours - midpoint) / midpoint)
            else:
                time_score = 0.0
            reasons.append(f"time={time_score:.2f}")

            # Price score (prefer mid near 0.5).
            mid = snap.mid
            if mid is not None:
                price_score = max(0.0, 1.0 - 2 * abs(mid - 0.5))
            else:
                price_score = 0.0
            reasons.append(f"price={price_score:.2f}")

            # Category diversity (penalise overrepresented categories).
            same_count = category_counts[market.category]
            diversity_score = 1.0 / (1 + same_count)
            reasons.append(f"diversity={diversity_score:.2f}")

            composite = (
                self._W_LIQUIDITY * liq_score
                + self._W_VOLUME * vol_score
                + self._W_TIME * time_score
                + self._W_PRICE * price_score
                + self._W_DIVERSITY * diversity_score
            )

            scored.append(
                CandidateScore(
                    market_id=market.market_id,
                    question=market.question,
                    category=market.category,
                    score=composite,
                    liquidity=liq,
                    volume=vol,
                    hours_to_resolution=hours,
                    mid_price=mid,
                    reasons=reasons,
                )
            )

            category_counts[market.category] += 1

        return scored

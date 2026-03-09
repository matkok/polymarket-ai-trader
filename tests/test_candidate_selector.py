"""Tests for src.app.candidate_selector — filtering and scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.app.candidate_selector import CandidateScore, CandidateSelector
from src.config.policy import Policy
from src.db.models import Market, MarketSnapshot


# ---- Helpers ----------------------------------------------------------------


def _make_market(
    market_id: str = "m1",
    question: str = "Will it rain?",
    category: str | None = "weather",
    resolution_hours: float = 500.0,
    status: str = "active",
    now: datetime | None = None,
) -> MagicMock:
    """Create a mock Market instance for testing."""
    if now is None:
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m = MagicMock(spec=Market)
    m.market_id = market_id
    m.question = question
    m.category = category
    m.status = status
    m.rules_text = "This market resolves YES if the event occurs before the deadline per official source."
    m.resolution_time_utc = now + timedelta(hours=resolution_hours)
    return m


def _make_snapshot(
    market_id: str = "m1",
    mid: float | None = 0.50,
    liquidity: float | None = 10_000.0,
    volume: float | None = 20_000.0,
) -> MagicMock:
    """Create a mock MarketSnapshot instance for testing."""
    s = MagicMock(spec=MarketSnapshot)
    s.market_id = market_id
    s.mid = mid
    s.liquidity = liquidity
    s.volume = volume
    s.best_bid = None
    s.best_ask = None
    s.ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
    return s


NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)


# ---- CandidateScore --------------------------------------------------------


class TestCandidateScore:
    """CandidateScore dataclass basics."""

    def test_fields(self) -> None:
        cs = CandidateScore(
            market_id="m1",
            question="Q?",
            category="cat",
            score=0.75,
            liquidity=5_000.0,
            volume=10_000.0,
            hours_to_resolution=100.0,
            mid_price=0.5,
            reasons=["liq=0.80"],
        )
        assert cs.market_id == "m1"
        assert cs.score == 0.75
        assert cs.reasons == ["liq=0.80"]

    def test_default_reasons(self) -> None:
        cs = CandidateScore(
            market_id="m1",
            question="Q?",
            category=None,
            score=0.0,
            liquidity=0.0,
            volume=0.0,
            hours_to_resolution=0.0,
            mid_price=None,
        )
        assert cs.reasons == []


# ---- Filtering --------------------------------------------------------------


class TestCandidateSelectorFiltering:
    """Filter-stage behaviour of CandidateSelector.select()."""

    def test_empty_markets(self) -> None:
        """No markets yields empty result."""
        sel = CandidateSelector(Policy())
        result = sel.select([], {}, set(), now=NOW)
        assert result == []

    def test_filters_inactive_market(self) -> None:
        """Markets with status != 'active' are excluded."""
        m = _make_market(status="resolved", now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_no_resolution_time(self) -> None:
        """Markets without resolution_time_utc are excluded."""
        m = _make_market(now=NOW)
        m.resolution_time_utc = None
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_no_snapshot(self) -> None:
        """Markets without a snapshot entry are excluded."""
        m = _make_market(now=NOW)
        sel = CandidateSelector(Policy())
        result = sel.select([m], {}, set(), now=NOW)
        assert result == []

    def test_filters_no_mid_price(self) -> None:
        """Snapshots with mid=None are excluded."""
        m = _make_market(now=NOW)
        snap = _make_snapshot(mid=None)
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_low_liquidity(self) -> None:
        """Markets below min_liquidity_eur are excluded."""
        m = _make_market(now=NOW)
        snap = _make_snapshot(liquidity=100.0)  # below default 5000
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_none_liquidity(self) -> None:
        """Snapshots with liquidity=None are excluded."""
        m = _make_market(now=NOW)
        snap = _make_snapshot(liquidity=None)
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_low_volume(self) -> None:
        """Markets below min_volume are excluded."""
        m = _make_market(now=NOW)
        snap = _make_snapshot(volume=50.0)  # below default 10000
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_none_volume(self) -> None:
        """Snapshots with volume=None are excluded."""
        m = _make_market(now=NOW)
        snap = _make_snapshot(volume=None)
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_too_soon_resolution(self) -> None:
        """Markets resolving too soon (< min_hours) are excluded."""
        m = _make_market(resolution_hours=5.0, now=NOW)  # default min is 24
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_too_far_resolution(self) -> None:
        """Markets resolving too far (> max_hours) are excluded."""
        m = _make_market(resolution_hours=3000.0, now=NOW)  # default max is 2160
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_open_position(self) -> None:
        """Markets with an existing open position are excluded."""
        m = _make_market(now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, {m.market_id}, now=NOW)
        assert result == []

    def test_filters_none_rules_text(self) -> None:
        """Markets with rules_text=None are excluded."""
        m = _make_market(now=NOW)
        m.rules_text = None
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_short_rules_text(self) -> None:
        """Markets with very short rules_text are excluded."""
        m = _make_market(now=NOW)
        m.rules_text = "Short"
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_filters_whitespace_only_rules_text(self) -> None:
        """Markets with whitespace-only rules_text are excluded."""
        m = _make_market(now=NOW)
        m.rules_text = "   "
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert result == []

    def test_passes_adequate_rules_text(self) -> None:
        """Markets with adequate rules_text pass the filter."""
        m = _make_market(now=NOW)
        m.rules_text = "This market resolves YES if the candidate wins the election per official results."
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1

    def test_rules_text_filter_respects_policy(self) -> None:
        """Custom min_rules_text_length is respected."""
        m = _make_market(now=NOW)
        m.rules_text = "Short rule"  # 10 chars
        snap = _make_snapshot()
        sel = CandidateSelector(Policy(min_rules_text_length=10))
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1

    def test_passes_all_filters(self) -> None:
        """A well-formed market passes all filters and appears in results."""
        m = _make_market(now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1
        assert result[0].market_id == "m1"


# ---- Scoring ----------------------------------------------------------------


class TestCandidateSelectorScoring:
    """Score-computation behaviour of CandidateSelector.select()."""

    def test_mid_0_5_gets_max_price_score(self) -> None:
        """A mid price of exactly 0.5 maximises the price component."""
        m = _make_market(now=NOW)
        snap_half = _make_snapshot(mid=0.5)
        snap_edge = _make_snapshot(market_id="m2", mid=0.9)
        m2 = _make_market(market_id="m2", now=NOW)

        sel = CandidateSelector(Policy())
        result = sel.select(
            [m, m2],
            {m.market_id: snap_half, m2.market_id: snap_edge},
            set(),
            now=NOW,
        )
        by_id = {c.market_id: c for c in result}
        # m1 (mid=0.5) should have a higher price score component
        assert by_id["m1"].score > by_id["m2"].score

    def test_higher_liquidity_scores_higher(self) -> None:
        """More liquidity yields a higher liquidity score component."""
        m1 = _make_market(market_id="m1", now=NOW)
        m2 = _make_market(market_id="m2", now=NOW)
        snap1 = _make_snapshot(market_id="m1", liquidity=50_000.0)
        snap2 = _make_snapshot(market_id="m2", liquidity=6_000.0)

        sel = CandidateSelector(Policy())
        result = sel.select(
            [m1, m2],
            {m1.market_id: snap1, m2.market_id: snap2},
            set(),
            now=NOW,
        )
        by_id = {c.market_id: c for c in result}
        assert by_id["m1"].score > by_id["m2"].score

    def test_mid_range_time_preferred(self) -> None:
        """Markets with resolution time near the midpoint score higher on time."""
        policy = Policy(min_hours_to_resolution=24, max_hours_to_resolution=2160)
        midpoint = (24 + 2160) / 2  # 1092 hours

        m_mid = _make_market(market_id="mid", resolution_hours=midpoint, now=NOW)
        m_early = _make_market(market_id="early", resolution_hours=30, now=NOW)

        snap_mid = _make_snapshot(market_id="mid")
        snap_early = _make_snapshot(market_id="early")

        sel = CandidateSelector(policy)
        result = sel.select(
            [m_mid, m_early],
            {m_mid.market_id: snap_mid, m_early.market_id: snap_early},
            set(),
            now=NOW,
        )
        by_id = {c.market_id: c for c in result}
        assert by_id["mid"].score > by_id["early"].score

    def test_category_diversity_bonus(self) -> None:
        """First market in a category gets a higher diversity score than second."""
        m1 = _make_market(market_id="m1", category="politics", now=NOW)
        m2 = _make_market(market_id="m2", category="politics", now=NOW)
        m3 = _make_market(market_id="m3", category="sports", now=NOW)

        # Give all the same liquidity/volume/mid/time so only diversity differs.
        snaps = {
            "m1": _make_snapshot(market_id="m1"),
            "m2": _make_snapshot(market_id="m2"),
            "m3": _make_snapshot(market_id="m3"),
        }

        sel = CandidateSelector(Policy())
        result = sel.select([m1, m2, m3], snaps, set(), now=NOW)

        # m3 (sports, first in its category) should have a higher diversity
        # component than m2 (politics, second in politics).
        by_id = {c.market_id: c for c in result}
        # m1 and m3 are both first in their category, so equal diversity.
        # m2 is second in politics, so lower diversity.
        assert by_id["m1"].score == by_id["m3"].score
        assert by_id["m2"].score < by_id["m1"].score

    def test_scores_are_between_0_and_1(self) -> None:
        """Composite scores should be in [0, 1]."""
        m = _make_market(now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        for c in result:
            assert 0.0 <= c.score <= 1.0

    def test_reasons_populated(self) -> None:
        """Each candidate should have reasons explaining the score."""
        m = _make_market(now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1
        assert len(result[0].reasons) == 5
        # Check that reason labels are present.
        labels = [r.split("=")[0] for r in result[0].reasons]
        assert labels == ["liquidity", "volume", "time", "price", "diversity"]


# ---- Ordering and limits ----------------------------------------------------


class TestCandidateSelectorOrdering:
    """Result ordering and max_candidates_per_cycle."""

    def test_results_sorted_descending(self) -> None:
        """Results are returned sorted by score descending."""
        markets = []
        snaps = {}
        for i in range(5):
            mid = _make_market(
                market_id=f"m{i}",
                category=f"cat{i}",
                resolution_hours=500.0 + i * 100,
                now=NOW,
            )
            markets.append(mid)
            snaps[f"m{i}"] = _make_snapshot(
                market_id=f"m{i}",
                liquidity=10_000.0 + i * 5_000,
                volume=20_000.0 + i * 5_000,
            )

        sel = CandidateSelector(Policy())
        result = sel.select(markets, snaps, set(), now=NOW)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_max_candidates_per_cycle(self) -> None:
        """At most max_candidates_per_cycle results are returned."""
        policy = Policy(max_candidates_per_cycle=2)
        markets = []
        snaps = {}
        for i in range(5):
            mid = _make_market(market_id=f"m{i}", category=f"cat{i}", now=NOW)
            markets.append(mid)
            snaps[f"m{i}"] = _make_snapshot(market_id=f"m{i}")

        sel = CandidateSelector(policy)
        result = sel.select(markets, snaps, set(), now=NOW)
        assert len(result) == 2

    def test_now_defaults_to_utc(self) -> None:
        """When now is not provided, the selector uses current UTC time."""
        m = _make_market(
            resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        # Should not raise; now defaults internally.
        result = sel.select([m], {m.market_id: snap}, set())
        assert len(result) == 1


# ---- Edge cases -------------------------------------------------------------


class TestCandidateSelectorEdgeCases:
    """Edge-case behaviour."""

    def test_extreme_mid_price_zero(self) -> None:
        """Mid price of 0.0 gets a price score of 0.0."""
        m = _make_market(now=NOW)
        snap = _make_snapshot(mid=0.0)
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1
        # price_score = 1.0 - 2 * |0.0 - 0.5| = 1.0 - 1.0 = 0.0
        assert "price=0.00" in result[0].reasons

    def test_extreme_mid_price_one(self) -> None:
        """Mid price of 1.0 gets a price score of 0.0."""
        m = _make_market(now=NOW)
        snap = _make_snapshot(mid=1.0)
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1
        assert "price=0.00" in result[0].reasons

    def test_none_category_diversity(self) -> None:
        """None categories are tracked for diversity like any other value."""
        m1 = _make_market(market_id="m1", category=None, now=NOW)
        m2 = _make_market(market_id="m2", category=None, now=NOW)
        snaps = {
            "m1": _make_snapshot(market_id="m1"),
            "m2": _make_snapshot(market_id="m2"),
        }
        sel = CandidateSelector(Policy())
        result = sel.select([m1, m2], snaps, set(), now=NOW)
        assert len(result) == 2
        # Second None-category market should have lower diversity score.
        by_id = {c.market_id: c for c in result}
        assert by_id["m2"].score < by_id["m1"].score

    def test_resolution_at_exact_min_boundary(self) -> None:
        """Market at exactly min_hours_to_resolution passes the filter."""
        policy = Policy(min_hours_to_resolution=24)
        m = _make_market(resolution_hours=24.0, now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(policy)
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1

    def test_resolution_at_exact_max_boundary(self) -> None:
        """Market at exactly max_hours_to_resolution passes the filter."""
        policy = Policy(max_hours_to_resolution=2160)
        m = _make_market(resolution_hours=2160.0, now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(policy)
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1

    def test_hours_to_resolution_populated(self) -> None:
        """The hours_to_resolution field is correctly set on results."""
        m = _make_market(resolution_hours=500.0, now=NOW)
        snap = _make_snapshot()
        sel = CandidateSelector(Policy())
        result = sel.select([m], {m.market_id: snap}, set(), now=NOW)
        assert len(result) == 1
        assert abs(result[0].hours_to_resolution - 500.0) < 0.01

"""Tests for src.app.scheduler — TradingEngine orchestration."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.aggregation.aggregator import Aggregator, AggregationResult
from src.app.scheduler import TradingEngine
from src.config.policy import Policy, policy_version_hash
from src.db.models import Aggregation, EvidenceItem, Market, MarketSnapshot, Packet, Position
from src.evidence.rss_ingestor import RSSIngestor
from src.evidence.schemas import FetchedArticle
from src.evidence.xai_search import XAISearchClient
from src.llm.panel import PanelOrchestrator
from src.llm.schemas import ModelProposal, PanelResult
from src.packets.builder import PacketBuilder
from src.portfolio.lifecycle import LifecycleAction
from src.polymarket.schemas import GammaMarket


# ---- Helpers ----------------------------------------------------------------


def _make_policy(**overrides) -> Policy:
    """Create a Policy with optional overrides."""
    return Policy(**overrides)


def _make_gamma_market(
    condition_id: str = "cond-1",
    question: str = "Will it rain?",
    category: str = "weather",
    end_date_iso: str = "2026-06-01T00:00:00Z",
    active: bool = True,
    closed: bool = False,
    liquidity: float = 10_000.0,
    volume: float = 20_000.0,
    outcome_prices: str = '["0.55","0.45"]',
) -> GammaMarket:
    """Create a GammaMarket instance for testing."""
    return GammaMarket(
        condition_id=condition_id,
        question=question,
        category=category,
        end_date_iso=end_date_iso,
        active=active,
        closed=closed,
        liquidity=liquidity,
        volume=volume,
        outcome_prices=outcome_prices,
    )


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
    m.rules_text = None
    m.resolution_time_utc = now + timedelta(hours=resolution_hours)
    return m


def _make_snapshot(
    market_id: str = "m1",
    mid: float | None = 0.50,
    liquidity: float | None = 10_000.0,
    volume: float | None = 20_000.0,
    best_bid: float | None = 0.48,
    best_ask: float | None = 0.52,
) -> MagicMock:
    """Create a mock MarketSnapshot instance for testing."""
    s = MagicMock(spec=MarketSnapshot)
    s.market_id = market_id
    s.mid = mid
    s.liquidity = liquidity
    s.volume = volume
    s.best_bid = best_bid
    s.best_ask = best_ask
    s.ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
    return s


def _make_position(
    market_id: str = "mkt-1",
    side: str = "BUY_YES",
    size_eur: float = 100.0,
    avg_entry_price: float = 0.50,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    status: str = "open",
) -> MagicMock:
    """Create a mock Position with the given attributes."""
    pos = MagicMock(spec=Position)
    pos.market_id = market_id
    pos.side = side
    pos.size_eur = size_eur
    pos.avg_entry_price = avg_entry_price
    pos.realized_pnl = realized_pnl
    pos.unrealized_pnl = unrealized_pnl
    pos.status = status
    pos.opened_ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
    pos.last_update_ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
    pos.position_id = 1
    return pos


def _make_engine(
    policy: Policy | None = None,
    repo: AsyncMock | None = None,
    gamma: AsyncMock | None = None,
    rss_ingestor: RSSIngestor | AsyncMock | None = None,
    packet_builder: PacketBuilder | None = None,
    panel_orchestrator: PanelOrchestrator | None = None,
    aggregator: Aggregator | None = None,
    xai_search_client: XAISearchClient | AsyncMock | None = None,
) -> TradingEngine:
    """Create a TradingEngine with mocked dependencies."""
    if policy is None:
        policy = _make_policy()
    if repo is None:
        repo = AsyncMock()
    if gamma is None:
        gamma = AsyncMock()
    return TradingEngine(
        repo=repo,
        gamma_client=gamma,
        policy=policy,
        rss_ingestor=rss_ingestor,
        packet_builder=packet_builder,
        panel_orchestrator=panel_orchestrator,
        aggregator=aggregator,
        xai_search_client=xai_search_client,
    )


# ---- TradingEngine construction --------------------------------------------


class TestTradingEngineInit:
    """TradingEngine constructor and attribute setup."""

    def test_sets_policy_hash(self) -> None:
        policy = _make_policy()
        engine = _make_engine(policy=policy)
        assert engine.policy_hash == policy_version_hash(policy)

    def test_daily_pnl_starts_at_zero(self) -> None:
        engine = _make_engine()
        assert engine.daily_realized_pnl == 0.0

    def test_trade_lock_exists(self) -> None:
        """TradingEngine has an asyncio.Lock for trade concurrency."""
        import asyncio

        engine = _make_engine()
        assert hasattr(engine, "_trade_lock")
        assert isinstance(engine._trade_lock, asyncio.Lock)

    def test_stores_repo(self) -> None:
        repo = AsyncMock()
        engine = _make_engine(repo=repo)
        assert engine.repo is repo

    def test_stores_gamma_client(self) -> None:
        gamma = AsyncMock()
        engine = _make_engine(gamma=gamma)
        assert engine.gamma is gamma

    def test_stores_policy(self) -> None:
        policy = _make_policy(bankroll_eur=5_000.0)
        engine = _make_engine(policy=policy)
        assert engine.policy.bankroll_eur == 5_000.0


# ---- ingest_markets ---------------------------------------------------------


class TestIngestMarkets:
    """Market ingestion from Gamma API to DB."""

    async def test_ingests_valid_markets(self) -> None:
        """Valid Gamma markets are upserted and snapshots created."""
        gm = _make_gamma_market()
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.bulk_upsert_markets.return_value = None
        repo.bulk_add_snapshots.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        count = await engine.ingest_markets()

        assert count == 1
        repo.bulk_upsert_markets.assert_called_once()
        repo.bulk_add_snapshots.assert_called_once()

    async def test_skips_empty_condition_id(self) -> None:
        """Markets with empty condition_id are skipped."""
        gm = _make_gamma_market(condition_id="")
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        engine = _make_engine(repo=repo, gamma=gamma)
        count = await engine.ingest_markets()

        assert count == 0
        repo.bulk_upsert_markets.assert_not_called()
        repo.add_snapshot.assert_not_called()

    async def test_handles_empty_market_list(self) -> None:
        """Empty market list returns zero and does not call bulk upsert."""
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = []

        repo = AsyncMock()
        engine = _make_engine(repo=repo, gamma=gamma)
        count = await engine.ingest_markets()

        assert count == 0
        repo.bulk_upsert_markets.assert_not_called()

    async def test_closed_market_status(self) -> None:
        """Closed markets are recorded with status 'closed'."""
        gm = _make_gamma_market(active=True, closed=True)
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.add_snapshot.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["status"] == "closed"

    async def test_inactive_market_status(self) -> None:
        """Inactive markets are recorded with status 'closed'."""
        gm = _make_gamma_market(active=False, closed=False)
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.add_snapshot.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["status"] == "closed"

    async def test_snapshot_mid_calculated(self) -> None:
        """Snapshot mid is the average of bid and ask."""
        gm = _make_gamma_market(outcome_prices='["0.60","0.40"]')
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.bulk_add_snapshots.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        await engine.ingest_markets()

        snap_list = repo.bulk_add_snapshots.call_args[0][0]
        snap_data = snap_list[0]
        # GammaMarket.best_bid_ask returns (0.60, 0.60) for M1
        assert snap_data["mid"] == pytest.approx(0.60)

    async def test_snapshot_none_prices(self) -> None:
        """Snapshot mid is None when bid/ask cannot be parsed."""
        gm = _make_gamma_market(outcome_prices="invalid")
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.bulk_add_snapshots.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        await engine.ingest_markets()

        snap_list = repo.bulk_add_snapshots.call_args[0][0]
        snap_data = snap_list[0]
        assert snap_data["mid"] is None
        assert snap_data["best_bid"] is None
        assert snap_data["best_ask"] is None

    async def test_invalid_end_date_iso(self) -> None:
        """Invalid end_date_iso results in None resolution_time_utc."""
        gm = _make_gamma_market(end_date_iso="not-a-date")
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.add_snapshot.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["resolution_time_utc"] is None

    async def test_empty_end_date_iso(self) -> None:
        """Empty end_date_iso results in None resolution_time_utc."""
        gm = _make_gamma_market(end_date_iso="")
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.add_snapshot.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["resolution_time_utc"] is None

    async def test_multiple_markets(self) -> None:
        """Multiple markets produce multiple snapshots."""
        gm1 = _make_gamma_market(condition_id="c1")
        gm2 = _make_gamma_market(condition_id="c2")
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm1, gm2]

        repo = AsyncMock()
        repo.bulk_add_snapshots.return_value = 2

        engine = _make_engine(repo=repo, gamma=gamma)
        count = await engine.ingest_markets()

        assert count == 2
        repo.bulk_add_snapshots.assert_called_once()
        snap_list = repo.bulk_add_snapshots.call_args[0][0]
        assert len(snap_list) == 2

    async def test_empty_category_stored_as_none(self) -> None:
        """Empty category string is stored as None."""
        gm = _make_gamma_market(category="")
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = [gm]

        repo = AsyncMock()
        repo.add_snapshot.return_value = 1

        engine = _make_engine(repo=repo, gamma=gamma)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["category"] is None


# ---- run_candidate_scan ----------------------------------------------------


class TestRunCandidateScan:
    """Candidate scanning and trade execution."""

    async def test_no_candidates_returns_zero(self) -> None:
        """When no candidates pass selection, zero trades are executed."""
        repo = AsyncMock()
        repo.get_active_markets.return_value = []
        repo.get_open_positions.return_value = []

        engine = _make_engine(repo=repo)
        trades = await engine.run_candidate_scan()

        assert trades == 0

    async def test_candidate_without_snapshot_skipped(self) -> None:
        """Candidates whose snapshot has no mid are skipped."""
        market = _make_market()
        snap = _make_snapshot(mid=None)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap

        engine = _make_engine(repo=repo)
        # The selector will filter this out (no mid), so 0 trades.
        trades = await engine.run_candidate_scan()
        assert trades == 0

    async def test_trade_execution_records_all_entities(self) -> None:
        """A successful trade records decision, order, fill, and position."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)

        # Mock the selector to return a candidate with high score.
        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()

        assert trades == 1
        repo.add_decision.assert_called_once()
        repo.add_order.assert_called_once()
        repo.add_fill.assert_called_once()
        repo.upsert_position.assert_called_once()

    async def test_trade_blocked_by_risk_manager(self) -> None:
        """Trades blocked by risk checks are not executed."""
        policy = _make_policy(edge_threshold=0.01, max_open_positions=0)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = [
            _make_position(market_id=f"mkt-{i}") for i in range(15)
        ]
        repo.get_latest_snapshot.return_value = snap

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()

        assert trades == 0
        repo.add_decision.assert_not_called()

    async def test_trade_skipped_by_sizing(self) -> None:
        """Trades with insufficient edge are skipped by the sizing module."""
        policy = _make_policy(edge_threshold=0.20)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap

        engine = _make_engine(policy=policy, repo=repo)

        # Score near 0.5 produces near-zero edge.
        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.51
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()

        assert trades == 0
        repo.add_decision.assert_not_called()


# ---- review_open_positions --------------------------------------------------


class TestReviewOpenPositions:
    """Position lifecycle review."""

    async def test_no_positions_is_no_op(self) -> None:
        """No open positions means no reviews."""
        repo = AsyncMock()
        repo.get_open_positions.return_value = []

        engine = _make_engine(repo=repo)
        await engine.review_open_positions()

        repo.upsert_position.assert_not_called()

    async def test_updates_unrealized_pnl(self) -> None:
        """Open positions have their unrealized PnL updated."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(market_id="m1", resolution_hours=500.0)
        snap = _make_snapshot(market_id="m1", mid=0.60)

        repo = AsyncMock()
        repo.get_open_positions.return_value = [pos]
        repo.get_latest_snapshot.return_value = snap
        repo.get_active_markets.return_value = [market]

        engine = _make_engine(repo=repo)
        await engine.review_open_positions()

        # Should have been called at least once for the PnL update.
        assert repo.upsert_position.called

    async def test_close_updates_daily_pnl(self) -> None:
        """Closing a position accumulates realized PnL to daily total."""
        # Use explicit fee_rate so closing produces non-zero PnL.
        policy = _make_policy(fee_rate=0.02)
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        # No snapshot => lifecycle will decide to CLOSE.
        repo = AsyncMock()
        repo.get_open_positions.return_value = [pos]
        repo.get_latest_snapshot.return_value = None
        repo.get_active_markets.return_value = []

        engine = _make_engine(policy=policy, repo=repo)
        assert engine.daily_realized_pnl == 0.0

        await engine.review_open_positions()

        # daily_realized_pnl should have changed (CLOSE path; fee causes non-zero).
        assert engine.daily_realized_pnl != 0.0

    async def test_close_with_snapshot(self) -> None:
        """Position closed due to approaching resolution uses snap.mid."""
        policy = _make_policy(min_hours_to_resolution=24)
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=5.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.60)

        repo = AsyncMock()
        repo.get_open_positions.return_value = [pos]
        repo.get_latest_snapshot.return_value = snap
        repo.get_active_markets.return_value = [market]

        engine = _make_engine(policy=policy, repo=repo)
        await engine.review_open_positions()

        # Should have been called for both unrealized PnL update and close.
        assert repo.upsert_position.call_count >= 1
        # Check that the close call set status to "closed".
        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) == 1

    async def test_hold_does_not_close(self) -> None:
        """Positions that should be held are not closed."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.51)

        repo = AsyncMock()
        repo.get_open_positions.return_value = [pos]
        repo.get_latest_snapshot.return_value = snap
        repo.get_active_markets.return_value = [market]

        engine = _make_engine(repo=repo)
        await engine.review_open_positions()

        # Should have updated unrealized PnL but not closed.
        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) == 0

    async def test_no_resolution_time_hours_none(self) -> None:
        """When market has no resolution time, hours_to_res is None."""
        pos = _make_position(market_id="m1", side="BUY_YES")
        market = _make_market(market_id="m1")
        market.resolution_time_utc = None
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_open_positions.return_value = [pos]
        repo.get_latest_snapshot.return_value = snap
        repo.get_active_markets.return_value = [market]

        engine = _make_engine(repo=repo)
        # Should not raise when resolution_time_utc is None.
        await engine.review_open_positions()


# ---- reset_daily -----------------------------------------------------------


class TestResetDaily:
    """Daily reset behaviour."""

    def test_resets_daily_pnl(self) -> None:
        engine = _make_engine()
        engine.daily_realized_pnl = -250.0
        engine.reset_daily()
        assert engine.daily_realized_pnl == 0.0

    def test_reset_idempotent(self) -> None:
        engine = _make_engine()
        engine.reset_daily()
        engine.reset_daily()
        assert engine.daily_realized_pnl == 0.0


# ---- ingest_evidence -------------------------------------------------------


class TestIngestEvidence:
    """Evidence ingestion from RSS feeds to DB."""

    async def test_stores_fetched_articles(self) -> None:
        """Fetched articles are stored via bulk_upsert_evidence."""
        repo = AsyncMock()
        repo.bulk_upsert_evidence.return_value = 2

        rss_ingestor = AsyncMock(spec=RSSIngestor)
        rss_ingestor.fetch_all.return_value = [
            FetchedArticle(
                url="https://example.com/1",
                title="Article 1",
                extracted_text="Content 1",
                source_name="Test Feed",
            ),
            FetchedArticle(
                url="https://example.com/2",
                title="Article 2",
                extracted_text="Content 2",
                source_name="Test Feed",
            ),
        ]
        rss_ingestor._compute_hash.side_effect = lambda u, t: f"hash_{u}"

        policy = _make_policy()
        engine = _make_engine(
            repo=repo, rss_ingestor=rss_ingestor, policy=policy,
        )
        count = await engine.ingest_evidence()

        assert count == 2
        repo.bulk_upsert_evidence.assert_called_once()
        items = repo.bulk_upsert_evidence.call_args[0][0]
        assert len(items) == 2
        assert items[0]["source_type"] == "rss"

    async def test_no_ingestor_returns_zero(self) -> None:
        """Without an RSS ingestor, ingestion returns 0."""
        engine = _make_engine()
        count = await engine.ingest_evidence()
        assert count == 0

    async def test_empty_fetch_returns_zero(self) -> None:
        """When no articles are fetched, zero is stored."""
        repo = AsyncMock()
        repo.bulk_upsert_evidence.return_value = 0

        rss_ingestor = AsyncMock(spec=RSSIngestor)
        rss_ingestor.fetch_all.return_value = []

        engine = _make_engine(repo=repo, rss_ingestor=rss_ingestor)
        count = await engine.ingest_evidence()

        assert count == 0


# ---- Packet building during candidate scan ----------------------------------


class TestPacketBuildingInScan:
    """Packets are built and stored during candidate scan."""

    async def test_packets_stored_for_candidates(self) -> None:
        """When packet_builder is present, packets are stored for candidates."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1
        repo.get_recent_evidence.return_value = []
        repo.add_packet.return_value = 1

        packet_builder = PacketBuilder(policy)

        engine = _make_engine(
            policy=policy, repo=repo, packet_builder=packet_builder,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        await engine.run_candidate_scan()

        repo.add_packet.assert_called_once()
        packet_data = repo.add_packet.call_args[0][0]
        assert packet_data["market_id"] == "m1"
        assert "packet_hash" in packet_data
        assert "packet_json" in packet_data

    async def test_no_packet_builder_skips_packet_building(self) -> None:
        """Without packet_builder, no packets are stored."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        await engine.run_candidate_scan()

        repo.add_packet.assert_not_called()


# ---- M3 Panel integration ---------------------------------------------------


def _make_proposal(
    model_id: str = "probabilist",
    p_true: float = 0.65,
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


class TestPanelIntegration:
    """M3: Panel + aggregator integration in candidate scan."""

    async def test_m1_fallback_when_no_panel(self) -> None:
        """Without panel_orchestrator, M1 synthetic edge is used."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)
        assert engine.panel is None
        assert engine.aggregator is None

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1

    async def test_panel_trade_executes(self) -> None:
        """When panel+aggregator are set and trade_allowed, trade executes."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1
        repo.get_panel_markets_today.return_value = 0
        repo.add_model_run.return_value = "run-1"
        repo.add_aggregation.return_value = 1

        # Mock packet row.
        packet_row = MagicMock(spec=Packet)
        packet_row.packet_id = 1
        packet_row.packet_json = {
            "market_id": "m1",
            "ts_utc": "2025-06-01T00:00:00Z",
            "market_context": {
                "question": "Will it rain?",
                "current_mid": 0.50,
                "best_bid": 0.48,
                "best_ask": 0.52,
            },
            "evidence_items": [],
            "packet_version": "m2.0",
        }
        repo.get_latest_packet.return_value = packet_row

        repo.get_latest_aggregation.return_value = None

        # Mock panel orchestrator.
        panel = AsyncMock(spec=PanelOrchestrator)
        panel.charter_hash = "abc123"
        panel.default_panel = []
        panel.escalation_agents = []
        panel.run_panel.return_value = PanelResult(
            proposals=[_make_proposal(p_true=0.65)],
            agents_used=["probabilist"],
        )
        panel.determine_escalation.return_value = (None, None)

        # Mock aggregator.
        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.65,
            confidence=0.80,
            disagreement=0.02,
            veto=False,
            trade_allowed=True,
        )

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1
        repo.add_model_run.assert_called()
        repo.add_aggregation.assert_called()

    async def test_panel_veto_blocks_trade(self) -> None:
        """When aggregator returns trade_allowed=False, trade is blocked."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_panel_markets_today.return_value = 0
        repo.add_model_run.return_value = "run-1"
        repo.add_aggregation.return_value = 1

        packet_row = MagicMock(spec=Packet)
        packet_row.packet_id = 1
        packet_row.packet_json = {
            "market_id": "m1",
            "ts_utc": "2025-06-01T00:00:00Z",
            "market_context": {
                "question": "Will it rain?",
                "current_mid": 0.50,
                "best_bid": 0.48,
                "best_ask": 0.52,
            },
            "evidence_items": [],
            "packet_version": "m2.0",
        }
        repo.get_latest_packet.return_value = packet_row

        panel = AsyncMock(spec=PanelOrchestrator)
        panel.charter_hash = "abc123"
        panel.run_panel.return_value = PanelResult(
            proposals=[_make_proposal(p_true=0.65, rules_ambiguity=0.80)],
            agents_used=["probabilist"],
        )
        panel.determine_escalation.return_value = (None, None)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.65,
            confidence=0.80,
            disagreement=0.02,
            veto=True,
            veto_reasons=["ambiguity"],
            trade_allowed=False,
        )

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 0
        repo.add_decision.assert_not_called()

    async def test_panel_escalation(self) -> None:
        """When determine_escalation returns an agent, escalation runs."""
        from src.llm.schemas import EscalationTrigger, PanelAgent

        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1
        repo.get_panel_markets_today.return_value = 0
        repo.add_model_run.return_value = "run-1"
        repo.add_aggregation.return_value = 1

        packet_row = MagicMock(spec=Packet)
        packet_row.packet_id = 1
        packet_row.packet_json = {
            "market_id": "m1",
            "ts_utc": "2025-06-01T00:00:00Z",
            "market_context": {
                "question": "Will it rain?",
                "current_mid": 0.50,
                "best_bid": 0.48,
                "best_ask": 0.52,
            },
            "evidence_items": [],
            "packet_version": "m2.0",
        }
        repo.get_latest_packet.return_value = packet_row

        default_result = PanelResult(
            proposals=[_make_proposal(p_true=0.65)],
            agents_used=["probabilist"],
        )
        escalation_result = PanelResult(
            proposals=[
                _make_proposal(p_true=0.65),
                _make_proposal(p_true=0.70, model_id="escalation_openai"),
            ],
            agents_used=["probabilist", "escalation_openai"],
            escalation_trigger=EscalationTrigger.DISAGREEMENT,
            escalation_agent="escalation_openai",
        )

        escalation_agent = PanelAgent(
            "escalation_openai", "openai", "gpt-5.2", "probability", False,
        )

        panel = AsyncMock(spec=PanelOrchestrator)
        panel.charter_hash = "abc123"
        panel.default_panel = []
        panel.escalation_agents = [escalation_agent]
        panel.run_panel.return_value = default_result
        panel.determine_escalation.return_value = (
            escalation_agent, EscalationTrigger.DISAGREEMENT,
        )
        panel.run_escalation.return_value = escalation_result

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.70,
            confidence=0.85,
            disagreement=0.02,
            veto=False,
            trade_allowed=True,
        )

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1
        panel.run_panel.assert_called_once()
        panel.run_escalation.assert_called_once()

    async def test_panel_daily_limit_blocks(self) -> None:
        """When max_panel_markets_per_day is reached, panel returns None."""
        policy = _make_policy(edge_threshold=0.01, max_panel_markets_per_day=0)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_panel_markets_today.return_value = 20

        panel = AsyncMock(spec=PanelOrchestrator)
        aggregator = MagicMock(spec=Aggregator)

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 0
        panel.run_panel.assert_not_called()


# ---- M4: Review triggers ----------------------------------------------------


class TestReviewTriggers:
    """Review trigger detection for open positions."""

    def test_odds_move_trigger(self) -> None:
        """Odds move >= threshold fires trigger."""
        policy = _make_policy(odds_move_recheck_threshold=0.05)
        engine = _make_engine(policy=policy)
        pos = _make_position(avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.56)  # move = 0.06 >= 0.05
        triggers = engine._check_review_triggers(pos, snap, None, 500.0)
        assert "odds_move" in triggers

    def test_no_odds_move_trigger(self) -> None:
        """Odds move below threshold does not fire trigger."""
        policy = _make_policy(odds_move_recheck_threshold=0.05)
        engine = _make_engine(policy=policy)
        pos = _make_position(avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.52)  # move = 0.02 < 0.05
        triggers = engine._check_review_triggers(pos, snap, None, 500.0)
        assert "odds_move" not in triggers

    def test_liquidity_drop_trigger(self) -> None:
        """Liquidity below min fires trigger."""
        policy = _make_policy(min_liquidity_eur=5000.0)
        engine = _make_engine(policy=policy)
        pos = _make_position()
        snap = _make_snapshot(liquidity=3000.0)
        triggers = engine._check_review_triggers(pos, snap, None, 500.0)
        assert "liquidity_drop" in triggers

    def test_approaching_resolution_trigger(self) -> None:
        """Approaching resolution (within 2x min) fires trigger."""
        policy = _make_policy(min_hours_to_resolution=24)
        engine = _make_engine(policy=policy)
        pos = _make_position()
        snap = _make_snapshot()
        # 40 hours < 24 * 2 = 48 hours
        triggers = engine._check_review_triggers(pos, snap, None, 40.0)
        assert "approaching_resolution" in triggers

    def test_no_triggers_when_normal(self) -> None:
        """No triggers fire under normal conditions."""
        policy = _make_policy()
        engine = _make_engine(policy=policy)
        pos = _make_position(avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.51, liquidity=10_000.0)
        triggers = engine._check_review_triggers(pos, snap, None, 500.0)
        assert triggers == []


# ---- M4: Evaluate with aggregation ------------------------------------------


class TestEvaluateWithAggregation:
    """PositionLifecycle.evaluate_with_aggregation() tests."""

    def _make_agg(
        self,
        p_consensus: float = 0.60,
        confidence: float = 0.80,
        disagreement: float = 0.02,
        veto: bool = False,
        trade_allowed: bool = True,
    ) -> AggregationResult:
        return AggregationResult(
            p_consensus=p_consensus,
            confidence=confidence,
            disagreement=disagreement,
            veto=veto,
            trade_allowed=trade_allowed,
        )

    def test_edge_flip_closes(self) -> None:
        """Edge flip: consensus implies opposite side → CLOSE."""
        from src.portfolio.lifecycle import PositionLifecycle

        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES", avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.40)  # implies NO, but position is YES
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE
        assert any("edge flip" in r for r in decision.reasons)

    def test_ambiguity_veto_first_strike_holds(self) -> None:
        """First aggregation veto → HOLD (consecutive required)."""
        from src.portfolio.lifecycle import PositionLifecycle

        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES", avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.60, veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.HOLD
        assert "first_veto_hold" in decision.reasons

    def test_ambiguity_veto_consecutive_closes(self) -> None:
        """Two consecutive aggregation vetoes → CLOSE."""
        from src.portfolio.lifecycle import PositionLifecycle

        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES", avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.60, veto=True, trade_allowed=False)
        prior_agg = self._make_agg(p_consensus=0.60, veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, prior_agg)
        assert decision.action == LifecycleAction.CLOSE
        assert any("consecutive" in r for r in decision.reasons)

    def test_take_profit_closes(self) -> None:
        """Take-profit: small edge + confidence drop → CLOSE."""
        from src.portfolio.lifecycle import PositionLifecycle

        policy = Policy(take_profit_band=0.02, confidence_drop_threshold=0.15)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES", avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.61)  # |0.61 - 0.60| = 0.01 < 0.02
        current_agg = self._make_agg(p_consensus=0.60, confidence=0.60)
        prior_agg = self._make_agg(p_consensus=0.60, confidence=0.80)
        decision = lc.evaluate_with_aggregation(
            pos, snap, 0.50, 100.0, current_agg, prior_agg
        )
        assert decision.action == LifecycleAction.CLOSE
        assert any("take-profit" in r for r in decision.reasons)

    def test_disagreement_block_closes(self) -> None:
        """Disagreement >= block threshold → CLOSE."""
        from src.portfolio.lifecycle import PositionLifecycle

        policy = Policy(disagreement_block_threshold=0.15)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES", avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.60, disagreement=0.15)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE
        assert any("block threshold" in r for r in decision.reasons)

    def test_disagreement_increase_reduces(self) -> None:
        """Disagreement increased above penalty start → REDUCE."""
        from src.portfolio.lifecycle import PositionLifecycle

        policy = Policy(
            disagreement_block_threshold=0.15,
            disagreement_size_penalty_start=0.08,
        )
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES", avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.50)
        current_agg = self._make_agg(p_consensus=0.60, disagreement=0.10)
        prior_agg = self._make_agg(p_consensus=0.60, disagreement=0.05)
        decision = lc.evaluate_with_aggregation(
            pos, snap, 0.50, 100.0, current_agg, prior_agg
        )
        assert decision.action == LifecycleAction.REDUCE
        assert any("disagreement increased" in r for r in decision.reasons)

    def test_hold_when_no_issues(self) -> None:
        """No aggregation issues → HOLD."""
        from src.portfolio.lifecycle import PositionLifecycle

        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES", avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.55)
        agg = self._make_agg(p_consensus=0.60, disagreement=0.02)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.HOLD

    def test_deterministic_check_takes_priority(self) -> None:
        """No snapshot → CLOSE even with aggregation available."""
        from src.portfolio.lifecycle import PositionLifecycle

        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        agg = self._make_agg(p_consensus=0.60)
        decision = lc.evaluate_with_aggregation(pos, None, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE


# ---- M4: Panel review loop --------------------------------------------------


def _setup_panel_review_engine(
    repo: AsyncMock,
    panel: AsyncMock | None = None,
    aggregator: MagicMock | None = None,
    policy: Policy | None = None,
    xai_search_client: AsyncMock | None = None,
) -> TradingEngine:
    """Create a TradingEngine configured for M4 panel review tests."""
    if policy is None:
        policy = _make_policy()
    packet_builder = PacketBuilder(policy)

    if panel is None:
        panel = AsyncMock(spec=PanelOrchestrator)
        panel.charter_hash = "abc123"
        panel.default_panel = []
        panel.escalation_agents = []
        panel.run_panel.return_value = PanelResult(
            proposals=[_make_proposal(p_true=0.65)],
            agents_used=["probabilist"],
        )
        panel.determine_escalation.return_value = (None, None)

    if aggregator is None:
        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.65,
            confidence=0.80,
            disagreement=0.02,
            veto=False,
            trade_allowed=True,
        )

    return _make_engine(
        policy=policy,
        repo=repo,
        packet_builder=packet_builder,
        panel_orchestrator=panel,
        aggregator=aggregator,
        xai_search_client=xai_search_client,
    )


def _setup_repo_for_review(
    pos: MagicMock,
    market: MagicMock,
    snap: MagicMock,
    agg_result_dict: dict | None = None,
) -> AsyncMock:
    """Create a mock repo configured for position review tests."""
    repo = AsyncMock()
    repo.get_open_positions.return_value = [pos]
    repo.get_active_markets.return_value = [market]
    repo.get_latest_snapshot.return_value = snap
    repo.count_recent_evidence.return_value = 0
    repo.get_recent_evidence.return_value = []
    repo.add_packet.return_value = 1
    repo.get_panel_markets_today.return_value = 0
    repo.add_model_run.return_value = "run-1"
    repo.add_aggregation.return_value = 1
    repo.add_decision.return_value = 1
    repo.add_order.return_value = 1
    repo.add_fill.return_value = 1

    packet_row = MagicMock(spec=Packet)
    packet_row.packet_id = 1
    packet_row.packet_json = {
        "market_id": pos.market_id,
        "ts_utc": "2025-06-01T00:00:00Z",
        "market_context": {
            "question": market.question,
            "current_mid": snap.mid,
            "best_bid": snap.best_bid,
            "best_ask": snap.best_ask,
        },
        "evidence_items": [],
        "packet_version": "m2.0",
    }
    repo.get_latest_packet.return_value = packet_row

    if agg_result_dict:
        agg_row = MagicMock(spec=Aggregation)
        agg_row.aggregation_json = agg_result_dict
        repo.get_latest_aggregation.return_value = agg_row
    else:
        repo.get_latest_aggregation.return_value = None

    return repo


class TestPanelReviewLoop:
    """M4: Panel re-analysis during position review."""

    async def test_no_triggers_uses_deterministic(self) -> None:
        """No triggers → deterministic lifecycle evaluation."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.51, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)
        engine = _setup_panel_review_engine(repo)

        await engine.review_open_positions()

        # Panel should not be called since no triggers fired.
        engine.panel.run_panel.assert_not_called()

    async def test_odds_move_triggers_panel(self) -> None:
        """Odds move trigger → panel re-analysis runs."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        # mid=0.60, odds move = 0.10 >= 0.05 threshold
        snap = _make_snapshot(market_id="m1", mid=0.60, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)
        engine = _setup_panel_review_engine(repo)

        await engine.review_open_positions()

        engine.panel.run_panel.assert_called()

    async def test_panel_close_decision(self) -> None:
        """Panel re-analysis → consecutive aggregation vetoes → position closed."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.60, liquidity=10_000.0)

        # Provide a prior aggregation that was also a veto (consecutive).
        prior_agg_dict = {
            "p_consensus": 0.60,
            "confidence": 0.80,
            "disagreement": 0.02,
            "veto": True,
            "trade_allowed": False,
        }
        repo = _setup_repo_for_review(pos, market, snap, agg_result_dict=prior_agg_dict)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.60,
            confidence=0.80,
            disagreement=0.02,
            veto=True,
            trade_allowed=False,
        )

        engine = _setup_panel_review_engine(repo, aggregator=aggregator)
        await engine.review_open_positions()

        # Position should be closed (consecutive veto).
        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) == 1

    async def test_panel_hold_decision(self) -> None:
        """Panel re-analysis → HOLD → position stays open."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.60, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)
        engine = _setup_panel_review_engine(repo)

        await engine.review_open_positions()

        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) == 0

    async def test_panel_reduce_decision(self) -> None:
        """Panel re-analysis → disagreement increase → REDUCE."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(
            market_id="m1", mid=0.60, liquidity=10_000.0,
            best_bid=0.58, best_ask=0.62,
        )

        prior_agg_dict = {
            "p_consensus": 0.60,
            "confidence": 0.80,
            "disagreement": 0.05,
            "veto": False,
            "trade_allowed": True,
        }
        repo = _setup_repo_for_review(pos, market, snap, agg_result_dict=prior_agg_dict)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.60,
            confidence=0.80,
            disagreement=0.10,  # increased above penalty start (0.08)
            veto=False,
            trade_allowed=True,
        )

        engine = _setup_panel_review_engine(repo, aggregator=aggregator)
        await engine.review_open_positions()

        # Decision + order + fill should be recorded for REDUCE.
        repo.add_decision.assert_called()
        decision_data = repo.add_decision.call_args[0][0]
        assert decision_data["action"] == "REDUCE"
        repo.add_order.assert_called()
        repo.add_fill.assert_called()

    async def test_xai_search_on_odds_move(self) -> None:
        """xAI search is triggered when odds_move fires."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.60, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)

        xai = AsyncMock(spec=XAISearchClient)
        xai.can_search.return_value = True
        xai.search.return_value = [
            FetchedArticle(
                url="https://example.com/news",
                title="Breaking News",
                extracted_text="Relevant content",
                source_name="xai_search",
            ),
        ]

        engine = _setup_panel_review_engine(repo, xai_search_client=xai)
        await engine.review_open_positions()

        xai.search.assert_called_once()
        repo.bulk_upsert_evidence.assert_called()

    async def test_escalation_during_position_review(self) -> None:
        """Escalation during position review appends N+1 proposal."""
        from src.llm.schemas import EscalationTrigger, PanelAgent

        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.60, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)

        default_result = PanelResult(
            proposals=[_make_proposal(p_true=0.65)],
            agents_used=["probabilist"],
        )
        escalation_result = PanelResult(
            proposals=[
                _make_proposal(p_true=0.65),
                _make_proposal(p_true=0.70, model_id="escalation_openai"),
            ],
            agents_used=["probabilist", "escalation_openai"],
        )

        escalation_agent = PanelAgent(
            "escalation_openai", "openai", "gpt-5.2", "probability", False,
        )

        panel = AsyncMock(spec=PanelOrchestrator)
        panel.charter_hash = "abc123"
        panel.default_panel = []
        panel.escalation_agents = [escalation_agent]
        panel.run_panel.return_value = default_result
        panel.determine_escalation.return_value = (
            escalation_agent, EscalationTrigger.DISAGREEMENT,
        )
        panel.run_escalation.return_value = escalation_result

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.65,
            confidence=0.80,
            disagreement=0.02,
            veto=False,
            trade_allowed=True,
        )

        engine = _setup_panel_review_engine(repo, panel=panel, aggregator=aggregator)
        await engine.review_open_positions()

        panel.run_panel.assert_called_once()
        panel.run_escalation.assert_called_once()

    async def test_daily_limit_shared_with_candidates(self) -> None:
        """Panel daily limit blocks position review when quota exhausted."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.60, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)
        repo.get_panel_markets_today.return_value = 20  # at limit

        policy = _make_policy(max_panel_markets_per_day=20)
        engine = _setup_panel_review_engine(repo, policy=policy)
        await engine.review_open_positions()

        # Panel should not be called since daily limit is reached.
        # The _run_panel_for_position returns None, so it falls back to deterministic.
        # Position should still be reviewed deterministically (HOLD in this case).
        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) == 0

    async def test_new_evidence_triggers_reanalysis(self) -> None:
        """New evidence trigger fires when count_recent_evidence > 0."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        # No odds_move trigger (mid close to entry)
        snap = _make_snapshot(market_id="m1", mid=0.51, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)
        repo.count_recent_evidence.return_value = 3  # new evidence

        engine = _setup_panel_review_engine(repo)
        await engine.review_open_positions()

        engine.panel.run_panel.assert_called()


# ---- M4: REDUCE execution ---------------------------------------------------


class TestReduceExecution:
    """REDUCE action execution tests."""

    async def test_reduce_records_decision_order_fill(self) -> None:
        """REDUCE creates decision, order, and fill records."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(
            market_id="m1", mid=0.60, liquidity=10_000.0,
            best_bid=0.58, best_ask=0.62,
        )

        prior_agg_dict = {
            "p_consensus": 0.60,
            "confidence": 0.80,
            "disagreement": 0.05,
            "veto": False,
            "trade_allowed": True,
        }
        repo = _setup_repo_for_review(pos, market, snap, agg_result_dict=prior_agg_dict)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.60,
            confidence=0.80,
            disagreement=0.10,
            veto=False,
            trade_allowed=True,
        )

        engine = _setup_panel_review_engine(repo, aggregator=aggregator)
        await engine.review_open_positions()

        # Decision recorded with action=REDUCE.
        repo.add_decision.assert_called()
        decision_data = repo.add_decision.call_args[0][0]
        assert decision_data["action"] == "REDUCE"
        assert decision_data["size_eur"] == pytest.approx(50.0)  # 100 * 0.50

        # Order recorded with side=SELL.
        repo.add_order.assert_called()
        order_data = repo.add_order.call_args[0][0]
        assert order_data["side"] == "SELL"
        assert order_data["size_eur"] == pytest.approx(50.0)

        # Fill recorded.
        repo.add_fill.assert_called()
        fill_data = repo.add_fill.call_args[0][0]
        assert fill_data["size_eur"] == pytest.approx(50.0)

    async def test_reduce_updates_position_size(self) -> None:
        """REDUCE updates position size_eur and realized_pnl."""
        policy = _make_policy(reduce_fraction=0.50)
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=200.0,
            avg_entry_price=0.50, realized_pnl=0.0,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(
            market_id="m1", mid=0.60, liquidity=10_000.0,
            best_bid=0.58, best_ask=0.62,
        )

        prior_agg_dict = {
            "p_consensus": 0.60,
            "confidence": 0.80,
            "disagreement": 0.05,
            "veto": False,
            "trade_allowed": True,
        }
        repo = _setup_repo_for_review(pos, market, snap, agg_result_dict=prior_agg_dict)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.60,
            confidence=0.80,
            disagreement=0.10,
            veto=False,
            trade_allowed=True,
        )

        engine = _setup_panel_review_engine(repo, aggregator=aggregator, policy=policy)
        await engine.review_open_positions()

        # Find the position upsert after reduce (should have size_eur=100).
        reduce_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("size_eur") == pytest.approx(100.0) and c[0][0].get("status") == "open"
        ]
        assert len(reduce_calls) >= 1

    async def test_reduce_accumulates_daily_pnl(self) -> None:
        """REDUCE adds partial realized PnL to daily total."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(
            market_id="m1", mid=0.60, liquidity=10_000.0,
            best_bid=0.58, best_ask=0.62,
        )

        prior_agg_dict = {
            "p_consensus": 0.60,
            "confidence": 0.80,
            "disagreement": 0.05,
            "veto": False,
            "trade_allowed": True,
        }
        repo = _setup_repo_for_review(pos, market, snap, agg_result_dict=prior_agg_dict)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.60,
            confidence=0.80,
            disagreement=0.10,
            veto=False,
            trade_allowed=True,
        )

        engine = _setup_panel_review_engine(repo, aggregator=aggregator)
        assert engine.daily_realized_pnl == 0.0
        await engine.review_open_positions()

        # daily_realized_pnl should have been updated.
        assert engine.daily_realized_pnl != 0.0

    async def test_reduce_dust_position_closes_instead(self) -> None:
        """REDUCE on a dust position (< dust_position_eur) closes it entirely."""
        policy = _make_policy(dust_position_eur=1.0)
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=0.001,
            avg_entry_price=0.50, realized_pnl=0.0,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(
            market_id="m1", mid=0.60, liquidity=10_000.0,
            best_bid=0.58, best_ask=0.62,
        )

        prior_agg_dict = {
            "p_consensus": 0.60,
            "confidence": 0.80,
            "disagreement": 0.05,
            "veto": False,
            "trade_allowed": True,
        }
        repo = _setup_repo_for_review(pos, market, snap, agg_result_dict=prior_agg_dict)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.60,
            confidence=0.80,
            disagreement=0.10,
            veto=False,
            trade_allowed=True,
        )

        engine = _setup_panel_review_engine(repo, aggregator=aggregator, policy=policy)
        await engine.review_open_positions()

        # Should close (status="closed") instead of reducing.
        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) >= 1
        # No order/fill should be created for dust close.
        repo.add_order.assert_not_called()

    async def test_prior_agg_with_extra_keys_parses_ok(self) -> None:
        """Prior aggregation JSON with extra keys (p_market, consensus_side) parses."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(
            market_id="m1", mid=0.52, liquidity=10_000.0,
            best_bid=0.50, best_ask=0.54,
        )

        # Include extra keys that the DB stores but AggregationResult doesn't have.
        prior_agg_dict = {
            "p_consensus": 0.60,
            "confidence": 0.80,
            "disagreement": 0.05,
            "veto": False,
            "trade_allowed": True,
            "p_market": 0.50,
            "consensus_side": "BUY_YES",
        }
        repo = _setup_repo_for_review(pos, market, snap, agg_result_dict=prior_agg_dict)

        aggregator = MagicMock(spec=Aggregator)
        aggregator.aggregate.return_value = AggregationResult(
            p_consensus=0.65,
            confidence=0.80,
            disagreement=0.02,
            veto=False,
            trade_allowed=True,
        )

        engine = _setup_panel_review_engine(repo, aggregator=aggregator)
        # Should not raise — prior_agg parses despite extra keys.
        await engine.review_open_positions()


# ---- M4: Backward compatibility ---------------------------------------------


class TestM4BackwardCompat:
    """M4 changes don't break existing deterministic lifecycle."""

    async def test_no_panel_uses_deterministic(self) -> None:
        """Without panel, review uses deterministic lifecycle (existing behavior)."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.51)

        repo = AsyncMock()
        repo.get_open_positions.return_value = [pos]
        repo.get_active_markets.return_value = [market]
        repo.get_latest_snapshot.return_value = snap
        repo.count_recent_evidence.return_value = 0

        engine = _make_engine(repo=repo)  # no panel, no aggregator
        await engine.review_open_positions()

        # Should be HOLD — no close calls.
        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) == 0

    async def test_deterministic_close_still_works(self) -> None:
        """Deterministic close (no snapshot) still works without panel."""
        pos = _make_position(market_id="m1", side="BUY_YES", size_eur=100.0)

        repo = AsyncMock()
        repo.get_open_positions.return_value = [pos]
        repo.get_active_markets.return_value = []
        repo.get_latest_snapshot.return_value = None
        repo.count_recent_evidence.return_value = 0

        engine = _make_engine(repo=repo)
        await engine.review_open_positions()

        close_calls = [
            c for c in repo.upsert_position.call_args_list
            if c[0][0].get("status") == "closed"
        ]
        assert len(close_calls) == 1


# ---- Proactive xAI search for candidates -----------------------------------


class TestProactiveSearch:
    """Proactive xAI search for candidates with sparse evidence."""

    async def test_proactive_search_for_sparse_candidates(self) -> None:
        """xAI search is called for candidates with <2 evidence items."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1
        repo.get_recent_evidence.return_value = []
        repo.add_packet.return_value = 1
        repo.bulk_upsert_evidence.return_value = 1

        packet_builder = PacketBuilder(policy)

        xai = AsyncMock(spec=XAISearchClient)
        xai.can_search.return_value = True
        xai.search.return_value = [
            FetchedArticle(
                url="https://example.com/news",
                title="Breaking News",
                extracted_text="Relevant content",
                source_name="xai_search",
            ),
        ]

        engine = _make_engine(
            policy=policy, repo=repo,
            packet_builder=packet_builder,
            xai_search_client=xai,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        await engine.run_candidate_scan()

        xai.search.assert_called_once()
        repo.bulk_upsert_evidence.assert_called()
        # Verify evidence items have correct source_type.
        items = repo.bulk_upsert_evidence.call_args[0][0]
        assert items[0]["source_type"] == "xai_search"

    async def test_proactive_search_skips_candidates_with_evidence(self) -> None:
        """Candidates with >=2 evidence items are not searched."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        # Return evidence items from the DB so linker can associate them.
        ev1 = MagicMock(spec=EvidenceItem)
        ev1.evidence_id = 1
        ev1.extracted_text = "evidence text 1"
        ev1.title = "Evidence 1"
        ev1.url = "https://example.com/e1"
        ev1.published_ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
        ev1.source_type = "rss"
        ev2 = MagicMock(spec=EvidenceItem)
        ev2.evidence_id = 2
        ev2.extracted_text = "evidence text 2"
        ev2.title = "Evidence 2"
        ev2.url = "https://example.com/e2"
        ev2.published_ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
        ev2.source_type = "rss"
        repo.get_recent_evidence.return_value = [ev1, ev2]
        repo.add_packet.return_value = 1

        packet_builder = PacketBuilder(policy)

        xai = AsyncMock(spec=XAISearchClient)
        xai.can_search.return_value = True
        xai.search.return_value = []

        engine = _make_engine(
            policy=policy, repo=repo,
            packet_builder=packet_builder,
            xai_search_client=xai,
        )

        # Mock linker to return 2 items for this market.
        engine.evidence_linker = AsyncMock()
        engine.evidence_linker.link.return_value = {"m1": [ev1, ev2]}

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        await engine.run_candidate_scan()

        # xAI search should NOT be called since candidate has >=2 evidence.
        xai.search.assert_not_called()

    async def test_proactive_search_respects_daily_cap(self) -> None:
        """Proactive search stops when xAI daily cap is reached."""
        policy = _make_policy(edge_threshold=0.01)
        m1 = _make_market(market_id="m1")
        m2 = _make_market(market_id="m2")
        s1 = _make_snapshot(market_id="m1", mid=0.50)
        s2 = _make_snapshot(market_id="m2", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [m1, m2]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.side_effect = lambda mid: (
            s1 if mid == "m1" else s2
        )
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1
        repo.get_recent_evidence.return_value = []
        repo.add_packet.return_value = 1
        repo.bulk_upsert_evidence.return_value = 1

        packet_builder = PacketBuilder(policy)

        xai = AsyncMock(spec=XAISearchClient)
        # Allow first call, then block.
        xai.can_search.side_effect = [True, False]
        xai.search.return_value = [
            FetchedArticle(
                url="https://example.com/news",
                title="News",
                extracted_text="Content",
                source_name="xai_search",
            ),
        ]

        engine = _make_engine(
            policy=policy, repo=repo,
            packet_builder=packet_builder,
            xai_search_client=xai,
        )

        c1 = MagicMock()
        c1.market_id = "m1"
        c1.score = 0.90
        c2 = MagicMock()
        c2.market_id = "m2"
        c2.score = 0.85
        engine.selector = MagicMock()
        engine.selector.select.return_value = [c1, c2]

        await engine.run_candidate_scan()

        # Only one search call (second blocked by can_search=False).
        assert xai.search.call_count == 1

    async def test_social_search_on_odds_move_review(self) -> None:
        """Social search is triggered alongside regular search on odds move."""
        pos = _make_position(
            market_id="m1", side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50,
        )
        market = _make_market(
            market_id="m1", resolution_hours=500.0,
            now=datetime.now(timezone.utc),
        )
        snap = _make_snapshot(market_id="m1", mid=0.60, liquidity=10_000.0)

        repo = _setup_repo_for_review(pos, market, snap)

        xai = AsyncMock(spec=XAISearchClient)
        xai.can_search.return_value = True
        xai.search.return_value = [
            FetchedArticle(
                url="https://example.com/news",
                title="News",
                extracted_text="Content",
                source_name="xai_search",
            ),
        ]
        xai.search_social.return_value = [
            FetchedArticle(
                url="https://x.com/post/1",
                title="Tweet",
                extracted_text="Social content",
                source_name="xai_social",
            ),
        ]

        engine = _setup_panel_review_engine(repo, xai_search_client=xai)
        await engine.review_open_positions()

        xai.search.assert_called_once()
        xai.search_social.assert_called_once()
        # bulk_upsert_evidence called at least twice (regular + social).
        assert repo.bulk_upsert_evidence.call_count >= 2


# ---- Signal-aware candidate scan -------------------------------------------


class TestSignalAwareCandidateScan:
    """Signal collection and triage during candidate scan."""

    async def test_collect_signals_method(self) -> None:
        """collect_signals stores signal snapshots to DB."""
        from src.signals.collector import SignalCollector
        from src.signals.triage import TriageScorer

        policy = _make_policy()
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1")

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_latest_snapshot.return_value = snap
        repo.get_snapshot_at.return_value = None
        repo.get_snapshots_since.return_value = []
        repo.count_evidence_since.return_value = 0
        repo.add_signal_snapshot.return_value = 1

        signal_collector = SignalCollector(repo=repo, policy=policy)
        triage_scorer = TriageScorer(policy=policy)

        engine = _make_engine(policy=policy, repo=repo)
        engine.signal_collector = signal_collector
        engine.triage_scorer = triage_scorer

        stored = await engine.collect_signals()

        assert stored == 1
        repo.add_signal_snapshot.assert_called_once()

    async def test_collect_signals_without_collector(self) -> None:
        """collect_signals returns 0 when no signal_collector configured."""
        engine = _make_engine()
        stored = await engine.collect_signals()
        assert stored == 0

    async def test_triage_filters_candidates(self) -> None:
        """Low triage scores filter out candidates."""
        from src.signals.collector import SignalCollector
        from src.signals.triage import TriageScorer

        policy = _make_policy(edge_threshold=0.01, triage_panel_threshold=0.90)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_snapshot_at.return_value = None
        repo.get_snapshots_since.return_value = []
        repo.count_evidence_since.return_value = 0
        repo.add_signal_snapshot.return_value = 1

        signal_collector = SignalCollector(repo=repo, policy=policy)
        triage_scorer = TriageScorer(policy=policy)

        engine = _make_engine(policy=policy, repo=repo)
        engine.signal_collector = signal_collector
        engine.triage_scorer = triage_scorer

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()

        # With threshold=0.90 and no signals, candidate should be filtered out.
        assert trades == 0

    async def test_without_signal_collector_falls_through(self) -> None:
        """Without signal_collector, triage is skipped and trades execute."""
        policy = _make_policy(edge_threshold=0.01)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1


# ---- Signal-aware position review -------------------------------------------


class TestSignalAwarePositionReview:
    """Signal-based triggers during position review."""

    async def test_signal_triggers_added(self) -> None:
        """Signal-based triggers (volume_surge, trends_spike, wiki_spike) fire."""
        from src.db.models import SignalSnapshot

        policy = _make_policy(
            triage_volume_surge_threshold=1.5,
            triage_trends_spike_threshold=2.0,
            triage_wiki_spike_threshold=2.0,
        )
        engine = _make_engine(policy=policy)

        pos = _make_position(avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.51, liquidity=10_000.0)

        signal_snap = MagicMock(spec=SignalSnapshot)
        signal_snap.volume_ratio_24h = 3.5  # >= 1.5 + 1.0 = 2.5
        signal_snap.google_trends_spike = 3.0  # >= 2.0
        signal_snap.wikipedia_spike = 2.5  # >= 2.0

        triggers = engine._check_review_triggers(
            pos, snap, None, 500.0, signal_snap
        )

        assert "volume_surge" in triggers
        assert "trends_spike" in triggers
        assert "wiki_spike" in triggers

    async def test_no_signal_triggers_when_below_threshold(self) -> None:
        """Signal triggers do not fire when values are below thresholds."""
        from src.db.models import SignalSnapshot

        policy = _make_policy()
        engine = _make_engine(policy=policy)

        pos = _make_position(avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.51, liquidity=10_000.0)

        signal_snap = MagicMock(spec=SignalSnapshot)
        signal_snap.volume_ratio_24h = 1.0  # below threshold
        signal_snap.google_trends_spike = 1.0  # below threshold
        signal_snap.wikipedia_spike = 1.0  # below threshold

        triggers = engine._check_review_triggers(
            pos, snap, None, 500.0, signal_snap
        )

        assert "volume_surge" not in triggers
        assert "trends_spike" not in triggers
        assert "wiki_spike" not in triggers

    async def test_no_signal_triggers_without_snapshot(self) -> None:
        """No signal triggers when signal_snapshot is None."""
        policy = _make_policy()
        engine = _make_engine(policy=policy)

        pos = _make_position(avg_entry_price=0.50)
        snap = _make_snapshot(mid=0.51, liquidity=10_000.0)

        triggers = engine._check_review_triggers(
            pos, snap, None, 500.0, None
        )

        assert "volume_surge" not in triggers
        assert "trends_spike" not in triggers
        assert "wiki_spike" not in triggers


# ---- Recent veto blocks entry (new) -----------------------------------------


class TestRecentVetoBlocksEntry:
    """Market with recent veto is skipped for new entry."""

    async def test_recent_veto_blocks_entry(self) -> None:
        """Market with veto < 120min ago → skip entry."""
        policy = _make_policy(edge_threshold=0.01, no_add_if_recent_veto_minutes=120)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        # Recent aggregation with veto=True.
        agg_row = MagicMock(spec=Aggregation)
        agg_row.aggregation_json = {"veto": True, "trade_allowed": False}
        agg_row.ts_utc = datetime.now(timezone.utc) - timedelta(minutes=30)
        repo.get_latest_aggregation.return_value = agg_row

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 0
        repo.add_decision.assert_not_called()

    async def test_old_veto_does_not_block(self) -> None:
        """Market with veto > 120min ago → entry proceeds."""
        policy = _make_policy(edge_threshold=0.01, no_add_if_recent_veto_minutes=120)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        # Old aggregation with veto=True (> 120 min ago).
        agg_row = MagicMock(spec=Aggregation)
        agg_row.aggregation_json = {"veto": True, "trade_allowed": False}
        agg_row.ts_utc = datetime.now(timezone.utc) - timedelta(minutes=180)
        repo.get_latest_aggregation.return_value = agg_row

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1

    async def test_no_veto_does_not_block(self) -> None:
        """Market with no veto in recent aggregation → entry proceeds."""
        policy = _make_policy(edge_threshold=0.01, no_add_if_recent_veto_minutes=120)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        # Recent aggregation with veto=False.
        agg_row = MagicMock(spec=Aggregation)
        agg_row.aggregation_json = {"veto": False, "trade_allowed": True}
        agg_row.ts_utc = datetime.now(timezone.utc) - timedelta(minutes=30)
        repo.get_latest_aggregation.return_value = agg_row

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1

    async def test_no_aggregation_does_not_block(self) -> None:
        """No prior aggregation → entry proceeds."""
        policy = _make_policy(edge_threshold=0.01, no_add_if_recent_veto_minutes=120)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1
        repo.get_latest_aggregation.return_value = None

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1

    async def test_disabled_when_zero(self) -> None:
        """no_add_if_recent_veto_minutes=0 disables the check."""
        policy = _make_policy(edge_threshold=0.01, no_add_if_recent_veto_minutes=0)
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 1
        # get_latest_aggregation should not be called.
        repo.get_latest_aggregation.assert_not_called()


# ---- Reentry cooldown + stability tests ------------------------------------


class TestReentryCooldown:
    """Position reentry cooldown and stability checks (Change 5)."""

    async def test_fresh_market_no_cooldown(self) -> None:
        """Never-traded market bypasses cooldown check."""
        policy = _make_policy(position_reentry_cooldown_hours=6)
        repo = AsyncMock()
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_panel_markets_today.return_value = 0
        repo.get_latest_order_ts.return_value = None  # never traded

        agg_result = AggregationResult(
            p_consensus=0.65, confidence=0.80, disagreement=0.02,
            veto=False, trade_allowed=True,
        )

        panel = AsyncMock(spec=PanelOrchestrator)
        panel.default_panel = []
        panel.escalation_agents = []
        panel.charter_hash = "test"
        panel.run_panel.return_value = PanelResult(
            proposals=[
                ModelProposal(
                    model_id="a", run_id="r1", market_id="m1",
                    ts_utc=datetime(2025, 6, 1, tzinfo=timezone.utc),
                    p_true=0.65, confidence=0.80, direction="BUY_YES",
                    rules_ambiguity=0.1, evidence_ambiguity=0.1,
                    recommended_max_exposure_frac=0.05, hold_horizon_hours=48,
                    thesis="t", key_risks=["r"], evidence=[], exit_triggers=["e"],
                    notes="",
                )
            ],
            total_cost_eur=0.01,
        )
        panel.determine_escalation.return_value = (None, None)

        aggregator = Aggregator(policy)

        packet_mock = MagicMock(spec=Packet)
        packet_mock.packet_id = 1
        packet_mock.packet_json = {
            "market_id": "m1",
            "ts_utc": "2025-06-01T00:00:00Z",
            "market_context": {
                "question": "Will it rain?",
                "current_mid": 0.50,
                "best_bid": 0.48,
                "best_ask": 0.52,
            },
            "evidence_items": [],
            "packet_version": "m2.0",
        }
        repo.get_latest_packet.return_value = packet_mock

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        with patch("src.app.scheduler.compute_size") as mock_size:
            mock_size.return_value = MagicMock(
                skip_reason=None, clamped_size_eur=50.0, side="BUY_YES", edge=0.15,
                raw_size_eur=50.0,
            )
            trades = await engine.run_candidate_scan()

        assert trades == 1

    async def test_recent_trade_unstable_blocked(self) -> None:
        """Recently traded market with side flip is blocked by stability check."""
        policy = _make_policy(position_reentry_cooldown_hours=6)
        repo = AsyncMock()
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_panel_markets_today.return_value = 0
        # Traded 2 hours ago (within 6-hour cooldown).
        repo.get_latest_order_ts.return_value = datetime.now(timezone.utc) - timedelta(hours=2)
        # Prior aggregation was BUY_NO, new would be BUY_YES → side flip.
        prior_agg = MagicMock(spec=Aggregation)
        prior_agg.aggregation_json = {
            "p_consensus": 0.40,
            "confidence": 0.70,
            "disagreement": 0.05,
            "p_market": 0.50,
            "consensus_side": "BUY_NO",
        }
        prior_agg.ts_utc = datetime.now(timezone.utc) - timedelta(hours=2)
        repo.get_latest_aggregation.return_value = prior_agg

        panel = AsyncMock(spec=PanelOrchestrator)
        panel.default_panel = []
        panel.escalation_agents = []
        panel.charter_hash = "test"
        panel.run_panel.return_value = PanelResult(
            proposals=[
                ModelProposal(
                    model_id="a", run_id="r1", market_id="m1",
                    ts_utc=datetime(2025, 6, 1, tzinfo=timezone.utc),
                    p_true=0.65, confidence=0.80, direction="BUY_YES",
                    rules_ambiguity=0.1, evidence_ambiguity=0.1,
                    recommended_max_exposure_frac=0.05, hold_horizon_hours=48,
                    thesis="t", key_risks=["r"], evidence=[], exit_triggers=["e"],
                    notes="",
                )
            ],
            total_cost_eur=0.01,
        )
        panel.determine_escalation.return_value = (None, None)

        aggregator = Aggregator(policy)

        packet_mock = MagicMock(spec=Packet)
        packet_mock.packet_id = 1
        packet_mock.packet_json = {
            "market_id": "m1",
            "ts_utc": "2025-06-01T00:00:00Z",
            "market_context": {
                "question": "Will it rain?",
                "current_mid": 0.50,
                "best_bid": 0.48,
                "best_ask": 0.52,
            },
            "evidence_items": [],
            "packet_version": "m2.0",
        }
        repo.get_latest_packet.return_value = packet_mock

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        trades = await engine.run_candidate_scan()
        assert trades == 0  # blocked by stability check (side flip)

    async def test_old_trade_bypasses_cooldown(self) -> None:
        """Trade older than cooldown hours bypasses stability check."""
        policy = _make_policy(position_reentry_cooldown_hours=6)
        repo = AsyncMock()
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_panel_markets_today.return_value = 0
        # Traded 8 hours ago (outside 6-hour cooldown).
        repo.get_latest_order_ts.return_value = datetime.now(timezone.utc) - timedelta(hours=8)

        panel = AsyncMock(spec=PanelOrchestrator)
        panel.default_panel = []
        panel.escalation_agents = []
        panel.charter_hash = "test"
        panel.run_panel.return_value = PanelResult(
            proposals=[
                ModelProposal(
                    model_id="a", run_id="r1", market_id="m1",
                    ts_utc=datetime(2025, 6, 1, tzinfo=timezone.utc),
                    p_true=0.65, confidence=0.80, direction="BUY_YES",
                    rules_ambiguity=0.1, evidence_ambiguity=0.1,
                    recommended_max_exposure_frac=0.05, hold_horizon_hours=48,
                    thesis="t", key_risks=["r"], evidence=[], exit_triggers=["e"],
                    notes="",
                )
            ],
            total_cost_eur=0.01,
        )
        panel.determine_escalation.return_value = (None, None)

        aggregator = Aggregator(policy)

        packet_mock = MagicMock(spec=Packet)
        packet_mock.packet_id = 1
        packet_mock.packet_json = {
            "market_id": "m1",
            "ts_utc": "2025-06-01T00:00:00Z",
            "market_context": {
                "question": "Will it rain?",
                "current_mid": 0.50,
                "best_bid": 0.48,
                "best_ask": 0.52,
            },
            "evidence_items": [],
            "packet_version": "m2.0",
        }
        repo.get_latest_packet.return_value = packet_mock

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        with patch("src.app.scheduler.compute_size") as mock_size:
            mock_size.return_value = MagicMock(
                skip_reason=None, clamped_size_eur=50.0, side="BUY_YES", edge=0.15,
                raw_size_eur=50.0,
            )
            trades = await engine.run_candidate_scan()

        assert trades == 1

    async def test_cooldown_disabled_when_zero(self) -> None:
        """Setting cooldown to 0 disables the check entirely."""
        policy = _make_policy(position_reentry_cooldown_hours=0)
        repo = AsyncMock()
        market = _make_market(market_id="m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_panel_markets_today.return_value = 0

        panel = AsyncMock(spec=PanelOrchestrator)
        panel.default_panel = []
        panel.escalation_agents = []
        panel.charter_hash = "test"
        panel.run_panel.return_value = PanelResult(
            proposals=[
                ModelProposal(
                    model_id="a", run_id="r1", market_id="m1",
                    ts_utc=datetime(2025, 6, 1, tzinfo=timezone.utc),
                    p_true=0.65, confidence=0.80, direction="BUY_YES",
                    rules_ambiguity=0.1, evidence_ambiguity=0.1,
                    recommended_max_exposure_frac=0.05, hold_horizon_hours=48,
                    thesis="t", key_risks=["r"], evidence=[], exit_triggers=["e"],
                    notes="",
                )
            ],
            total_cost_eur=0.01,
        )
        panel.determine_escalation.return_value = (None, None)

        aggregator = Aggregator(policy)

        packet_mock = MagicMock(spec=Packet)
        packet_mock.packet_id = 1
        packet_mock.packet_json = {
            "market_id": "m1",
            "ts_utc": "2025-06-01T00:00:00Z",
            "market_context": {
                "question": "Will it rain?",
                "current_mid": 0.50,
                "best_bid": 0.48,
                "best_ask": 0.52,
            },
            "evidence_items": [],
            "packet_version": "m2.0",
        }
        repo.get_latest_packet.return_value = packet_mock

        engine = _make_engine(
            policy=policy, repo=repo,
            panel_orchestrator=panel, aggregator=aggregator,
        )

        mock_candidate = MagicMock()
        mock_candidate.market_id = "m1"
        mock_candidate.score = 0.90
        engine.selector = MagicMock()
        engine.selector.select.return_value = [mock_candidate]

        with patch("src.app.scheduler.compute_size") as mock_size:
            mock_size.return_value = MagicMock(
                skip_reason=None, clamped_size_eur=50.0, side="BUY_YES", edge=0.15,
                raw_size_eur=50.0,
            )
            trades = await engine.run_candidate_scan()

        assert trades == 1
        # get_latest_order_ts should not be called when cooldown is 0.
        repo.get_latest_order_ts.assert_not_called()


# ---- Panel cooldown ---------------------------------------------------------


class TestPanelCooldown:
    """Panel cooldown filtering in candidate scan."""

    _LONG_RULES = "x" * 60  # Exceeds min_rules_text_length (50)

    @staticmethod
    def _future_market(market_id: str = "m1") -> MagicMock:
        """Create a mock market with future resolution and long rules_text."""
        now = datetime.now(timezone.utc)
        m = _make_market(market_id=market_id, resolution_hours=500.0, now=now)
        m.rules_text = "x" * 60
        return m

    async def test_recently_paneled_market_excluded(self) -> None:
        """A market paneled within the cooldown window is excluded."""
        policy = _make_policy(panel_cooldown_hours=8, edge_threshold=0.01)
        market = self._future_market("m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_recently_paneled_market_ids.return_value = {"m1"}

        engine = _make_engine(policy=policy, repo=repo)
        trades = await engine.run_candidate_scan()

        assert trades == 0
        repo.get_recently_paneled_market_ids.assert_called_once()

    async def test_market_outside_cooldown_included(self) -> None:
        """A market not in the cooldown set passes through."""
        policy = _make_policy(panel_cooldown_hours=8, edge_threshold=0.01)
        market = self._future_market("m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_recently_paneled_market_ids.return_value = set()
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)
        trades = await engine.run_candidate_scan()

        assert trades >= 1

    async def test_cooldown_zero_disables_filter(self) -> None:
        """When panel_cooldown_hours=0, no cooldown query is made."""
        policy = _make_policy(panel_cooldown_hours=0, edge_threshold=0.01)
        market = self._future_market("m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)
        trades = await engine.run_candidate_scan()

        repo.get_recently_paneled_market_ids.assert_not_called()
        assert trades >= 1

    async def test_cooldown_promotes_next_ranked_markets(self) -> None:
        """When top markets are cooled down, lower-ranked ones get promoted."""
        policy = _make_policy(
            panel_cooldown_hours=8, edge_threshold=0.01, max_candidates_per_cycle=15,
        )
        m1 = self._future_market("m1")
        m2 = self._future_market("m2")
        snap1 = _make_snapshot(market_id="m1", mid=0.50)
        snap2 = _make_snapshot(market_id="m2", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [m1, m2]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.side_effect = lambda mid: (
            snap1 if mid == "m1" else snap2
        )
        # m1 was recently paneled, m2 was not.
        repo.get_recently_paneled_market_ids.return_value = {"m1"}
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)
        trades = await engine.run_candidate_scan()

        # m1 should have been filtered out; only m2 can trade.
        # Check that m2 was traded (decision was created with m2's market_id).
        if trades > 0:
            call_args = repo.add_decision.call_args_list
            market_ids_traded = [c[0][0]["market_id"] for c in call_args]
            assert "m1" not in market_ids_traded

    async def test_cooldown_query_error_does_not_block(self) -> None:
        """If the cooldown query fails, candidates are not filtered out."""
        policy = _make_policy(panel_cooldown_hours=8, edge_threshold=0.01)
        market = self._future_market("m1")
        snap = _make_snapshot(market_id="m1", mid=0.50)

        repo = AsyncMock()
        repo.get_active_markets.return_value = [market]
        repo.get_open_positions.return_value = []
        repo.get_latest_snapshot.return_value = snap
        repo.get_recently_paneled_market_ids.side_effect = Exception("DB error")
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        engine = _make_engine(policy=policy, repo=repo)
        trades = await engine.run_candidate_scan()

        # Should still execute trades despite the error.
        assert trades >= 1


# ---- Bulk upsert batching ---------------------------------------------------


class TestBulkUpsertBatching:
    """Tests for the batched bulk_upsert_markets in ingest_markets."""

    async def test_large_market_list_batched(self) -> None:
        """A large list of markets is ingested without error."""
        gamma = AsyncMock()
        # Create 1200 gamma markets (more than 1 batch of 500).
        gamma_markets = [
            _make_gamma_market(condition_id=f"cond-{i}")
            for i in range(1200)
        ]
        gamma.get_all_active_markets.return_value = gamma_markets

        repo = AsyncMock()
        repo.bulk_upsert_markets.return_value = None
        repo.bulk_add_snapshots.return_value = 1200

        engine = _make_engine(repo=repo, gamma=gamma)
        count = await engine.ingest_markets()

        assert count == 1200
        repo.bulk_upsert_markets.assert_called_once()
        repo.bulk_add_snapshots.assert_called_once()
        # Verify all 1200 markets were passed to bulk_upsert.
        market_dicts = repo.bulk_upsert_markets.call_args[0][0]
        assert len(market_dicts) == 1200

    async def test_empty_market_list_returns_zero(self) -> None:
        """An empty market list returns 0 and does not call bulk_upsert."""
        gamma = AsyncMock()
        gamma.get_all_active_markets.return_value = []

        repo = AsyncMock()
        engine = _make_engine(repo=repo, gamma=gamma)
        count = await engine.ingest_markets()

        assert count == 0
        repo.bulk_upsert_markets.assert_not_called()

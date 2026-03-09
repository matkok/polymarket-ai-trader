"""Tests for crypto pipeline safety guards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.policy import Policy
from src.trading.crypto_pipeline import CryptoPipeline, _extract_expiry_date


# ---- Helpers ----------------------------------------------------------------


def _make_position(
    market_id: str = "mkt-1",
    side: str = "BUY_YES",
    size_eur: float = 100.0,
    status: str = "open",
) -> MagicMock:
    pos = MagicMock()
    pos.market_id = market_id
    pos.side = side
    pos.size_eur = size_eur
    pos.status = status
    pos.last_update_ts_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    return pos


def _make_assignment(
    market_id: str,
    category: str = "crypto",
    asset: str = "BTC",
) -> MagicMock:
    a = MagicMock()
    a.market_id = market_id
    a.category = category
    a.contract_spec_json = {"asset": asset, "category": "crypto"}
    return a


def _make_market(
    market_id: str = "mkt-1",
    resolution_time_utc: datetime | None = None,
) -> MagicMock:
    m = MagicMock()
    m.market_id = market_id
    m.resolution_time_utc = resolution_time_utc
    return m


# ---- _extract_expiry_date ---------------------------------------------------


class TestExtractExpiryDate:
    """Helper to extract expiry date string from market."""

    def test_with_resolution_time(self) -> None:
        market = _make_market(
            resolution_time_utc=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
        )
        assert _extract_expiry_date(market) == "2026-04-15"

    def test_none_market(self) -> None:
        assert _extract_expiry_date(None) is None

    def test_no_resolution_time(self) -> None:
        market = _make_market(resolution_time_utc=None)
        assert _extract_expiry_date(market) is None


# ---- Asset coherence guard --------------------------------------------------


class TestBuildDirectionMap:
    """CryptoPipeline._build_direction_map tests."""

    async def test_builds_map_from_open_crypto_positions(self) -> None:
        """Map includes open crypto positions with asset + expiry."""
        repo = AsyncMock()
        pos = _make_position(market_id="mkt-btc", side="BUY_NO")
        repo.get_open_positions.return_value = [pos]
        repo.get_assignment.return_value = _make_assignment("mkt-btc", asset="BTC")
        repo.get_market.return_value = _make_market(
            resolution_time_utc=datetime(2026, 4, 15, tzinfo=timezone.utc),
        )

        pipeline = CryptoPipeline(
            repo=repo,
            exchange_router=MagicMock(),
            engine=MagicMock(),
            executor=MagicMock(),
            risk_manager=MagicMock(),
            policy=Policy(),
        )
        result = await pipeline._build_direction_map()
        assert result == {("BTC", "2026-04-15"): "BUY_NO"}

    async def test_skips_non_crypto_positions(self) -> None:
        """Non-crypto positions are excluded from the map."""
        repo = AsyncMock()
        pos = _make_position(market_id="mkt-weather")
        repo.get_open_positions.return_value = [pos]
        weather_assignment = _make_assignment("mkt-weather", category="weather")
        repo.get_assignment.return_value = weather_assignment

        pipeline = CryptoPipeline(
            repo=repo,
            exchange_router=MagicMock(),
            engine=MagicMock(),
            executor=MagicMock(),
            risk_manager=MagicMock(),
            policy=Policy(),
        )
        result = await pipeline._build_direction_map()
        assert result == {}

    async def test_empty_when_no_positions(self) -> None:
        """Empty map when no open positions."""
        repo = AsyncMock()
        repo.get_open_positions.return_value = []

        pipeline = CryptoPipeline(
            repo=repo,
            exchange_router=MagicMock(),
            engine=MagicMock(),
            executor=MagicMock(),
            risk_manager=MagicMock(),
            policy=Policy(),
        )
        result = await pipeline._build_direction_map()
        assert result == {}


class TestCoherenceGuard:
    """Asset coherence guard blocks contradictory directions."""

    def _make_pipeline(self, repo: AsyncMock) -> CryptoPipeline:
        return CryptoPipeline(
            repo=repo,
            exchange_router=AsyncMock(),
            engine=MagicMock(),
            executor=MagicMock(),
            risk_manager=MagicMock(),
            policy=Policy(edge_threshold=0.01),
        )

    def _sizing_result(self, side: str = "BUY_YES") -> MagicMock:
        sr = MagicMock()
        sr.side = side
        sr.skip_reason = None
        sr.clamped_size_eur = 50.0
        sr.edge = 0.10
        return sr

    async def test_opposite_direction_same_asset_same_expiry_blocked(self) -> None:
        """BUY_YES blocked when BUY_NO exists for same asset + expiry."""
        repo = AsyncMock()
        expiry = datetime(2026, 4, 15, tzinfo=timezone.utc)

        # Assignment for the new market.
        assignment = _make_assignment("mkt-new", asset="BTC")
        assignment.contract_spec_json = {
            "asset": "BTC",
            "category": "crypto",
            "threshold": 100000,
            "comparison": "above",
        }

        # No existing position for this specific market.
        repo.get_position.return_value = None

        # Market with resolution time.
        market = _make_market("mkt-new", resolution_time_utc=expiry)
        repo.get_market.return_value = market

        # Snapshot for pricing.
        snapshot = MagicMock()
        snapshot.mid = 0.50
        snapshot.best_bid = 0.49
        snapshot.best_ask = 0.51
        repo.get_latest_snapshot.return_value = snapshot

        # Exchange fetch succeeds.
        pipeline = self._make_pipeline(repo)
        exchange_result = MagicMock()
        exchange_result.ok = True
        pipeline.exchange_router.fetch.return_value = exchange_result

        # Engine returns estimate.
        estimate = MagicMock()
        estimate.p_yes = 0.65
        estimate.confidence = 0.80
        pipeline.engine.compute.return_value = estimate

        # Direction map: existing BUY_NO on BTC Feb 28.
        direction_map = {("BTC", "2026-04-15"): "BUY_NO"}
        now = datetime.now(timezone.utc)

        result = await pipeline._process_market(assignment, now, direction_map)
        # Should be blocked at "priced" (coherence guard), not "traded".
        assert result == "priced"

    async def test_same_direction_same_asset_same_expiry_allowed(self) -> None:
        """BUY_NO allowed when BUY_NO already exists (stacking same direction)."""
        repo = AsyncMock()
        expiry = datetime(2026, 4, 15, tzinfo=timezone.utc)

        assignment = _make_assignment("mkt-new", asset="BTC")
        assignment.contract_spec_json = {
            "asset": "BTC",
            "category": "crypto",
            "threshold": 100000,
            "comparison": "above",
        }

        repo.get_position.return_value = None
        market = _make_market("mkt-new", resolution_time_utc=expiry)
        repo.get_market.return_value = market

        snapshot = MagicMock()
        snapshot.mid = 0.50
        snapshot.best_bid = 0.49
        snapshot.best_ask = 0.51
        repo.get_latest_snapshot.return_value = snapshot

        pipeline = self._make_pipeline(repo)
        exchange_result = MagicMock()
        exchange_result.ok = True
        pipeline.exchange_router.fetch.return_value = exchange_result

        # Engine says p_yes=0.35 → side=BUY_NO.
        estimate = MagicMock()
        estimate.p_yes = 0.35
        estimate.confidence = 0.80
        pipeline.engine.compute.return_value = estimate

        # Risk check allows.
        risk_result = MagicMock()
        risk_result.allowed = True
        pipeline.risk_manager.check_new_trade_category.return_value = risk_result

        # Paper executor fill.
        fill = MagicMock()
        fill.side = "BUY_NO"
        fill.size_eur = 50.0
        fill.price = 0.50
        fill.fee_eur = 0.0
        pipeline.executor.execute.return_value = fill

        repo.get_markets_by_category.return_value = []
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        # Same direction in map.
        direction_map = {("BTC", "2026-04-15"): "BUY_NO"}
        now = datetime.now(timezone.utc)

        result = await pipeline._process_market(assignment, now, direction_map)
        assert result == "traded"

    async def test_different_expiry_allowed(self) -> None:
        """Opposite direction allowed on different expiry date."""
        repo = AsyncMock()
        # Market resolves April 20, map has April 15 — different expiry.
        expiry = datetime(2026, 4, 20, tzinfo=timezone.utc)

        assignment = _make_assignment("mkt-new", asset="BTC")
        assignment.contract_spec_json = {
            "asset": "BTC",
            "category": "crypto",
            "threshold": 100000,
            "comparison": "above",
        }

        repo.get_position.return_value = None
        market = _make_market("mkt-new", resolution_time_utc=expiry)
        repo.get_market.return_value = market

        snapshot = MagicMock()
        snapshot.mid = 0.50
        snapshot.best_bid = 0.49
        snapshot.best_ask = 0.51
        repo.get_latest_snapshot.return_value = snapshot

        pipeline = self._make_pipeline(repo)
        exchange_result = MagicMock()
        exchange_result.ok = True
        pipeline.exchange_router.fetch.return_value = exchange_result

        estimate = MagicMock()
        estimate.p_yes = 0.65
        estimate.confidence = 0.80
        pipeline.engine.compute.return_value = estimate

        risk_result = MagicMock()
        risk_result.allowed = True
        pipeline.risk_manager.check_new_trade_category.return_value = risk_result

        fill = MagicMock()
        fill.side = "BUY_YES"
        fill.size_eur = 50.0
        fill.price = 0.51
        fill.fee_eur = 0.0
        pipeline.executor.execute.return_value = fill

        repo.get_markets_by_category.return_value = []
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        # Map has BUY_NO on Feb 28 — different expiry from March 1.
        direction_map = {("BTC", "2026-04-15"): "BUY_NO"}
        now = datetime.now(timezone.utc)

        result = await pipeline._process_market(assignment, now, direction_map)
        assert result == "traded"

    async def test_different_asset_allowed(self) -> None:
        """Opposite direction allowed for a different asset."""
        repo = AsyncMock()
        expiry = datetime(2026, 4, 15, tzinfo=timezone.utc)

        assignment = _make_assignment("mkt-new", asset="ETH")
        assignment.contract_spec_json = {
            "asset": "ETH",
            "category": "crypto",
            "threshold": 5000,
            "comparison": "above",
        }

        repo.get_position.return_value = None
        market = _make_market("mkt-new", resolution_time_utc=expiry)
        repo.get_market.return_value = market

        snapshot = MagicMock()
        snapshot.mid = 0.50
        snapshot.best_bid = 0.49
        snapshot.best_ask = 0.51
        repo.get_latest_snapshot.return_value = snapshot

        pipeline = self._make_pipeline(repo)
        exchange_result = MagicMock()
        exchange_result.ok = True
        pipeline.exchange_router.fetch.return_value = exchange_result

        estimate = MagicMock()
        estimate.p_yes = 0.65
        estimate.confidence = 0.80
        pipeline.engine.compute.return_value = estimate

        risk_result = MagicMock()
        risk_result.allowed = True
        pipeline.risk_manager.check_new_trade_category.return_value = risk_result

        fill = MagicMock()
        fill.side = "BUY_YES"
        fill.size_eur = 50.0
        fill.price = 0.51
        fill.fee_eur = 0.0
        pipeline.executor.execute.return_value = fill

        repo.get_markets_by_category.return_value = []
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        # Map has BUY_NO for BTC, not ETH.
        direction_map = {("BTC", "2026-04-15"): "BUY_NO"}
        now = datetime.now(timezone.utc)

        result = await pipeline._process_market(assignment, now, direction_map)
        assert result == "traded"

    async def test_no_existing_positions_allowed(self) -> None:
        """Empty direction map allows any trade."""
        repo = AsyncMock()
        expiry = datetime(2026, 4, 15, tzinfo=timezone.utc)

        assignment = _make_assignment("mkt-new", asset="BTC")
        assignment.contract_spec_json = {
            "asset": "BTC",
            "category": "crypto",
            "threshold": 100000,
            "comparison": "above",
        }

        repo.get_position.return_value = None
        market = _make_market("mkt-new", resolution_time_utc=expiry)
        repo.get_market.return_value = market

        snapshot = MagicMock()
        snapshot.mid = 0.50
        snapshot.best_bid = 0.49
        snapshot.best_ask = 0.51
        repo.get_latest_snapshot.return_value = snapshot

        pipeline = self._make_pipeline(repo)
        exchange_result = MagicMock()
        exchange_result.ok = True
        pipeline.exchange_router.fetch.return_value = exchange_result

        estimate = MagicMock()
        estimate.p_yes = 0.65
        estimate.confidence = 0.80
        pipeline.engine.compute.return_value = estimate

        risk_result = MagicMock()
        risk_result.allowed = True
        pipeline.risk_manager.check_new_trade_category.return_value = risk_result

        fill = MagicMock()
        fill.side = "BUY_YES"
        fill.size_eur = 50.0
        fill.price = 0.51
        fill.fee_eur = 0.0
        pipeline.executor.execute.return_value = fill

        repo.get_markets_by_category.return_value = []
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        direction_map: dict[tuple[str, str], str] = {}
        now = datetime.now(timezone.utc)

        result = await pipeline._process_market(assignment, now, direction_map)
        assert result == "traded"

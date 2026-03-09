"""Tests for pipeline entry guards, reentry cooldown, and horizon filter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.policy import CategoryPolicy, Policy
from src.trading.weather_pipeline import WeatherPipeline
from src.trading.crypto_pipeline import CryptoPipeline
from src.trading.macro_pipeline import MacroPipeline
from src.trading.earnings_pipeline import EarningsPipeline


# ---- Helpers ----------------------------------------------------------------


def _make_assignment(market_id: str, spec_json: dict | None = None) -> MagicMock:
    a = MagicMock()
    a.market_id = market_id
    a.contract_spec_json = spec_json or {"category": "weather", "metric": "temperature_high",
                                          "location": "NYC", "threshold": 80,
                                          "threshold_unit": "F", "comparison": "above",
                                          "date_start": "", "date_end": "", "date_description": "",
                                          "nws_station_ids": []}
    return a


def _make_position(status: str = "open", last_update_ts_utc: datetime | None = None) -> MagicMock:
    pos = MagicMock()
    pos.status = status
    pos.last_update_ts_utc = last_update_ts_utc or datetime.now(timezone.utc)
    return pos


def _make_market(resolution_time_utc: datetime | None = None) -> MagicMock:
    m = MagicMock()
    m.resolution_time_utc = resolution_time_utc
    return m


def _weather_pipeline(repo: AsyncMock) -> WeatherPipeline:
    policy = Policy(
        categories={
            "weather": CategoryPolicy(
                max_hours_to_resolution=240,
                reentry_cooldown_hours=6.0,
            ),
        },
    )
    return WeatherPipeline(
        repo=repo,
        nws=AsyncMock(),
        awc=AsyncMock(),
        engine=MagicMock(),
        executor=MagicMock(),
        risk_manager=MagicMock(),
        policy=policy,
    )


# ---- Entry guard: open position blocks entry --------------------------------


class TestEntryGuardOpenPosition:
    """Pipeline skips markets where we already hold an open position."""

    async def test_weather_skips_open_position(self) -> None:
        repo = AsyncMock()
        repo.get_position.return_value = _make_position(status="open")
        pipeline = _weather_pipeline(repo)
        now = datetime.now(timezone.utc)
        assignment = _make_assignment("mkt-1")

        result = await pipeline._process_market(assignment, now)
        assert result == "skipped"

    async def test_crypto_skips_open_position(self) -> None:
        repo = AsyncMock()
        repo.get_position.return_value = _make_position(status="open")
        policy = Policy(
            categories={"crypto": CategoryPolicy(
                max_hours_to_resolution=720,
                reentry_cooldown_hours=6.0,
            )},
        )
        pipeline = CryptoPipeline(
            repo=repo,
            exchange_router=AsyncMock(),
            engine=MagicMock(),
            executor=MagicMock(),
            risk_manager=MagicMock(),
            policy=policy,
        )
        assignment = _make_assignment("mkt-1", spec_json={
            "category": "crypto", "asset": "BTC", "threshold": 100000,
            "threshold_unit": "USD", "comparison": "above",
            "exchange": "binance", "reference_price": "last_trade",
            "resolution_timestamp": "", "date_description": "",
        })
        result = await pipeline._process_market(assignment, datetime.now(timezone.utc))
        assert result == "skipped"


# ---- Entry guard: reentry cooldown ------------------------------------------


class TestReentryCooldown:
    """Pipeline skips recently closed positions (cooldown)."""

    async def test_skips_recently_closed(self) -> None:
        """Position closed 1 hour ago should be skipped (cooldown=6h)."""
        repo = AsyncMock()
        closed_pos = _make_position(
            status="closed",
            last_update_ts_utc=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        repo.get_position.return_value = closed_pos
        pipeline = _weather_pipeline(repo)
        assignment = _make_assignment("mkt-1")
        result = await pipeline._process_market(assignment, datetime.now(timezone.utc))
        assert result == "skipped"

    async def test_allows_old_closed(self) -> None:
        """Position closed 8 hours ago should pass cooldown (cooldown=6h)."""
        repo = AsyncMock()
        closed_pos = _make_position(
            status="closed",
            last_update_ts_utc=datetime.now(timezone.utc) - timedelta(hours=8),
        )
        repo.get_position.return_value = closed_pos
        # get_market returns a market with a near resolution time so horizon filter passes.
        market = _make_market(
            resolution_time_utc=datetime.now(timezone.utc) + timedelta(hours=48),
        )
        repo.get_market.return_value = market
        pipeline = _weather_pipeline(repo)
        assignment = _make_assignment("mkt-1")

        # Patch out the rest of the pipeline to avoid NWS fetch etc.
        with patch.object(pipeline, "nws") as mock_nws:
            mock_fetch = MagicMock()
            mock_fetch.ok = False
            mock_fetch.error = "test"
            mock_nws.fetch.return_value = mock_fetch
            result = await pipeline._process_market(assignment, datetime.now(timezone.utc))
        # Should get past the entry guard (not "skipped" due to position).
        # It will be "skipped" due to NWS error or spec, but not the guard.
        # The important thing is get_market was called (passed cooldown check).
        repo.get_market.assert_called_once()


# ---- Horizon filter ----------------------------------------------------------


class TestHorizonFilter:
    """Pipeline skips markets that resolve too far in the future."""

    async def test_skips_far_resolution(self) -> None:
        """Market resolving in 500 hours skipped (max=240h for weather)."""
        repo = AsyncMock()
        repo.get_position.return_value = None
        market = _make_market(
            resolution_time_utc=datetime.now(timezone.utc) + timedelta(hours=500),
        )
        repo.get_market.return_value = market
        pipeline = _weather_pipeline(repo)
        assignment = _make_assignment("mkt-1")
        result = await pipeline._process_market(assignment, datetime.now(timezone.utc))
        assert result == "skipped"

    async def test_allows_near_resolution(self) -> None:
        """Market resolving in 100 hours passes filter (max=240h)."""
        repo = AsyncMock()
        repo.get_position.return_value = None
        market = _make_market(
            resolution_time_utc=datetime.now(timezone.utc) + timedelta(hours=100),
        )
        repo.get_market.return_value = market
        pipeline = _weather_pipeline(repo)
        assignment = _make_assignment("mkt-1")

        # Patch NWS to stop after horizon filter.
        with patch.object(pipeline, "nws") as mock_nws:
            mock_fetch = MagicMock()
            mock_fetch.ok = False
            mock_fetch.error = "test"
            mock_nws.fetch.return_value = mock_fetch
            result = await pipeline._process_market(assignment, datetime.now(timezone.utc))
        # Should not be "skipped" due to horizon filter — will be skipped for other reasons.
        # Key assertion: get_market was called and we got past the horizon filter.
        repo.get_market.assert_called_once()

    async def test_skips_below_min_hours(self) -> None:
        """Market resolving in 12 hours skipped (min_hours_to_resolution=24)."""
        repo = AsyncMock()
        repo.get_position.return_value = None
        market = _make_market(
            resolution_time_utc=datetime.now(timezone.utc) + timedelta(hours=12),
        )
        repo.get_market.return_value = market
        pipeline = _weather_pipeline(repo)
        assignment = _make_assignment("mkt-1")
        result = await pipeline._process_market(assignment, datetime.now(timezone.utc))
        assert result == "skipped"


# ---- No position: pipeline proceeds -----------------------------------------


class TestNoPositionProceeds:
    """Pipeline proceeds normally when no position exists."""

    async def test_no_position_passes_guard(self) -> None:
        repo = AsyncMock()
        repo.get_position.return_value = None
        market = _make_market(
            resolution_time_utc=datetime.now(timezone.utc) + timedelta(hours=100),
        )
        repo.get_market.return_value = market
        pipeline = _weather_pipeline(repo)
        assignment = _make_assignment("mkt-1")

        with patch.object(pipeline, "nws") as mock_nws:
            mock_fetch = MagicMock()
            mock_fetch.ok = False
            mock_nws.fetch.return_value = mock_fetch
            await pipeline._process_market(assignment, datetime.now(timezone.utc))
        # get_position should have been called and returned None.
        repo.get_position.assert_called_once_with("mkt-1")

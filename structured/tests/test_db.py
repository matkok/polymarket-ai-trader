"""Tests for src.db — models and repository for the structured trader."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import (
    BacktestRun,
    Base,
    CalibrationStat,
    CategoryAssignment,
    CategoryPnLDaily,
    CategoryPortfolioRow,
    Decision,
    EnginePrice,
    Fill,
    Market,
    MarketResolution,
    MarketSnapshot,
    Order,
    Position,
    Resolution,
    SourceObservation,
)
from src.db.repository import Repository


# ---- Models -----------------------------------------------------------------


class TestModelsExist:
    """Verify all 15 tables are defined."""

    def test_core_tables(self) -> None:
        """7 core tables exist."""
        table_names = {t.name for t in Base.metadata.tables.values()}
        core = {
            "markets",
            "market_snapshots",
            "decisions",
            "orders",
            "fills",
            "positions",
            "resolutions",
        }
        assert core.issubset(table_names)

    def test_new_tables(self) -> None:
        """8 new tables exist."""
        table_names = {t.name for t in Base.metadata.tables.values()}
        new_tables = {
            "category_assignments",
            "source_observations",
            "engine_prices",
            "category_portfolios",
            "category_pnl_daily",
            "calibration_stats",
            "backtest_runs",
            "market_resolutions",
        }
        assert new_tables.issubset(table_names)

    def test_total_table_count(self) -> None:
        """15 tables total (7 core + 8 new)."""
        table_count = len(Base.metadata.tables)
        assert table_count == 15


class TestMarketModel:
    """Market model basic construction."""

    def test_market_tablename(self) -> None:
        assert Market.__tablename__ == "markets"


class TestCategoryAssignmentModel:
    """CategoryAssignment model."""

    def test_tablename(self) -> None:
        assert CategoryAssignment.__tablename__ == "category_assignments"


class TestSourceObservationModel:
    """SourceObservation model."""

    def test_tablename(self) -> None:
        assert SourceObservation.__tablename__ == "source_observations"


class TestEnginePriceModel:
    """EnginePrice model."""

    def test_tablename(self) -> None:
        assert EnginePrice.__tablename__ == "engine_prices"


class TestCategoryPortfolioRowModel:
    """CategoryPortfolioRow model."""

    def test_tablename(self) -> None:
        assert CategoryPortfolioRow.__tablename__ == "category_portfolios"


class TestCategoryPnLDailyModel:
    """CategoryPnLDaily model."""

    def test_tablename(self) -> None:
        assert CategoryPnLDaily.__tablename__ == "category_pnl_daily"


class TestCalibrationStatModel:
    """CalibrationStat model."""

    def test_tablename(self) -> None:
        assert CalibrationStat.__tablename__ == "calibration_stats"


class TestBacktestRunModel:
    """BacktestRun model."""

    def test_tablename(self) -> None:
        assert BacktestRun.__tablename__ == "backtest_runs"


class TestMarketResolutionModel:
    """MarketResolution model."""

    def test_tablename(self) -> None:
        assert MarketResolution.__tablename__ == "market_resolutions"


# ---- Repository (mocked session) -------------------------------------------


def _make_mock_session_factory():
    """Create a mock async session factory for unit tests."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_factory = MagicMock()
    mock_factory.return_value = mock_session

    return mock_factory, mock_session


class TestRepositoryInit:
    """Repository can be instantiated."""

    def test_init(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert repo.session_factory is mock_factory


class TestRepositoryCoreMethodsExist:
    """Verify repository has the expected core methods."""

    def test_market_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.upsert_market)
        assert callable(repo.bulk_upsert_markets)
        assert callable(repo.get_active_markets)
        assert callable(repo.get_market)

    def test_snapshot_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.add_snapshot)
        assert callable(repo.get_latest_snapshot)

    def test_position_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.upsert_position)
        assert callable(repo.get_open_positions)
        assert callable(repo.get_position)
        assert callable(repo.get_closed_positions)


class TestRepositoryNewMethodsExist:
    """Verify repository has the new structured trader methods."""

    def test_category_assignment_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.upsert_category_assignment)
        assert callable(repo.get_assignment)
        assert callable(repo.get_markets_by_category)
        assert callable(repo.get_unparsed_markets)

    def test_source_observation_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.add_source_observation)
        assert callable(repo.get_latest_observations)

    def test_engine_price_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.add_engine_price)
        assert callable(repo.get_latest_engine_price)

    def test_category_portfolio_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.upsert_category_portfolio)
        assert callable(repo.get_category_portfolio)

    def test_pnl_and_calibration_methods(self) -> None:
        mock_factory, _ = _make_mock_session_factory()
        repo = Repository(mock_factory)
        assert callable(repo.add_category_pnl_daily)
        assert callable(repo.add_calibration_stat)


class TestRepositoryBulkUpsertMarketsEmpty:
    """bulk_upsert_markets with empty list should no-op."""

    async def test_empty_list_returns_immediately(self) -> None:
        mock_factory, mock_session = _make_mock_session_factory()
        repo = Repository(mock_factory)
        await repo.bulk_upsert_markets([])
        mock_session.execute.assert_not_called()

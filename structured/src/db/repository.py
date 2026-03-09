from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sqlalchemy import func as sa_func

from src.db.models import (
    BacktestRun,
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


class Repository:
    """Async CRUD helpers for the structured trader database."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    async def upsert_market(self, market_data: dict) -> None:
        """Insert a market or update it on conflict."""
        async with self.session_factory() as session:
            stmt = pg_insert(Market).values(**market_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["market_id"],
                set_={
                    key: stmt.excluded[key]
                    for key in market_data
                    if key != "market_id"
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def bulk_upsert_markets(self, markets: list[dict], batch_size: int = 500) -> None:
        """Batch-upsert a list of markets in chunks to stay under PG param limits."""
        if not markets:
            return
        async with self.session_factory() as session:
            for i in range(0, len(markets), batch_size):
                batch = markets[i : i + batch_size]
                stmt = pg_insert(Market).values(batch)
                update_keys = {
                    key for m in batch for key in m if key != "market_id"
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["market_id"],
                    set_={key: stmt.excluded[key] for key in update_keys},
                )
                await session.execute(stmt)
            await session.commit()

    async def get_active_markets(self) -> list[Market]:
        """Return all markets with ``status='active'``."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Market).where(Market.status == "active")
            )
            return list(result.scalars().all())

    async def get_market(self, market_id: str) -> Market | None:
        """Return a single market by id."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Market).where(Market.market_id == market_id)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    async def add_snapshot(self, snapshot_data: dict) -> int:
        """Insert a market snapshot and return its id."""
        async with self.session_factory() as session:
            snapshot = MarketSnapshot(**snapshot_data)
            session.add(snapshot)
            await session.commit()
            return snapshot.snapshot_id

    async def bulk_add_snapshots(self, snapshots: list[dict], batch_size: int = 500) -> int:
        """Bulk-insert market snapshots in batches. Return count inserted."""
        if not snapshots:
            return 0
        async with self.session_factory() as session:
            for i in range(0, len(snapshots), batch_size):
                batch = snapshots[i : i + batch_size]
                session.add_all([MarketSnapshot(**s) for s in batch])
            await session.commit()
        return len(snapshots)

    async def get_latest_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """Return the most recent snapshot for *market_id*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(MarketSnapshot)
                .where(MarketSnapshot.market_id == market_id)
                .order_by(MarketSnapshot.ts_utc.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    async def add_decision(self, decision_data: dict) -> int:
        """Insert a decision and return its id."""
        async with self.session_factory() as session:
            decision = Decision(**decision_data)
            session.add(decision)
            await session.commit()
            return decision.decision_id

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def add_order(self, order_data: dict) -> int:
        """Insert an order and return its id."""
        async with self.session_factory() as session:
            order = Order(**order_data)
            session.add(order)
            await session.commit()
            return order.order_id

    async def get_latest_order_ts(self, market_id: str) -> datetime | None:
        """Return the most recent order timestamp for *market_id*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(sa_func.max(Order.created_ts_utc)).where(
                    Order.market_id == market_id
                )
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    async def add_fill(self, fill_data: dict) -> int:
        """Insert a fill and return its id."""
        async with self.session_factory() as session:
            fill = Fill(**fill_data)
            session.add(fill)
            await session.commit()
            return fill.fill_id

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def upsert_position(self, position_data: dict) -> None:
        """Insert a position or update it on conflict (by market_id)."""
        async with self.session_factory() as session:
            stmt = pg_insert(Position).values(**position_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["market_id"],
                set_={
                    key: stmt.excluded[key]
                    for key in position_data
                    if key != "market_id"
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def get_open_positions(self) -> list[Position]:
        """Return all positions with ``status='open'``."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Position).where(Position.status == "open")
            )
            return list(result.scalars().all())

    async def get_position(self, market_id: str) -> Position | None:
        """Return the position for *market_id*, or ``None``."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Position).where(Position.market_id == market_id)
            )
            return result.scalar_one_or_none()

    async def get_closed_positions(self) -> list[Position]:
        """Return all positions with ``status='closed'``."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Position).where(Position.status == "closed")
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Resolutions
    # ------------------------------------------------------------------

    async def add_resolution(self, resolution_data: dict) -> int:
        """Insert a resolution and return its id."""
        async with self.session_factory() as session:
            resolution = Resolution(**resolution_data)
            session.add(resolution)
            await session.commit()
            return resolution.resolution_id

    async def get_resolutions_since(self, since: datetime) -> list[Resolution]:
        """Return resolutions resolved since *since*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Resolution)
                .where(Resolution.resolved_ts_utc >= since)
                .order_by(Resolution.resolved_ts_utc.desc())
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Snapshots (extended)
    # ------------------------------------------------------------------

    async def get_snapshots_since(
        self, market_id: str, since: datetime
    ) -> list[MarketSnapshot]:
        """Return snapshots for *market_id* since *since*, oldest first."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(MarketSnapshot)
                .where(
                    MarketSnapshot.market_id == market_id,
                    MarketSnapshot.ts_utc >= since,
                )
                .order_by(MarketSnapshot.ts_utc.asc())
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Category Assignments (new)
    # ------------------------------------------------------------------

    async def upsert_category_assignment(self, data: dict) -> int:
        """Insert or update a category assignment. Returns assignment_id."""
        async with self.session_factory() as session:
            assignment = CategoryAssignment(**data)
            session.add(assignment)
            await session.commit()
            return assignment.assignment_id

    async def get_assignment(self, market_id: str) -> CategoryAssignment | None:
        """Return the latest category assignment for *market_id*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(CategoryAssignment)
                .where(CategoryAssignment.market_id == market_id)
                .order_by(CategoryAssignment.created_ts_utc.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_markets_by_category(self, category: str) -> list[CategoryAssignment]:
        """Return all assignments for a given category."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(CategoryAssignment)
                .where(
                    CategoryAssignment.category == category,
                    CategoryAssignment.parse_status == "parsed",
                )
            )
            return list(result.scalars().all())

    async def get_unparsed_markets(self) -> list[Market]:
        """Return active markets without a category assignment."""
        async with self.session_factory() as session:
            # Subquery: market_ids that have assignments.
            assigned_ids = (
                select(CategoryAssignment.market_id)
                .distinct()
                .scalar_subquery()
            )
            result = await session.execute(
                select(Market)
                .where(
                    Market.status == "active",
                    Market.market_id.notin_(assigned_ids),
                )
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Source Observations (new)
    # ------------------------------------------------------------------

    async def add_source_observation(self, data: dict) -> int:
        """Insert a source observation and return its id."""
        async with self.session_factory() as session:
            obs = SourceObservation(**data)
            session.add(obs)
            await session.commit()
            return obs.observation_id

    async def get_latest_observations(
        self, category: str, source_name: str, limit: int = 10
    ) -> list[SourceObservation]:
        """Return the latest observations for a category/source."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SourceObservation)
                .where(
                    SourceObservation.category == category,
                    SourceObservation.source_name == source_name,
                )
                .order_by(SourceObservation.ts_ingested.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Engine Prices (new)
    # ------------------------------------------------------------------

    async def add_engine_price(self, data: dict) -> int:
        """Insert an engine price and return its id."""
        async with self.session_factory() as session:
            price = EnginePrice(**data)
            session.add(price)
            await session.commit()
            return price.engine_price_id

    async def get_latest_engine_price(self, market_id: str) -> EnginePrice | None:
        """Return the most recent engine price for *market_id*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(EnginePrice)
                .where(EnginePrice.market_id == market_id)
                .order_by(EnginePrice.ts_utc.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Category Portfolio (new)
    # ------------------------------------------------------------------

    async def upsert_category_portfolio(self, data: dict) -> None:
        """Insert or update a category portfolio row."""
        async with self.session_factory() as session:
            stmt = pg_insert(CategoryPortfolioRow).values(**data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["category"],
                set_={
                    key: stmt.excluded[key]
                    for key in data
                    if key != "category"
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def get_category_portfolio(self, category: str) -> CategoryPortfolioRow | None:
        """Return the portfolio row for a category."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(CategoryPortfolioRow)
                .where(CategoryPortfolioRow.category == category)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Category PnL Daily (new)
    # ------------------------------------------------------------------

    async def add_category_pnl_daily(self, data: dict) -> int:
        """Insert a daily PnL row and return its id."""
        async with self.session_factory() as session:
            pnl = CategoryPnLDaily(**data)
            session.add(pnl)
            await session.commit()
            return pnl.pnl_id

    # ------------------------------------------------------------------
    # Calibration Stats (new)
    # ------------------------------------------------------------------

    async def add_calibration_stat(self, data: dict) -> int:
        """Insert a calibration stat and return its id."""
        async with self.session_factory() as session:
            stat = CalibrationStat(**data)
            session.add(stat)
            await session.commit()
            return stat.stat_id

    # ------------------------------------------------------------------
    # Engine Prices (batch query)
    # ------------------------------------------------------------------

    async def get_engine_prices_for_markets(
        self, market_ids: list[str]
    ) -> list[EnginePrice]:
        """Fetch the latest engine price per market_id."""
        if not market_ids:
            return []
        async with self.session_factory() as session:
            # Subquery: latest ts_utc per market_id.
            latest_ts = (
                select(
                    EnginePrice.market_id,
                    sa_func.max(EnginePrice.ts_utc).label("max_ts"),
                )
                .where(EnginePrice.market_id.in_(market_ids))
                .group_by(EnginePrice.market_id)
                .subquery()
            )
            result = await session.execute(
                select(EnginePrice).join(
                    latest_ts,
                    (EnginePrice.market_id == latest_ts.c.market_id)
                    & (EnginePrice.ts_utc == latest_ts.c.max_ts),
                )
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Backtest Runs
    # ------------------------------------------------------------------

    async def add_backtest_run(self, data: dict) -> int:
        """Persist a replay/backtest run and return its id."""
        async with self.session_factory() as session:
            run = BacktestRun(**data)
            session.add(run)
            await session.commit()
            return run.backtest_id

    # ------------------------------------------------------------------
    # Category PnL Daily (query)
    # ------------------------------------------------------------------

    async def get_category_pnl_daily(
        self, category: str, pnl_date: datetime
    ) -> CategoryPnLDaily | None:
        """Retrieve daily PnL for a category on a given date."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(CategoryPnLDaily).where(
                    CategoryPnLDaily.category == category,
                    CategoryPnLDaily.pnl_date == pnl_date,
                )
            )
            return result.scalar_one_or_none()

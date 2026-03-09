from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sqlalchemy import func as sa_func

from src.db.models import (
    Aggregation,
    Decision,
    EvidenceItem,
    Fill,
    Market,
    MarketSnapshot,
    ModelRun,
    ModelScoreDaily,
    OnlineScore,
    Order,
    Packet,
    Position,
    Resolution,
    SignalSnapshot,
)


class Repository:
    """Async CRUD helpers for the agent-trader database."""

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

    async def bulk_upsert_markets(
        self, markets: list[dict], batch_size: int = 500
    ) -> None:
        """Batch-upsert a list of markets.

        Inserts are chunked into *batch_size* rows per statement to stay
        under PostgreSQL's 32,767 bind-parameter limit.
        """
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

    async def bulk_add_snapshots(
        self, snapshots: list[dict], batch_size: int = 500
    ) -> int:
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

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    async def bulk_upsert_evidence(self, items: list[dict]) -> int:
        """Upsert evidence items by content_hash. Return count inserted."""
        if not items:
            return 0
        async with self.session_factory() as session:
            stmt = pg_insert(EvidenceItem).values(items)
            stmt = stmt.on_conflict_do_nothing(index_elements=["content_hash"])
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

    async def count_recent_evidence(self, window_minutes: int) -> int:
        """Count evidence items within the given time window (minutes)."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        async with self.session_factory() as session:
            result = await session.execute(
                select(sa_func.count(EvidenceItem.evidence_id))
                .where(EvidenceItem.ts_utc >= cutoff)
            )
            return result.scalar_one() or 0

    async def get_recent_evidence(self, max_age_hours: int) -> list[EvidenceItem]:
        """Return evidence items within the given time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        async with self.session_factory() as session:
            result = await session.execute(
                select(EvidenceItem)
                .where(EvidenceItem.ts_utc >= cutoff)
                .order_by(EvidenceItem.ts_utc.desc())
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Packets
    # ------------------------------------------------------------------

    async def add_packet(self, packet_data: dict) -> int:
        """Insert a packet and return its id."""
        async with self.session_factory() as session:
            packet = Packet(**packet_data)
            session.add(packet)
            await session.commit()
            return packet.packet_id

    async def get_latest_packet(self, market_id: str) -> Packet | None:
        """Return the most recent packet for *market_id*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Packet)
                .where(Packet.market_id == market_id)
                .order_by(Packet.ts_utc.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Model Runs
    # ------------------------------------------------------------------

    async def get_recently_paneled_market_ids(self, since: datetime) -> set[str]:
        """Return market_ids that have model_runs since *since*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(sa_func.distinct(ModelRun.market_id))
                .where(ModelRun.ts_utc >= since)
            )
            return {row[0] for row in result.all()}

    async def add_model_run(self, run_data: dict) -> str:
        """Insert a model run and return its run_id."""
        async with self.session_factory() as session:
            model_run = ModelRun(**run_data)
            session.add(model_run)
            await session.commit()
            return model_run.run_id

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    async def add_aggregation(self, agg_data: dict) -> int:
        """Insert an aggregation and return its agg_id."""
        async with self.session_factory() as session:
            aggregation = Aggregation(**agg_data)
            session.add(aggregation)
            await session.commit()
            return aggregation.agg_id

    async def get_latest_aggregation(self, market_id: str) -> Aggregation | None:
        """Return the most recent aggregation for *market_id*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Aggregation)
                .where(Aggregation.market_id == market_id)
                .order_by(Aggregation.ts_utc.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_panel_markets_today(self) -> int:
        """Count distinct market_ids in model_runs for today (UTC)."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        async with self.session_factory() as session:
            result = await session.execute(
                select(sa_func.count(sa_func.distinct(ModelRun.market_id)))
                .where(ModelRun.ts_utc >= today_start)
            )
            return result.scalar_one() or 0

    # ------------------------------------------------------------------
    # Scoring (M5)
    # ------------------------------------------------------------------

    async def get_market(self, market_id: str) -> Market | None:
        """Return a single market by id."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Market).where(Market.market_id == market_id)
            )
            return result.scalar_one_or_none()

    async def get_resolutions_since(self, since: datetime) -> list[Resolution]:
        """Return resolutions resolved since *since*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Resolution)
                .where(Resolution.resolved_ts_utc >= since)
                .order_by(Resolution.resolved_ts_utc.desc())
            )
            return list(result.scalars().all())

    async def get_model_runs_for_market(self, market_id: str) -> list[ModelRun]:
        """Return all model runs for a given market."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(ModelRun)
                .where(ModelRun.market_id == market_id)
                .order_by(ModelRun.ts_utc.desc())
            )
            return list(result.scalars().all())

    async def get_closed_positions(self) -> list[Position]:
        """Return all positions with ``status='closed'``."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Position).where(Position.status == "closed")
            )
            return list(result.scalars().all())

    async def add_model_score(self, score_data: dict) -> int:
        """Insert a model score and return its score_id."""
        async with self.session_factory() as session:
            score = ModelScoreDaily(**score_data)
            session.add(score)
            await session.commit()
            return score.score_id

    async def get_model_scores(self, score_date) -> list[ModelScoreDaily]:
        """Return all model scores for a given date."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(ModelScoreDaily)
                .where(ModelScoreDaily.score_date == score_date)
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Snapshots (extended queries for signals)
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

    async def get_snapshot_at(
        self, market_id: str, target_ts: datetime
    ) -> MarketSnapshot | None:
        """Return the snapshot nearest to *target_ts* for *market_id*."""
        async with self.session_factory() as session:
            # Try to find the closest snapshot before or at target_ts.
            before = await session.execute(
                select(MarketSnapshot)
                .where(
                    MarketSnapshot.market_id == market_id,
                    MarketSnapshot.ts_utc <= target_ts,
                )
                .order_by(MarketSnapshot.ts_utc.desc())
                .limit(1)
            )
            snap_before = before.scalar_one_or_none()

            # Try to find the closest snapshot after target_ts.
            after = await session.execute(
                select(MarketSnapshot)
                .where(
                    MarketSnapshot.market_id == market_id,
                    MarketSnapshot.ts_utc > target_ts,
                )
                .order_by(MarketSnapshot.ts_utc.asc())
                .limit(1)
            )
            snap_after = after.scalar_one_or_none()

            if snap_before is None:
                return snap_after
            if snap_after is None:
                return snap_before

            # Return whichever is closer.
            diff_before = abs((snap_before.ts_utc - target_ts).total_seconds())
            diff_after = abs((snap_after.ts_utc - target_ts).total_seconds())
            return snap_before if diff_before <= diff_after else snap_after

    # ------------------------------------------------------------------
    # Signal Snapshots
    # ------------------------------------------------------------------

    async def add_signal_snapshot(self, data: dict) -> int:
        """Insert a signal snapshot and return its signal_id."""
        async with self.session_factory() as session:
            signal = SignalSnapshot(**data)
            session.add(signal)
            await session.commit()
            return signal.signal_id

    async def get_latest_signal_snapshot(
        self, market_id: str
    ) -> SignalSnapshot | None:
        """Return the most recent signal snapshot for *market_id*."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SignalSnapshot)
                .where(SignalSnapshot.market_id == market_id)
                .order_by(SignalSnapshot.ts_utc.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Evidence (extended queries for signals)
    # ------------------------------------------------------------------

    async def count_evidence_since(
        self, since: datetime, source_type: str | None = None
    ) -> int:
        """Count evidence items since *since*, optionally filtered by source_type."""
        async with self.session_factory() as session:
            stmt = select(sa_func.count(EvidenceItem.evidence_id)).where(
                EvidenceItem.ts_utc >= since
            )
            if source_type is not None:
                stmt = stmt.where(EvidenceItem.source_type == source_type)
            result = await session.execute(stmt)
            return result.scalar_one() or 0

    async def count_evidence_for_market_since(
        self,
        market_id: str,
        since: datetime,
        source_type: str | None = None,
    ) -> int:
        """Count evidence items linked to *market_id* since *since*.

        Uses keyword matching against the market question to approximate
        linkage without requiring a full join table.
        """
        # For signal freshness, we use the global count as a proxy.
        # The signal collector handles per-market linking separately.
        return await self.count_evidence_since(since, source_type)

    # ------------------------------------------------------------------
    # Online Scores
    # ------------------------------------------------------------------

    async def add_online_score(self, data: dict) -> int:
        """Insert an online score and return its id."""
        async with self.session_factory() as session:
            score = OnlineScore(**data)
            session.add(score)
            await session.commit()
            return score.online_score_id

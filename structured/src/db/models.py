from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# 7 Core tables (from parent)
# ---------------------------------------------------------------------------


class Market(Base):
    __tablename__ = "markets"

    market_id: Mapped[str] = mapped_column(String, primary_key=True)
    question: Mapped[str] = mapped_column(String, nullable=False)
    rules_text: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution_time_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    snapshots: Mapped[list[MarketSnapshot]] = relationship(back_populates="market")
    decisions: Mapped[list[Decision]] = relationship(back_populates="market")
    orders: Mapped[list[Order]] = relationship(back_populates="market")
    position: Mapped[Position | None] = relationship(back_populates="market")
    resolution: Mapped[Resolution | None] = relationship(back_populates="market")
    category_assignments: Mapped[list[CategoryAssignment]] = relationship(back_populates="market")
    engine_prices: Mapped[list[EnginePrice]] = relationship(back_populates="market")
    market_resolutions: Mapped[list[MarketResolution]] = relationship(back_populates="market")


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"
    __table_args__ = (
        Index("ix_market_snapshots_market_ts", "market_id", "ts_utc", postgresql_using="btree"),
    )

    snapshot_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    best_bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)

    market: Mapped[Market] = relationship(back_populates="snapshots")


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (
        Index("ix_decisions_ts", "ts_utc", postgresql_using="btree"),
    )

    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    size_eur: Mapped[float] = mapped_column(Float, nullable=False)
    reason_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    policy_version: Mapped[str | None] = mapped_column(String, nullable=True)

    market: Mapped[Market] = relationship(back_populates="decisions")
    orders: Mapped[list[Order]] = relationship(back_populates="decision")


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("decisions.decision_id"), nullable=False
    )
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    side: Mapped[str] = mapped_column(String, nullable=False)
    size_eur: Mapped[float] = mapped_column(Float, nullable=False)
    limit_price_ref: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    created_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    decision: Mapped[Decision] = relationship(back_populates="orders")
    market: Mapped[Market] = relationship(back_populates="orders")
    fills: Mapped[list[Fill]] = relationship(back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    fill_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("orders.order_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    size_eur: Mapped[float] = mapped_column(Float, nullable=False)
    fee_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0)

    order: Mapped[Order] = relationship(back_populates="fills")


class Position(Base):
    __tablename__ = "positions"

    position_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), unique=True, nullable=False
    )
    side: Mapped[str] = mapped_column(String, nullable=False)
    size_eur: Mapped[float] = mapped_column(Float, nullable=False)
    avg_entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String, nullable=False, default="open")
    opened_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_update_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    market: Mapped[Market] = relationship(back_populates="position")


class Resolution(Base):
    __tablename__ = "resolutions"

    resolution_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), unique=True, nullable=False
    )
    resolved_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    outcome: Mapped[str] = mapped_column(String, nullable=False)

    market: Mapped[Market] = relationship(back_populates="resolution")


# ---------------------------------------------------------------------------
# 8 New tables (structured trader)
# ---------------------------------------------------------------------------


class CategoryAssignment(Base):
    __tablename__ = "category_assignments"
    __table_args__ = (
        Index("ix_category_assignments_category", "category"),
        Index("ix_category_assignments_status", "parse_status"),
    )

    assignment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    category: Mapped[str] = mapped_column(String, nullable=False)
    parser_name: Mapped[str] = mapped_column(String, nullable=False)
    parse_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    parse_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    contract_spec_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    market: Mapped[Market] = relationship(back_populates="category_assignments")


class SourceObservation(Base):
    __tablename__ = "source_observations"
    __table_args__ = (
        Index("ix_source_observations_category_source", "category", "source_name"),
        Index("ix_source_observations_ts", "ts_ingested"),
    )

    observation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_key: Mapped[str] = mapped_column(String, nullable=False)
    ts_source: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ts_ingested: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    normalized_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class EnginePrice(Base):
    __tablename__ = "engine_prices"
    __table_args__ = (
        Index("ix_engine_prices_market_ts", "market_id", "ts_utc"),
        Index("ix_engine_prices_category", "category"),
    )

    engine_price_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    category: Mapped[str] = mapped_column(String, nullable=False)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    engine_version: Mapped[str] = mapped_column(String, nullable=False)
    p_yes: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_before_costs: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_after_costs: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    market: Mapped[Market] = relationship(back_populates="engine_prices")


class CategoryPortfolioRow(Base):
    __tablename__ = "category_portfolios"

    category: Mapped[str] = mapped_column(String, primary_key=True)
    bankroll_eur: Mapped[float] = mapped_column(Float, nullable=False)
    exposure_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class CategoryPnLDaily(Base):
    __tablename__ = "category_pnl_daily"
    __table_args__ = (
        Index("ix_category_pnl_daily_date", "pnl_date"),
    )

    pnl_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    pnl_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    realized_pnl_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trades_opened: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trades_closed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class CalibrationStat(Base):
    __tablename__ = "calibration_stats"
    __table_args__ = (
        Index("ix_calibration_stats_category_date", "category", "stat_date"),
    )

    stat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    engine_version: Mapped[str] = mapped_column(String, nullable=False)
    stat_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    n_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_resolved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    backtest_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    engine_version: Mapped[str] = mapped_column(String, nullable=False)
    ts_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ts_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    results_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class MarketResolution(Base):
    __tablename__ = "market_resolutions"
    __table_args__ = (
        Index("ix_market_resolutions_market", "market_id"),
    )

    market_resolution_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    resolved_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    outcome: Mapped[str] = mapped_column(String, nullable=False)
    settlement_source: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    market: Mapped[Market] = relationship(back_populates="market_resolutions")

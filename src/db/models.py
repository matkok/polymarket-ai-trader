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
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


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
    packets: Mapped[list[Packet]] = relationship(back_populates="market")
    model_runs: Mapped[list[ModelRun]] = relationship(back_populates="market")
    aggregations: Mapped[list[Aggregation]] = relationship(back_populates="market")
    signal_snapshots: Mapped[list[SignalSnapshot]] = relationship(back_populates="market")


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


class EvidenceItem(Base):
    __tablename__ = "evidence_items"
    __table_args__ = (
        Index("ix_evidence_items_content_hash", "content_hash", unique=True),
        Index("ix_evidence_items_url", "url"),
    )

    evidence_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    published_ts_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    raw_content_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    extracted_text: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)


class Packet(Base):
    __tablename__ = "packets"

    packet_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    packet_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    packet_hash: Mapped[str] = mapped_column(String, nullable=False)
    packet_version: Mapped[str] = mapped_column(String, nullable=False)

    market: Mapped[Market] = relationship(back_populates="packets")


class ModelRun(Base):
    __tablename__ = "model_runs"
    __table_args__ = (
        Index("ix_model_runs_market_ts", "market_id", "ts_utc", postgresql_using="btree"),
    )

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    packet_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("packets.packet_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    tier: Mapped[str] = mapped_column(String, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    charter_version: Mapped[str] = mapped_column(String, nullable=False)
    policy_version: Mapped[str] = mapped_column(String, nullable=False)
    raw_response: Mapped[str] = mapped_column(String, nullable=False)
    parsed_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    parse_ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    budget_skip: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    estimated_cost_eur: Mapped[float] = mapped_column(Float, nullable=False)

    market: Mapped[Market] = relationship(back_populates="model_runs")


class Aggregation(Base):
    __tablename__ = "aggregations"
    __table_args__ = (
        Index("ix_aggregations_market_ts", "market_id", "ts_utc", postgresql_using="btree"),
    )

    agg_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    aggregation_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    policy_version: Mapped[str] = mapped_column(String, nullable=False)

    market: Mapped[Market] = relationship(back_populates="aggregations")


class SignalSnapshot(Base):
    __tablename__ = "signal_snapshots"
    __table_args__ = (
        Index("ix_signal_snapshots_market_ts", "market_id", "ts_utc", postgresql_using="btree"),
        Index("ix_signal_snapshots_ts", "ts_utc", postgresql_using="btree"),
    )

    signal_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    odds_move_1h: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_move_6h: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_move_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_ratio_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_widening: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_count_6h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_count_24h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    credible_evidence_6h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    google_trends_spike: Mapped[float | None] = mapped_column(Float, nullable=True)
    wikipedia_spike: Mapped[float | None] = mapped_column(Float, nullable=True)
    triage_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    triage_reasons: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    market: Mapped[Market] = relationship(back_populates="signal_snapshots")


class OnlineScore(Base):
    __tablename__ = "online_scores"
    __table_args__ = (
        Index("ix_online_scores_market_ts", "market_id", "ts_utc", postgresql_using="btree"),
        Index("ix_online_scores_position", "position_id"),
    )

    online_score_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.market_id"), nullable=False
    )
    position_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("positions.position_id"), nullable=False
    )
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hours_since_entry: Mapped[float] = mapped_column(Float, nullable=False)
    p_consensus_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    p_market_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    p_market_now: Mapped[float] = mapped_column(Float, nullable=False)
    edge_capture: Mapped[float] = mapped_column(Float, nullable=False)
    direction_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    unrealized_pnl_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class ModelScoreDaily(Base):
    __tablename__ = "model_scores_daily"
    __table_args__ = (
        UniqueConstraint("model_id", "score_date", name="uq_model_scores_model_date"),
        Index("ix_model_scores_date", "score_date"),
    )

    score_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    score_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    markets_scored: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pnl_attrib_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    veto_value_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)

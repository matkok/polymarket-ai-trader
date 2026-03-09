from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PacketEvidence(BaseModel):
    """A single evidence item within a packet."""

    url: str
    title: str
    published_ts: datetime | None = None
    excerpt: str
    source_type: str


class PacketMarketContext(BaseModel):
    """Market data context within a packet."""

    question: str
    rules_text: str | None = None
    current_mid: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    liquidity: float | None = None
    volume: float | None = None
    implied_probability: float | None = None


class PacketPositionSummary(BaseModel):
    """Current position summary within a packet (None if no position)."""

    side: str
    size_eur: float
    avg_entry_price: float
    unrealized_pnl: float


class Packet(BaseModel):
    """Complete packet sent to the LLM panel for a single market."""

    market_id: str
    ts_utc: datetime
    market_context: PacketMarketContext
    position_summary: PacketPositionSummary | None = None
    evidence_items: list[PacketEvidence] = []
    packet_version: str = "m2.0"

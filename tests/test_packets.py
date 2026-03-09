"""Tests for src.packets — packet building, hashing, and pruning."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.config.policy import Policy
from src.db.models import EvidenceItem, Market, MarketSnapshot, Position
from src.packets.builder import PacketBuilder
from src.packets.schemas import Packet, PacketEvidence, PacketMarketContext


# ---- Helpers ----------------------------------------------------------------


def _make_policy(**overrides) -> Policy:
    return Policy(**overrides)


def _make_market(
    market_id: str = "m1",
    question: str = "Will it rain?",
    rules_text: str | None = "Standard rules",
) -> MagicMock:
    m = MagicMock(spec=Market)
    m.market_id = market_id
    m.question = question
    m.rules_text = rules_text
    return m


def _make_snapshot(
    market_id: str = "m1",
    mid: float = 0.55,
    best_bid: float = 0.53,
    best_ask: float = 0.57,
    liquidity: float = 10_000.0,
    volume: float = 20_000.0,
) -> MagicMock:
    s = MagicMock(spec=MarketSnapshot)
    s.market_id = market_id
    s.mid = mid
    s.best_bid = best_bid
    s.best_ask = best_ask
    s.liquidity = liquidity
    s.volume = volume
    s.ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
    return s


def _make_evidence(
    evidence_id: int = 1,
    title: str = "Test article",
    url: str = "https://example.com/article",
    extracted_text: str = "Some extracted text content.",
    source_type: str = "rss",
    published_ts_utc: datetime | None = None,
) -> MagicMock:
    item = MagicMock(spec=EvidenceItem)
    item.evidence_id = evidence_id
    item.title = title
    item.url = url
    item.extracted_text = extracted_text
    item.source_type = source_type
    item.published_ts_utc = published_ts_utc or datetime(2025, 6, 1, tzinfo=timezone.utc)
    item.ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
    item.content_hash = f"hash_{evidence_id}"
    return item


def _make_position(
    market_id: str = "m1",
    side: str = "BUY_YES",
    size_eur: float = 100.0,
    avg_entry_price: float = 0.50,
    unrealized_pnl: float = 5.0,
    status: str = "open",
) -> MagicMock:
    pos = MagicMock(spec=Position)
    pos.market_id = market_id
    pos.side = side
    pos.size_eur = size_eur
    pos.avg_entry_price = avg_entry_price
    pos.unrealized_pnl = unrealized_pnl
    pos.status = status
    return pos


# ---- PacketBuilder.build ----------------------------------------------------


class TestPacketBuild:
    """Packet assembly from market, snapshot, evidence, and position."""

    def test_builds_complete_packet(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        market = _make_market()
        snapshot = _make_snapshot()
        evidence = [_make_evidence()]
        position = _make_position()

        packet = builder.build(market, snapshot, evidence, position)

        assert packet.market_id == "m1"
        assert packet.market_context.question == "Will it rain?"
        assert packet.market_context.current_mid == 0.55
        assert packet.market_context.best_bid == 0.53
        assert packet.market_context.best_ask == 0.57
        assert packet.market_context.rules_text == "Standard rules"
        assert packet.position_summary is not None
        assert packet.position_summary.side == "BUY_YES"
        assert packet.position_summary.size_eur == 100.0
        assert len(packet.evidence_items) == 1
        assert packet.packet_version == "m2.0"

    def test_no_position(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        packet = builder.build(
            _make_market(), _make_snapshot(), [_make_evidence()], None
        )

        assert packet.position_summary is None

    def test_closed_position_excluded(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        position = _make_position(status="closed")
        packet = builder.build(
            _make_market(), _make_snapshot(), [], position
        )

        assert packet.position_summary is None

    def test_no_evidence(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        packet = builder.build(
            _make_market(), _make_snapshot(), [], None
        )

        assert len(packet.evidence_items) == 0

    def test_evidence_excerpt_truncated(self) -> None:
        policy = _make_policy(evidence_excerpt_max_chars=20)
        builder = PacketBuilder(policy)

        evidence = _make_evidence(
            extracted_text="A" * 100,
        )
        packet = builder.build(
            _make_market(), _make_snapshot(), [evidence], None
        )

        assert len(packet.evidence_items[0].excerpt) == 20

    def test_market_context_fields(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        market = _make_market(question="Test question?", rules_text=None)
        snapshot = _make_snapshot(
            mid=0.60, best_bid=0.58, best_ask=0.62,
            liquidity=5000.0, volume=15000.0,
        )

        packet = builder.build(market, snapshot, [], None)

        ctx = packet.market_context
        assert ctx.question == "Test question?"
        assert ctx.rules_text is None
        assert ctx.current_mid == 0.60
        assert ctx.implied_probability == 0.60
        assert ctx.liquidity == 5000.0
        assert ctx.volume == 15000.0


# ---- PacketBuilder.compute_hash ---------------------------------------------


class TestComputeHash:
    """Deterministic packet hashing."""

    def test_deterministic(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        packet = builder.build(
            _make_market(), _make_snapshot(), [_make_evidence()], None
        )

        h1 = builder.compute_hash(packet)
        h2 = builder.compute_hash(packet)
        assert h1 == h2

    def test_different_when_packet_changes(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        packet1 = builder.build(
            _make_market(), _make_snapshot(), [_make_evidence()], None
        )
        packet2 = builder.build(
            _make_market(market_id="m2"),
            _make_snapshot(market_id="m2"),
            [_make_evidence()],
            None,
        )

        assert builder.compute_hash(packet1) != builder.compute_hash(packet2)

    def test_returns_hex_string(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        packet = builder.build(
            _make_market(), _make_snapshot(), [], None
        )

        h = builder.compute_hash(packet)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---- PacketBuilder pruning --------------------------------------------------


class TestPruning:
    """Evidence pruning respects max items and ordering."""

    def test_keeps_max_items(self) -> None:
        policy = _make_policy(max_evidence_items_per_packet=3)
        builder = PacketBuilder(policy)

        base_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
        evidence = [
            _make_evidence(
                evidence_id=i,
                url=f"https://example.com/{i}",
                published_ts_utc=base_time + timedelta(hours=i),
            )
            for i in range(10)
        ]

        packet = builder.build(
            _make_market(), _make_snapshot(), evidence, None
        )

        assert len(packet.evidence_items) == 3

    def test_newest_first(self) -> None:
        policy = _make_policy(max_evidence_items_per_packet=2)
        builder = PacketBuilder(policy)

        old = _make_evidence(
            evidence_id=1,
            url="https://example.com/old",
            title="Old article",
            published_ts_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        new = _make_evidence(
            evidence_id=2,
            url="https://example.com/new",
            title="New article",
            published_ts_utc=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )

        packet = builder.build(
            _make_market(), _make_snapshot(), [old, new], None
        )

        assert packet.evidence_items[0].title == "New article"

    def test_keeps_official_source(self) -> None:
        policy = _make_policy(max_evidence_items_per_packet=2)
        builder = PacketBuilder(policy)

        base_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
        rss_items = [
            _make_evidence(
                evidence_id=i,
                url=f"https://example.com/{i}",
                source_type="rss",
                published_ts_utc=base_time + timedelta(hours=i),
            )
            for i in range(5)
        ]
        official = _make_evidence(
            evidence_id=99,
            url="https://example.com/official",
            source_type="official",
            published_ts_utc=base_time - timedelta(days=1),  # oldest
        )

        all_evidence = rss_items + [official]
        packet = builder.build(
            _make_market(), _make_snapshot(), all_evidence, None
        )

        source_types = [e.source_type for e in packet.evidence_items]
        assert "official" in source_types

    def test_fewer_than_max_keeps_all(self) -> None:
        policy = _make_policy(max_evidence_items_per_packet=10)
        builder = PacketBuilder(policy)

        evidence = [
            _make_evidence(evidence_id=i, url=f"https://example.com/{i}")
            for i in range(3)
        ]

        packet = builder.build(
            _make_market(), _make_snapshot(), evidence, None
        )

        assert len(packet.evidence_items) == 3


# ---- Packet schema validation -----------------------------------------------


class TestPacketSchema:
    """Packet Pydantic model validation."""

    def test_required_fields_present(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        packet = builder.build(
            _make_market(), _make_snapshot(), [_make_evidence()], _make_position()
        )

        assert packet.market_id is not None
        assert packet.ts_utc is not None
        assert packet.market_context is not None
        assert packet.packet_version is not None

    def test_serializes_to_json(self) -> None:
        policy = _make_policy()
        builder = PacketBuilder(policy)

        packet = builder.build(
            _make_market(), _make_snapshot(), [_make_evidence()], None
        )

        json_str = packet.model_dump_json()
        assert "market_id" in json_str
        assert "market_context" in json_str
        assert "evidence_items" in json_str

    def test_packet_evidence_fields(self) -> None:
        ev = PacketEvidence(
            url="https://example.com",
            title="Test",
            excerpt="Some text",
            source_type="rss",
        )
        assert ev.url == "https://example.com"
        assert ev.published_ts is None

    def test_packet_market_context_defaults(self) -> None:
        ctx = PacketMarketContext(question="Will it rain?")
        assert ctx.question == "Will it rain?"
        assert ctx.rules_text is None
        assert ctx.current_mid is None

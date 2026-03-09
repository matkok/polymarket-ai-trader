from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from src.config.policy import Policy
from src.db.models import EvidenceItem, Market, MarketSnapshot, Position
from src.packets.schemas import (
    Packet,
    PacketEvidence,
    PacketMarketContext,
    PacketPositionSummary,
)


class PacketBuilder:
    """Assembles packets from market data, evidence, and position state."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def build(
        self,
        market: Market,
        snapshot: MarketSnapshot,
        evidence: list[EvidenceItem],
        position: Position | None,
    ) -> Packet:
        """Assemble a complete packet for a single market."""
        now = datetime.now(timezone.utc)

        market_context = PacketMarketContext(
            question=market.question,
            rules_text=market.rules_text,
            current_mid=snapshot.mid,
            best_bid=snapshot.best_bid,
            best_ask=snapshot.best_ask,
            liquidity=snapshot.liquidity,
            volume=snapshot.volume,
            implied_probability=snapshot.mid,
        )

        position_summary: PacketPositionSummary | None = None
        if position is not None and position.status == "open":
            position_summary = PacketPositionSummary(
                side=position.side,
                size_eur=position.size_eur,
                avg_entry_price=position.avg_entry_price,
                unrealized_pnl=position.unrealized_pnl,
            )

        pruned_evidence = self._prune_evidence(evidence)
        evidence_items = [
            PacketEvidence(
                url=item.url,
                title=item.title,
                published_ts=item.published_ts_utc,
                excerpt=item.extracted_text[: self.policy.evidence_excerpt_max_chars],
                source_type=item.source_type,
            )
            for item in pruned_evidence
        ]

        return Packet(
            market_id=market.market_id,
            ts_utc=now,
            market_context=market_context,
            position_summary=position_summary,
            evidence_items=evidence_items,
            packet_version="m2.0",
        )

    def compute_hash(self, packet: Packet) -> str:
        """SHA-256 of canonical JSON (sorted keys, no whitespace)."""
        data = json.loads(packet.model_dump_json())
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _prune_evidence(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        """Sort by newest first, keep at most max items, preserve official sources."""
        max_items = self.policy.max_evidence_items_per_packet

        # Sort by published_ts_utc descending (newest first), None last.
        sorted_ev = sorted(
            evidence,
            key=lambda e: e.published_ts_utc or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        if len(sorted_ev) <= max_items:
            return sorted_ev

        # Keep the top N items but ensure at least one official source is kept.
        top = sorted_ev[:max_items]
        official_in_top = any(e.source_type == "official" for e in top)

        if not official_in_top:
            # Find the first official source in the remainder.
            for item in sorted_ev[max_items:]:
                if item.source_type == "official":
                    top[-1] = item
                    break

        return top

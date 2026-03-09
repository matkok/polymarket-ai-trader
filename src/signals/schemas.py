"""Signal dataclasses for market microstructure, evidence freshness, and external sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MicrostructureSignal:
    """Market microstructure signals computed from snapshot history."""

    odds_move_1h: float | None = None
    odds_move_6h: float | None = None
    odds_move_24h: float | None = None
    volume_ratio_24h: float | None = None
    spread_current: float | None = None
    spread_widening: float | None = None


@dataclass
class EvidenceFreshnessSignal:
    """Evidence freshness signals from the evidence database."""

    evidence_count_6h: int = 0
    evidence_count_24h: int = 0
    credible_evidence_6h: int = 0


@dataclass
class GoogleTrendsSignal:
    """Google Trends spike signal for an entity."""

    entity: str = ""
    current_interest: float = 0.0
    trailing_baseline: float = 0.0
    spike_score: float = 0.0


@dataclass
class WikipediaSignal:
    """Wikipedia pageview spike signal for an article."""

    article: str = ""
    current_views: float = 0.0
    trailing_baseline: float = 0.0
    spike_score: float = 0.0


@dataclass
class MarketSignalBundle:
    """All signals for one market in one cycle."""

    market_id: str = ""
    ts_utc: datetime | None = None
    microstructure: MicrostructureSignal | None = None
    evidence_freshness: EvidenceFreshnessSignal | None = None
    google_trends: GoogleTrendsSignal | None = None
    wikipedia: WikipediaSignal | None = None


@dataclass
class TriageResult:
    """Result of triage scoring for a market."""

    triage_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    should_panel: bool = False
    guardrail_flags: list[str] = field(default_factory=list)

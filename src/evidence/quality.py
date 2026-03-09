"""Evidence quality scoring — recency, credibility, and relevance."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import structlog

from src.db.models import EvidenceItem

logger = structlog.get_logger(__name__)

# Source credibility ratings (0-1 scale).
SOURCE_CREDIBILITY: dict[str, float] = {
    # Tier 1: Major wire services and broadcasters
    "BBC News": 0.95,
    "NPR News": 0.90,
    "Al Jazeera": 0.85,
    # Tier 2: Specialized outlets
    "Politico": 0.85,
    "The Hill": 0.80,
    "ESPN": 0.90,
    "Ars Technica": 0.85,
    "TechCrunch": 0.80,
    # Tier 3: Crypto outlets
    "CoinDesk": 0.75,
    "The Block": 0.75,
    "CoinTelegraph": 0.70,
    # Search results
    "xai_search": 0.60,
    "xai_social": 0.50,
}

DEFAULT_CREDIBILITY = 0.50


def recency_score(published_ts: datetime | None, now: datetime | None = None) -> float:
    """Score 0-1 based on how recent the evidence is.

    Uses exponential decay: score = exp(-hours/48).
    24h old = 0.61, 48h old = 0.37, 72h old = 0.22.
    """
    if published_ts is None:
        return 0.3  # Unknown age gets moderate score

    now = now or datetime.now(timezone.utc)
    if published_ts.tzinfo is None:
        published_ts = published_ts.replace(tzinfo=timezone.utc)

    age_hours = max((now - published_ts).total_seconds() / 3600, 0)
    return math.exp(-age_hours / 48)


def credibility_score(source_name: str) -> float:
    """Return credibility score for a source."""
    return SOURCE_CREDIBILITY.get(source_name, DEFAULT_CREDIBILITY)


def compute_quality_score(
    item: EvidenceItem,
    relevance_score: float = 1.0,
    now: datetime | None = None,
) -> float:
    """Compute composite quality score (0-1).

    quality = 0.4 * recency + 0.3 * credibility + 0.3 * relevance
    """
    recency = recency_score(item.published_ts_utc, now)

    # Determine credibility: xai sources use source_type directly.
    if item.source_type in ("xai_search", "xai_social"):
        cred = SOURCE_CREDIBILITY.get(item.source_type, DEFAULT_CREDIBILITY)
    else:
        cred = DEFAULT_CREDIBILITY

    return 0.4 * recency + 0.3 * cred + 0.3 * relevance_score


def rank_evidence(
    items: list[EvidenceItem],
    relevance_scores: dict[int, float] | None = None,
    now: datetime | None = None,
) -> list[tuple[float, EvidenceItem]]:
    """Rank evidence items by composite quality score.

    Returns list of (score, item) tuples sorted descending.
    """
    relevance_scores = relevance_scores or {}
    scored = []
    for item in items:
        rel = relevance_scores.get(item.evidence_id, 1.0)
        quality = compute_quality_score(item, rel, now)
        scored.append((quality, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

"""Orchestrate signal collection from all sources for a set of markets."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from src.config.policy import Policy
from src.db.repository import Repository
from src.signals.microstructure import MicrostructureComputer
from src.signals.schemas import (
    EvidenceFreshnessSignal,
    MarketSignalBundle,
)
from src.signals.trends import GoogleTrendsClient
from src.signals.wikipedia import WikipediaPageviewClient

logger = structlog.get_logger(__name__)


class SignalCollector:
    """Orchestrate all signal sources for a set of markets.

    Each source is independent: one failure does not block others.
    """

    def __init__(
        self,
        repo: Repository,
        policy: Policy,
        trends_client: GoogleTrendsClient | None = None,
        wiki_client: WikipediaPageviewClient | None = None,
    ) -> None:
        self.repo = repo
        self.policy = policy
        self.micro = MicrostructureComputer(repo)
        self.trends_client = trends_client
        self.wiki_client = wiki_client

    async def collect_all(
        self,
        markets: list,
        snapshots: dict,
    ) -> dict[str, MarketSignalBundle]:
        """Collect signals for each market.

        Args:
            markets: List of Market objects (or mocks with .market_id, .question).
            snapshots: Dict of market_id -> MarketSnapshot.

        Returns:
            Dict of market_id -> MarketSignalBundle.
        """
        now = datetime.now(timezone.utc)
        bundles: dict[str, MarketSignalBundle] = {}

        # Sort markets by snapshot volume (descending) so high-volume ones
        # are prioritised when we cap.
        sorted_markets = sorted(
            markets,
            key=lambda m: (
                snapshots.get(m.market_id).volume
                if snapshots.get(m.market_id) and snapshots.get(m.market_id).volume is not None
                else 0.0
            ),
            reverse=True,
        )

        # Cap the number of markets we compute signals for.
        capped = sorted_markets[: self.policy.triage_max_markets_for_signals]

        for market in capped:
            snap = snapshots.get(market.market_id)
            bundle = MarketSignalBundle(
                market_id=market.market_id,
                ts_utc=now,
            )

            # 1. Microstructure (always).
            try:
                bundle.microstructure = await self.micro.compute(
                    market.market_id, snap
                )
            except Exception:
                logger.warning(
                    "signal_microstructure_error", market_id=market.market_id
                )

            # 2. Evidence freshness (always).
            try:
                bundle.evidence_freshness = await self._compute_evidence_freshness()
            except Exception:
                logger.warning(
                    "signal_evidence_freshness_error", market_id=market.market_id
                )

            # 3. Google Trends (if enabled).
            if self.trends_client is not None:
                try:
                    entity = GoogleTrendsClient.extract_entity(market.question)
                    if entity:
                        bundle.google_trends = await self.trends_client.get_spike_score(
                            entity
                        )
                except Exception:
                    logger.warning(
                        "signal_trends_error", market_id=market.market_id
                    )

            # 4. Wikipedia (if enabled).
            if self.wiki_client is not None:
                try:
                    article = WikipediaPageviewClient.extract_article(market.question)
                    if article:
                        bundle.wikipedia = await self.wiki_client.get_spike_score(
                            article
                        )
                except Exception:
                    logger.warning(
                        "signal_wikipedia_error", market_id=market.market_id
                    )

            bundles[market.market_id] = bundle

        logger.info(
            "signal_collection_done",
            markets_processed=len(bundles),
            total_markets=len(markets),
        )
        return bundles

    async def _compute_evidence_freshness(self) -> EvidenceFreshnessSignal:
        """Count recent evidence from the database."""
        now = datetime.now(timezone.utc)
        since_6h = now - timedelta(hours=6)
        since_24h = now - timedelta(hours=24)

        count_6h = await self.repo.count_evidence_since(since_6h)
        count_24h = await self.repo.count_evidence_since(since_24h)
        credible_6h = await self.repo.count_evidence_since(since_6h, source_type="rss")

        return EvidenceFreshnessSignal(
            evidence_count_6h=count_6h,
            evidence_count_24h=count_24h,
            credible_evidence_6h=credible_6h,
        )

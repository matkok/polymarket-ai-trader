"""Google Trends spike detection client."""

from __future__ import annotations

import asyncio
from functools import partial

import structlog

from src.signals.schemas import GoogleTrendsSignal

logger = structlog.get_logger(__name__)


class GoogleTrendsClient:
    """Detect Google Trends spikes for market-related entities.

    Uses the pytrends library (synchronous) run in an executor to avoid
    blocking the event loop.
    """

    def __init__(
        self,
        trailing_days: int = 30,
        spike_threshold: float = 2.0,
    ) -> None:
        self.trailing_days = trailing_days
        self.spike_threshold = spike_threshold

    async def get_spike_score(self, entity: str) -> GoogleTrendsSignal | None:
        """Fetch Google Trends data and compute a spike score.

        Args:
            entity: The search term to query.

        Returns:
            GoogleTrendsSignal or None on failure.
        """
        if not entity:
            return None

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, partial(self._fetch_trends, entity)
            )
            return result
        except Exception:
            logger.warning("google_trends_error", entity=entity, exc_info=True)
            return None

    def _fetch_trends(self, entity: str) -> GoogleTrendsSignal | None:
        """Synchronous pytrends fetch — runs in executor."""
        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl="en-US", tz=0)
            pytrends.build_payload(
                [entity],
                timeframe=f"today {self.trailing_days}-d",
            )
            df = pytrends.interest_over_time()

            if df.empty or entity not in df.columns:
                return None

            values = df[entity].tolist()
            if not values:
                return None

            current_interest = float(values[-1])
            trailing_baseline = sum(values[:-1]) / max(len(values) - 1, 1)

            if trailing_baseline > 0:
                spike_score = current_interest / trailing_baseline
            else:
                spike_score = 0.0

            return GoogleTrendsSignal(
                entity=entity,
                current_interest=current_interest,
                trailing_baseline=trailing_baseline,
                spike_score=spike_score,
            )
        except Exception:
            logger.warning("google_trends_fetch_error", entity=entity, exc_info=True)
            return None

    @staticmethod
    def extract_entity(question: str) -> str:
        """Extract the best entity from a market question.

        Takes the longest keyword as a proxy for the proper noun.
        """
        stop_words = frozenset({
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "it", "be", "as", "was", "were",
            "are", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall", "can",
            "this", "that", "these", "those", "what", "which", "who", "whom",
            "how", "when", "where", "why", "if", "not", "no", "yes", "so",
            "up", "out", "about", "into", "over", "after", "before", "between",
            "under", "again", "then", "once", "here", "there", "all", "any",
            "each", "every", "both", "few", "more", "most", "other", "some",
            "such", "than", "too", "very",
        })
        words = question.split()
        cleaned = []
        for w in words:
            word = "".join(c for c in w if c.isalnum() or c == " ")
            if word and len(word) > 2 and word.lower() not in stop_words:
                cleaned.append(word)
        if not cleaned:
            return ""
        return max(cleaned, key=len)

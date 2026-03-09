"""Wikipedia pageview spike detection client."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import structlog

from src.signals.schemas import WikipediaSignal

logger = structlog.get_logger(__name__)

WIKIMEDIA_API_BASE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/user"
)


class WikipediaPageviewClient:
    """Detect Wikipedia pageview spikes for market-related articles.

    Uses the free Wikimedia REST API (no authentication required).
    """

    def __init__(
        self,
        trailing_days: int = 30,
        spike_threshold: float = 2.0,
    ) -> None:
        self.trailing_days = trailing_days
        self.spike_threshold = spike_threshold

    async def get_spike_score(self, article: str) -> WikipediaSignal | None:
        """Fetch pageview data and compute a spike score.

        Args:
            article: The Wikipedia article name (e.g. "Donald_Trump").

        Returns:
            WikipediaSignal or None on failure.
        """
        if not article:
            return None

        try:
            now = datetime.now(timezone.utc)
            end = now - timedelta(days=1)  # yesterday (today's data may be incomplete)
            start = end - timedelta(days=self.trailing_days)

            start_str = start.strftime("%Y%m%d")
            end_str = end.strftime("%Y%m%d")

            formatted_article = article.replace(" ", "_")
            url = f"{WIKIMEDIA_API_BASE}/{formatted_article}/daily/{start_str}/{end_str}"

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "agent-trader/1.0 (research project)"},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()

            items = data.get("items", [])
            if not items:
                return None

            views = [item.get("views", 0) for item in items]
            if not views:
                return None

            current_views = float(views[-1])
            trailing_baseline = sum(views[:-1]) / max(len(views) - 1, 1)

            if trailing_baseline > 0:
                spike_score = current_views / trailing_baseline
            else:
                spike_score = 0.0

            return WikipediaSignal(
                article=formatted_article,
                current_views=current_views,
                trailing_baseline=trailing_baseline,
                spike_score=spike_score,
            )
        except Exception:
            logger.warning("wikipedia_pageview_error", article=article, exc_info=True)
            return None

    @staticmethod
    def extract_article(question: str) -> str:
        """Extract the best article name from a market question.

        Takes the longest keyword, capitalised, as a proxy for the article name.
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
        longest = max(cleaned, key=len)
        return longest.title()

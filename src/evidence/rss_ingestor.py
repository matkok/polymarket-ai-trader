from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx
import structlog
from bs4 import BeautifulSoup

from src.config.settings import Settings
from src.evidence.schemas import DEFAULT_RSS_FEEDS, FetchedArticle, RSSFeedConfig

logger = structlog.get_logger(__name__)


class RSSIngestor:
    """Fetches and parses RSS feeds into deduplicated articles."""

    def __init__(
        self,
        settings: Settings,
        feeds: list[RSSFeedConfig] | None = None,
    ) -> None:
        self.settings = settings
        self.feeds = feeds or DEFAULT_RSS_FEEDS

    async def fetch_all(self) -> list[FetchedArticle]:
        """Fetch all configured feeds concurrently and return deduplicated articles."""
        async with httpx.AsyncClient(
            timeout=self.settings.rss_fetch_timeout,
            follow_redirects=True,
        ) as client:
            tasks = [self._fetch_feed(client, feed) for feed in self.feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[FetchedArticle] = []
        for feed_cfg, result in zip(self.feeds, results):
            if isinstance(result, Exception):
                logger.warning(
                    "rss_feed_fetch_failed",
                    feed=feed_cfg.name,
                    error=str(result),
                )
                continue
            articles.extend(result)

        # Deduplicate by content_hash.
        seen: set[str] = set()
        unique: list[FetchedArticle] = []
        for article in articles:
            h = self._compute_hash(article.url, article.extracted_text)
            if h not in seen:
                seen.add(h)
                unique.append(article)

        logger.info(
            "rss_fetch_complete",
            total_articles=len(articles),
            unique_articles=len(unique),
        )
        return unique

    async def _fetch_feed(
        self, client: httpx.AsyncClient, feed_config: RSSFeedConfig
    ) -> list[FetchedArticle]:
        """Fetch a single feed and parse its entries."""
        response = await client.get(feed_config.url)
        response.raise_for_status()
        return self._parse_feed(response.content, feed_config)

    def _parse_feed(
        self, raw_bytes: bytes, feed_config: RSSFeedConfig
    ) -> list[FetchedArticle]:
        """Parse RSS XML bytes into a list of FetchedArticle."""
        parsed = feedparser.parse(raw_bytes)
        articles: list[FetchedArticle] = []

        for entry in parsed.entries:
            url = getattr(entry, "link", "") or ""
            title = getattr(entry, "title", "") or ""
            if not url:
                continue

            # Extract published timestamp.
            published_ts: datetime | None = None
            published_str = getattr(entry, "published", None)
            if published_str:
                try:
                    published_ts = parsedate_to_datetime(published_str)
                    if published_ts.tzinfo is None:
                        published_ts = published_ts.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            # Extract text from summary/content.
            raw_html = ""
            if hasattr(entry, "content") and entry.content:
                raw_html = entry.content[0].get("value", "")
            elif hasattr(entry, "summary"):
                raw_html = entry.summary or ""

            extracted = self._extract_text(raw_html)

            articles.append(
                FetchedArticle(
                    url=url,
                    title=title,
                    published_ts=published_ts,
                    raw_html=raw_html,
                    extracted_text=extracted,
                    source_name=feed_config.name,
                )
            )

        return articles

    def _extract_text(self, html: str) -> str:
        """Strip HTML tags and normalize whitespace."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")
        # Normalize whitespace.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _compute_hash(self, url: str, text: str) -> str:
        """SHA-256 of normalized URL + text for dedup."""
        normalized = f"{url.strip().lower()}|{text.strip().lower()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

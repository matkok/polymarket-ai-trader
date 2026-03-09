"""Full-text article fetcher — extracts main content from web pages."""

from __future__ import annotations

import asyncio
import re

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger(__name__)

# Max content length to avoid overwhelming LLM context.
MAX_CONTENT_CHARS = 3000


class FullTextFetcher:
    """Fetches and extracts main article content from URLs."""

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    async def fetch(self, url: str) -> str | None:
        """Fetch a URL and extract the main text content.

        Returns extracted text or None on failure.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AgentTrader/1.0)"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                return None

            return self._extract_article(response.text)

        except Exception:
            logger.debug("fulltext_fetch_failed", url=url[:100])
            return None

    def _extract_article(self, html: str) -> str:
        """Extract main article text from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements.
        for tag in soup.find_all(["script", "style", "nav", "header", "footer",
                                   "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # Try to find article content in order of specificity.
        article = (
            soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find("main")
            or soup.find(class_=re.compile(r"article|post|content|story", re.I))
        )

        target = article if article else soup.body if soup.body else soup
        text = target.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        return text[:MAX_CONTENT_CHARS]

    async def enrich_articles(
        self,
        articles: list[dict],
        max_concurrent: int = 5,
    ) -> list[dict]:
        """Enrich article dicts with full text, up to max_concurrent at a time.

        Each dict should have 'url' and 'extracted_text' keys.
        Only fetches when extracted_text is short (<200 chars).
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _enrich_one(article: dict) -> None:
            if len(article.get("extracted_text", "")) >= 200:
                return  # Already has sufficient text
            async with semaphore:
                full_text = await self.fetch(article["url"])
                if full_text and len(full_text) > len(article.get("extracted_text", "")):
                    article["extracted_text"] = full_text

        tasks = [_enrich_one(a) for a in articles]
        await asyncio.gather(*tasks, return_exceptions=True)
        return articles

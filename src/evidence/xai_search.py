"""xAI web search client for evidence augmentation."""

from __future__ import annotations

import json

import openai
import structlog

from src.evidence.schemas import FetchedArticle

logger = structlog.get_logger(__name__)


class XAISearchClient:
    """Uses Grok chat completions with web search to find evidence."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        daily_cap: int = 30,
    ) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.daily_cap = daily_cap
        self.daily_calls_used = 0

    def can_search(self) -> bool:
        """Check if daily search quota is available."""
        return self.daily_calls_used < self.daily_cap

    def reset_daily(self) -> None:
        """Reset the daily call counter."""
        self.daily_calls_used = 0

    async def search(
        self, query: str, max_results: int = 5
    ) -> list[FetchedArticle]:
        """Search for evidence using Grok with web search capability."""
        if not self.can_search():
            logger.warning("xai_search_daily_cap_reached", cap=self.daily_cap)
            return []

        self.daily_calls_used += 1

        try:
            response = await self.client.chat.completions.create(
                model="grok-3",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. Search the web for recent "
                            "evidence related to the query. Return a JSON array of "
                            "objects with fields: url, title, excerpt. "
                            f"Return at most {max_results} results. "
                            "Output ONLY valid JSON, no commentary."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "[]"
            data = json.loads(content)

            # Handle both {"results": [...]} and direct array formats.
            items = data if isinstance(data, list) else data.get("results", [])

            articles: list[FetchedArticle] = []
            for item in items[:max_results]:
                articles.append(
                    FetchedArticle(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        extracted_text=item.get("excerpt", ""),
                        source_name="xai_search",
                    )
                )
            return articles

        except Exception:
            logger.exception("xai_search_error", query=query)
            return []

    async def search_social(
        self, query: str, max_results: int = 5
    ) -> list[FetchedArticle]:
        """Search for tweets, social posts, and online discussions via Grok."""
        if not self.can_search():
            logger.warning("xai_social_daily_cap_reached", cap=self.daily_cap)
            return []

        self.daily_calls_used += 1

        try:
            response = await self.client.chat.completions.create(
                model="grok-3-fast",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a social media research assistant. Search "
                            "Twitter/X, social media posts, and online discussions "
                            "for recent commentary related to the query. Return a "
                            "JSON array of objects with fields: url, title, excerpt. "
                            f"Return at most {max_results} results. "
                            "Output ONLY valid JSON, no commentary."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "[]"
            data = json.loads(content)

            # Handle both {"results": [...]} and direct array formats.
            items = data if isinstance(data, list) else data.get("results", [])

            articles: list[FetchedArticle] = []
            for item in items[:max_results]:
                articles.append(
                    FetchedArticle(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        extracted_text=item.get("excerpt", ""),
                        source_name="xai_social",
                    )
                )
            return articles

        except Exception:
            logger.exception("xai_social_search_error", query=query)
            return []

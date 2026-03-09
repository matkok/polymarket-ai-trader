"""Tests for src.evidence.xai_search — XAISearchClient."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evidence.xai_search import XAISearchClient


# ---- Helpers ----------------------------------------------------------------


def _make_mock_response(results: list[dict]) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({"results": results})
    return response


# ---- XAISearchClient -------------------------------------------------------


class TestXAISearchClient:
    """xAI web search evidence client."""

    async def test_search_returns_articles(self) -> None:
        client = XAISearchClient(api_key="test-key", daily_cap=10)
        mock_response = _make_mock_response(
            [
                {"url": "https://example.com/1", "title": "Article 1", "excerpt": "text1"},
                {"url": "https://example.com/2", "title": "Article 2", "excerpt": "text2"},
            ]
        )
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        articles = await client.search("test query")
        assert len(articles) == 2
        assert articles[0].url == "https://example.com/1"
        assert articles[0].title == "Article 1"
        assert articles[0].source_name == "xai_search"

    async def test_search_respects_max_results(self) -> None:
        client = XAISearchClient(api_key="test-key")
        mock_response = _make_mock_response(
            [
                {"url": f"https://example.com/{i}", "title": f"Article {i}", "excerpt": f"text{i}"}
                for i in range(10)
            ]
        )
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        articles = await client.search("query", max_results=3)
        assert len(articles) == 3

    async def test_daily_cap_enforcement(self) -> None:
        client = XAISearchClient(api_key="test-key", daily_cap=2)
        mock_response = _make_mock_response([{"url": "u", "title": "t", "excerpt": "e"}])
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        assert client.can_search() is True
        await client.search("q1")
        assert client.can_search() is True
        await client.search("q2")
        assert client.can_search() is False

        articles = await client.search("q3")
        assert articles == []

    async def test_reset_daily_clears_counter(self) -> None:
        client = XAISearchClient(api_key="test-key", daily_cap=1)
        client.daily_calls_used = 1
        assert client.can_search() is False

        client.reset_daily()
        assert client.can_search() is True
        assert client.daily_calls_used == 0

    async def test_search_handles_api_error(self) -> None:
        client = XAISearchClient(api_key="test-key")
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        articles = await client.search("query")
        assert articles == []
        assert client.daily_calls_used == 1

    async def test_search_handles_direct_array_response(self) -> None:
        client = XAISearchClient(api_key="test-key")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps(
            [{"url": "https://example.com/1", "title": "Article 1", "excerpt": "text1"}]
        )
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(return_value=response)

        articles = await client.search("query")
        assert len(articles) == 1

    async def test_default_daily_cap_is_30(self) -> None:
        client = XAISearchClient(api_key="test-key")
        assert client.daily_cap == 30


# ---- XAISearchClient social search -----------------------------------------


class TestXAISearchSocial:
    """xAI social media search via Grok-3-fast."""

    async def test_search_social_returns_articles(self) -> None:
        """Social search returns FetchedArticle with source_name='xai_social'."""
        client = XAISearchClient(api_key="test-key", daily_cap=10)
        mock_response = _make_mock_response(
            [
                {"url": "https://x.com/post/1", "title": "Tweet 1", "excerpt": "text1"},
                {"url": "https://x.com/post/2", "title": "Tweet 2", "excerpt": "text2"},
            ]
        )
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        articles = await client.search_social("test query")
        assert len(articles) == 2
        assert articles[0].url == "https://x.com/post/1"
        assert articles[0].source_name == "xai_social"
        assert articles[1].source_name == "xai_social"

    async def test_search_social_shares_daily_cap(self) -> None:
        """Social search shares the daily_calls_used counter with regular search."""
        client = XAISearchClient(api_key="test-key", daily_cap=2)
        mock_response = _make_mock_response([{"url": "u", "title": "t", "excerpt": "e"}])
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Use one call via regular search.
        await client.search("q1")
        assert client.daily_calls_used == 1

        # Use second call via social search.
        await client.search_social("q2")
        assert client.daily_calls_used == 2
        assert client.can_search() is False

        # Third call blocked.
        articles = await client.search_social("q3")
        assert articles == []

    async def test_search_social_uses_fast_model(self) -> None:
        """Social search uses grok-3-fast model."""
        client = XAISearchClient(api_key="test-key", daily_cap=10)
        mock_response = _make_mock_response(
            [{"url": "https://x.com/post/1", "title": "Tweet", "excerpt": "text"}]
        )
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.search_social("test query")

        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "grok-3-fast"

    async def test_search_social_handles_api_error(self) -> None:
        """Social search returns empty list on API error."""
        client = XAISearchClient(api_key="test-key")
        client.client = AsyncMock()
        client.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        articles = await client.search_social("query")
        assert articles == []
        assert client.daily_calls_used == 1

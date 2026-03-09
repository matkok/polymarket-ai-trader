"""Tests for src.evidence — RSS ingestion and evidence linking."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.db.models import EvidenceItem, Market
from src.evidence.embedder import EvidenceEmbedder
from src.evidence.linker import EvidenceLinker
from src.evidence.rss_ingestor import RSSIngestor
from src.evidence.schemas import FetchedArticle, RSSFeedConfig


# ---- Helpers ----------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    defaults = {"rss_fetch_timeout": 10, "evidence_max_age_hours": 72}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _make_feed_config(
    name: str = "Test Feed",
    url: str = "https://example.com/feed.xml",
    category: str = "general",
) -> RSSFeedConfig:
    return RSSFeedConfig(name=name, url=url, category=category)


def _make_evidence_item(
    evidence_id: int = 1,
    title: str = "Bitcoin hits new high",
    url: str = "https://example.com/article",
    extracted_text: str = "Bitcoin surged to a new all-time high today.",
    source_type: str = "rss",
    published_ts_utc: datetime | None = None,
) -> MagicMock:
    item = MagicMock(spec=EvidenceItem)
    item.evidence_id = evidence_id
    item.title = title
    item.url = url
    item.extracted_text = extracted_text
    item.source_type = source_type
    item.published_ts_utc = published_ts_utc or datetime(2025, 6, 1, tzinfo=timezone.utc)
    item.ts_utc = datetime(2025, 6, 1, tzinfo=timezone.utc)
    item.content_hash = "abc123"
    return item


def _make_market(
    market_id: str = "m1",
    question: str = "Will Bitcoin exceed $100,000 by end of 2025?",
) -> MagicMock:
    m = MagicMock(spec=Market)
    m.market_id = market_id
    m.question = question
    return m


# ---- RSSIngestor._parse_feed ------------------------------------------------


class TestParseFeed:
    """Parsing raw RSS feed bytes into FetchedArticle list."""

    def test_parses_valid_rss(self) -> None:
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Test Article</title>
              <link>https://example.com/article1</link>
              <description>&lt;p&gt;Some HTML content&lt;/p&gt;</description>
              <pubDate>Mon, 01 Jan 2025 12:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>"""

        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        feed_config = _make_feed_config()
        articles = ingestor._parse_feed(rss_xml, feed_config)

        assert len(articles) == 1
        assert articles[0].url == "https://example.com/article1"
        assert articles[0].title == "Test Article"
        assert articles[0].source_name == "Test Feed"
        assert articles[0].published_ts is not None

    def test_skips_entries_without_link(self) -> None:
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>No Link Article</title>
              <description>Some text</description>
            </item>
          </channel>
        </rss>"""

        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        articles = ingestor._parse_feed(rss_xml, _make_feed_config())

        assert len(articles) == 0

    def test_handles_missing_pubdate(self) -> None:
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>No Date</title>
              <link>https://example.com/nodate</link>
            </item>
          </channel>
        </rss>"""

        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        articles = ingestor._parse_feed(rss_xml, _make_feed_config())

        assert len(articles) == 1
        assert articles[0].published_ts is None

    def test_multiple_entries(self) -> None:
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Article 1</title>
              <link>https://example.com/1</link>
            </item>
            <item>
              <title>Article 2</title>
              <link>https://example.com/2</link>
            </item>
          </channel>
        </rss>"""

        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        articles = ingestor._parse_feed(rss_xml, _make_feed_config())

        assert len(articles) == 2


# ---- RSSIngestor._extract_text ----------------------------------------------


class TestExtractText:
    """HTML stripping and whitespace normalization."""

    def test_strips_html_tags(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        result = ingestor._extract_text("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_normalizes_whitespace(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        result = ingestor._extract_text("<p>  lots   of   spaces  </p>")
        assert result == "lots of spaces"

    def test_empty_html(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        result = ingestor._extract_text("")
        assert result == ""

    def test_nested_tags(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        result = ingestor._extract_text("<div><p>nested <span>tags</span></p></div>")
        assert result == "nested tags"


# ---- RSSIngestor._compute_hash ----------------------------------------------


class TestComputeHash:
    """Deterministic dedup hashing."""

    def test_deterministic(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        h1 = ingestor._compute_hash("https://example.com", "hello")
        h2 = ingestor._compute_hash("https://example.com", "hello")
        assert h1 == h2

    def test_different_input_different_hash(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        h1 = ingestor._compute_hash("https://example.com/a", "hello")
        h2 = ingestor._compute_hash("https://example.com/b", "hello")
        assert h1 != h2

    def test_case_insensitive(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        h1 = ingestor._compute_hash("https://Example.COM", "Hello World")
        h2 = ingestor._compute_hash("https://example.com", "hello world")
        assert h1 == h2

    def test_returns_hex_string(self) -> None:
        settings = _make_settings()
        ingestor = RSSIngestor(settings=settings, feeds=[])
        h = ingestor._compute_hash("url", "text")
        assert len(h) == 64  # SHA-256 hex digest


# ---- RSSIngestor.fetch_all --------------------------------------------------


class TestFetchAll:
    """Concurrent feed fetching with error handling."""

    async def test_fetch_all_with_mock_responses(self) -> None:
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Article</title>
              <link>https://example.com/article</link>
              <description>Content</description>
            </item>
          </channel>
        </rss>"""

        settings = _make_settings()
        feed = _make_feed_config()
        ingestor = RSSIngestor(settings=settings, feeds=[feed])

        mock_response = MagicMock()
        mock_response.content = rss_xml
        mock_response.raise_for_status = MagicMock()

        with patch("src.evidence.rss_ingestor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            articles = await ingestor.fetch_all()

        assert len(articles) == 1
        assert articles[0].title == "Article"

    async def test_failed_feed_continues(self) -> None:
        """If one feed fails, others still succeed."""
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Good</title>
              <link>https://example.com/good</link>
            </item>
          </channel>
        </rss>"""

        settings = _make_settings()
        good_feed = _make_feed_config(name="Good", url="https://good.com/feed")
        bad_feed = _make_feed_config(name="Bad", url="https://bad.com/feed")
        ingestor = RSSIngestor(settings=settings, feeds=[good_feed, bad_feed])

        good_response = MagicMock()
        good_response.content = rss_xml
        good_response.raise_for_status = MagicMock()

        async def mock_get(url: str, **kwargs):
            if "bad.com" in url:
                raise ConnectionError("Connection refused")
            return good_response

        with patch("src.evidence.rss_ingestor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            articles = await ingestor.fetch_all()

        assert len(articles) == 1
        assert articles[0].title == "Good"

    async def test_deduplicates_articles(self) -> None:
        """Duplicate articles across feeds are deduplicated."""
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Same Article</title>
              <link>https://example.com/same</link>
              <description>Same content</description>
            </item>
          </channel>
        </rss>"""

        settings = _make_settings()
        feed1 = _make_feed_config(name="Feed1", url="https://feed1.com/rss")
        feed2 = _make_feed_config(name="Feed2", url="https://feed2.com/rss")
        ingestor = RSSIngestor(settings=settings, feeds=[feed1, feed2])

        mock_response = MagicMock()
        mock_response.content = rss_xml
        mock_response.raise_for_status = MagicMock()

        with patch("src.evidence.rss_ingestor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            articles = await ingestor.fetch_all()

        # Same URL + same text => same hash => deduplicated to 1
        assert len(articles) == 1


# ---- EvidenceLinker.link ----------------------------------------------------


class TestEvidenceLinker:
    """Keyword-based evidence-to-market linking."""

    async def test_links_matching_evidence(self) -> None:
        market = _make_market(
            market_id="m1",
            question="Will Bitcoin exceed $100,000?",
        )
        evidence = _make_evidence_item(
            title="Bitcoin hits new high",
            extracted_text="Bitcoin surged to a new all-time high today.",
        )

        linker = EvidenceLinker(max_per_market=10)
        result = await linker.link([evidence], [market])

        assert "m1" in result
        assert len(result["m1"]) == 1

    async def test_no_match_excluded(self) -> None:
        market = _make_market(
            market_id="m1",
            question="Will it rain in Tokyo tomorrow?",
        )
        evidence = _make_evidence_item(
            title="Bitcoin hits new high",
            extracted_text="Bitcoin surged to a new all-time high today.",
        )

        linker = EvidenceLinker(max_per_market=10)
        result = await linker.link([evidence], [market])

        assert result.get("m1", []) == []

    async def test_respects_max_per_market(self) -> None:
        market = _make_market(
            market_id="m1",
            question="Will Bitcoin exceed $100,000?",
        )
        items = [
            _make_evidence_item(
                evidence_id=i,
                title=f"Bitcoin article {i}",
                extracted_text=f"Bitcoin content {i}",
                url=f"https://example.com/{i}",
            )
            for i in range(20)
        ]

        linker = EvidenceLinker(max_per_market=5)
        result = await linker.link(items, [market])

        assert len(result["m1"]) == 5

    async def test_multiple_markets(self) -> None:
        market1 = _make_market(market_id="m1", question="Will Bitcoin price rise?")
        market2 = _make_market(market_id="m2", question="Will Ethereum merge succeed?")

        bitcoin_ev = _make_evidence_item(
            evidence_id=1,
            title="Bitcoin rally",
            extracted_text="Bitcoin is rallying hard.",
            url="https://example.com/btc",
        )
        ethereum_ev = _make_evidence_item(
            evidence_id=2,
            title="Ethereum update",
            extracted_text="Ethereum merge is on track.",
            url="https://example.com/eth",
        )

        linker = EvidenceLinker(max_per_market=10)
        result = await linker.link([bitcoin_ev, ethereum_ev], [market1, market2])

        assert len(result.get("m1", [])) >= 1
        assert len(result.get("m2", [])) >= 1

    async def test_sorted_by_score_descending(self) -> None:
        market = _make_market(
            market_id="m1",
            question="Will Bitcoin price reach $100,000?",
        )
        low_match = _make_evidence_item(
            evidence_id=1,
            title="Market update",
            extracted_text="Bitcoin mentioned once.",
            url="https://example.com/low",
        )
        high_match = _make_evidence_item(
            evidence_id=2,
            title="Bitcoin Bitcoin price analysis",
            extracted_text="Bitcoin price Bitcoin $100,000 bitcoin reach.",
            url="https://example.com/high",
        )

        linker = EvidenceLinker(max_per_market=10)
        result = await linker.link([low_match, high_match], [market])

        # High match should come first.
        assert result["m1"][0].evidence_id == 2

    async def test_empty_evidence(self) -> None:
        market = _make_market()
        linker = EvidenceLinker(max_per_market=10)
        result = await linker.link([], [market])
        assert result.get("m1", []) == []

    async def test_empty_markets(self) -> None:
        evidence = _make_evidence_item()
        linker = EvidenceLinker(max_per_market=10)
        result = await linker.link([evidence], [])
        assert result == {}

    async def test_semantic_link_with_mocked_embedder(self) -> None:
        """Semantic path: embedder returns vectors, linker ranks by cosine sim."""
        market = _make_market(
            market_id="m1",
            question="Will Bitcoin exceed $100,000?",
        )
        relevant = _make_evidence_item(
            evidence_id=1,
            title="Bitcoin price surge",
            extracted_text="Bitcoin is climbing toward six figures.",
            url="https://example.com/relevant",
        )
        irrelevant = _make_evidence_item(
            evidence_id=2,
            title="Weather forecast",
            extracted_text="Sunny skies expected tomorrow.",
            url="https://example.com/irrelevant",
        )

        # Mock embedder returning unit vectors: market and relevant are close,
        # irrelevant is orthogonal.
        embedder = AsyncMock(spec=EvidenceEmbedder)
        embedder.embed_texts = AsyncMock(return_value=[
            [1.0, 0.0, 0.0],   # market
            [0.9, 0.1, 0.0],   # relevant evidence (high cosine sim)
            [0.0, 0.0, 1.0],   # irrelevant evidence (zero cosine sim)
        ])

        linker = EvidenceLinker(
            max_per_market=10,
            embedder=embedder,
            similarity_threshold=0.25,
        )
        result = await linker.link([relevant, irrelevant], [market])

        assert "m1" in result
        assert len(result["m1"]) == 1
        assert result["m1"][0].evidence_id == 1

    def test_default_similarity_threshold_is_035(self) -> None:
        """EvidenceLinker default similarity_threshold matches policy default."""
        from src.evidence.linker import DEFAULT_SIMILARITY_THRESHOLD

        linker = EvidenceLinker(max_per_market=10)
        assert linker.similarity_threshold == 0.35
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.35

    async def test_semantic_fallback_on_error(self) -> None:
        """When embedder raises, linker falls back to keyword matching."""
        market = _make_market(
            market_id="m1",
            question="Will Bitcoin exceed $100,000?",
        )
        evidence = _make_evidence_item(
            title="Bitcoin hits new high",
            extracted_text="Bitcoin surged to a new all-time high today.",
        )

        embedder = AsyncMock(spec=EvidenceEmbedder)
        embedder.embed_texts = AsyncMock(side_effect=RuntimeError("API down"))

        linker = EvidenceLinker(max_per_market=10, embedder=embedder)
        result = await linker.link([evidence], [market])

        # Should still match via keyword fallback.
        assert "m1" in result
        assert len(result["m1"]) == 1


# ---- EvidenceEmbedder -------------------------------------------------------


class TestEvidenceEmbedder:
    """Unit tests for the embedding helper methods."""

    def test_cosine_similarity_identical(self) -> None:
        vec = [1.0, 0.0, 0.0]
        assert EvidenceEmbedder.cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert EvidenceEmbedder.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert EvidenceEmbedder.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_cosine_similarity_empty_vectors(self) -> None:
        assert EvidenceEmbedder.cosine_similarity([], [1.0]) == 0.0
        assert EvidenceEmbedder.cosine_similarity([1.0], []) == 0.0
        assert EvidenceEmbedder.cosine_similarity([], []) == 0.0

    def test_cosine_similarity_mismatched_lengths(self) -> None:
        assert EvidenceEmbedder.cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_cosine_similarity_zero_vector(self) -> None:
        assert EvidenceEmbedder.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_cache_key_deterministic(self) -> None:
        k1 = EvidenceEmbedder._cache_key("hello world")
        k2 = EvidenceEmbedder._cache_key("hello world")
        assert k1 == k2

    def test_cache_key_case_insensitive(self) -> None:
        k1 = EvidenceEmbedder._cache_key("Hello World")
        k2 = EvidenceEmbedder._cache_key("hello world")
        assert k1 == k2

    def test_cache_key_strips_whitespace(self) -> None:
        k1 = EvidenceEmbedder._cache_key("  hello world  ")
        k2 = EvidenceEmbedder._cache_key("hello world")
        assert k1 == k2

    def test_clear_cache(self) -> None:
        from src.evidence.embedder import _embedding_cache

        _embedding_cache["test_key"] = [1.0, 2.0]
        EvidenceEmbedder.clear_cache()
        assert len(_embedding_cache) == 0

    async def test_embed_texts_caches_results(self) -> None:
        """Verify that repeated calls use the cache."""
        from src.evidence.embedder import _embedding_cache

        EvidenceEmbedder.clear_cache()

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
        ]

        embedder = EvidenceEmbedder(api_key="test-key")
        embedder.client = AsyncMock()
        embedder.client.embeddings.create = AsyncMock(return_value=mock_response)

        # First call should hit the API.
        result1 = await embedder.embed_texts(["hello"])
        assert result1 == [[0.1, 0.2, 0.3]]
        assert embedder.client.embeddings.create.call_count == 1

        # Second call should use cache — no new API call.
        result2 = await embedder.embed_texts(["hello"])
        assert result2 == [[0.1, 0.2, 0.3]]
        assert embedder.client.embeddings.create.call_count == 1

        EvidenceEmbedder.clear_cache()

    async def test_embed_texts_returns_empty_on_error(self) -> None:
        """API errors produce empty vectors, not exceptions."""
        EvidenceEmbedder.clear_cache()

        embedder = EvidenceEmbedder(api_key="test-key")
        embedder.client = AsyncMock()
        embedder.client.embeddings.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        result = await embedder.embed_texts(["hello"])
        assert result == [[]]

        EvidenceEmbedder.clear_cache()

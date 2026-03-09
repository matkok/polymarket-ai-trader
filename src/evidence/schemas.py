from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RSSFeedConfig(BaseModel):
    """Configuration for a single RSS feed source."""

    name: str
    url: str
    category: str  # general, crypto, politics, sports, tech


class FetchedArticle(BaseModel):
    """An article fetched from an RSS feed."""

    url: str
    title: str
    published_ts: datetime | None = None
    raw_html: str = ""
    extracted_text: str = ""
    source_name: str = ""


class EvidenceRecord(BaseModel):
    """Pydantic mirror of the EvidenceItem ORM model."""

    evidence_id: int | None = None
    ts_utc: datetime
    source_type: str
    url: str
    title: str
    published_ts_utc: datetime | None = None
    raw_content_ref: str | None = None
    extracted_text: str = ""
    content_hash: str = ""


DEFAULT_RSS_FEEDS: list[RSSFeedConfig] = [
    # General news
    RSSFeedConfig(name="BBC News", url="https://feeds.bbci.co.uk/news/rss.xml", category="general"),
    RSSFeedConfig(name="NPR News", url="https://feeds.npr.org/1001/rss.xml", category="general"),
    RSSFeedConfig(name="Al Jazeera", url="https://www.aljazeera.com/xml/rss/all.xml", category="general"),
    # Crypto
    RSSFeedConfig(name="CoinDesk", url="https://www.coindesk.com/arc/outboundfeeds/rss", category="crypto"),
    RSSFeedConfig(name="The Block", url="https://www.theblock.co/rss.xml", category="crypto"),
    RSSFeedConfig(name="CoinTelegraph", url="https://cointelegraph.com/rss", category="crypto"),
    # Politics
    RSSFeedConfig(name="Politico", url="https://rss.politico.com/politics-news.xml", category="politics"),
    RSSFeedConfig(name="The Hill", url="https://thehill.com/feed/?feed=partnerfeed-news-feed&format=rss", category="politics"),
    # Sports
    RSSFeedConfig(name="ESPN", url="https://www.espn.com/espn/rss/news", category="sports"),
    # Tech
    RSSFeedConfig(name="Ars Technica", url="https://feeds.arstechnica.com/arstechnica/index", category="tech"),
    RSSFeedConfig(name="TechCrunch", url="https://techcrunch.com/feed/", category="tech"),
]

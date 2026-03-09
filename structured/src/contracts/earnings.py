"""Earnings/SEC filing contract parser — EPS, revenue, guidance, 10-K/Q filings."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from src.contracts.base import ContractParser, ContractSpec, ParseResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

_EARNINGS_KEYWORDS: list[str] = [
    "earnings",
    "quarterly results",
    "annual report",
    "beat estimates",
    "miss estimates",
    "guidance",
    "sec filing",
]

_EARNINGS_KEYWORDS_BOUNDARY: list[str] = [
    "eps",
    "revenue",
    "10-k",
    "10-q",
    "8-k",
]

_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in _EARNINGS_KEYWORDS_BOUNDARY
]

_PLAIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(re.escape(kw), re.IGNORECASE)
    for kw in _EARNINGS_KEYWORDS
]

# ---------------------------------------------------------------------------
# Negative context filters
# ---------------------------------------------------------------------------

_EARNINGS_NEGATIVES: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bcrypto\s+earnings\b",
        r"\bearnings\s+call\s+sentiment\b",
        r"\bsocial\s+media\s+earnings\b",
        r"\bstaking\s+earnings\b",
        r"\bmining\s+earnings\b",
    ]
]

# ---------------------------------------------------------------------------
# EarningsContractSpec
# ---------------------------------------------------------------------------


@dataclass
class EarningsContractSpec(ContractSpec):
    """Structured fields for an earnings/filing market."""

    company: str = ""
    ticker: str = ""
    metric: str = ""          # eps, revenue, guidance, filing
    threshold: float | None = None
    threshold_unit: str = ""  # USD, B, M
    comparison: str = ""      # beat, miss, above, below
    filing_type: str = ""     # 10-K, 10-Q, 8-K
    fiscal_period: str = ""   # Q1 2026, FY 2025
    cik: str = ""

    def to_json(self) -> dict[str, object]:  # type: ignore[override]
        return {
            "category": self.category,
            "company": self.company,
            "ticker": self.ticker,
            "metric": self.metric,
            "threshold": self.threshold,
            "threshold_unit": self.threshold_unit,
            "comparison": self.comparison,
            "filing_type": self.filing_type,
            "fiscal_period": self.fiscal_period,
            "cik": self.cik,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_comparison(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("beat", "beats", "exceed", "exceeds", "above", "surpass"):
        return "above"
    if raw in ("miss", "misses", "below", "under", "fall short"):
        return "below"
    if raw in ("at least", "at_least"):
        return "at_least"
    return raw


def _normalize_metric(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("eps", "earnings per share"):
        return "eps"
    if raw in ("revenue", "revenues", "sales", "top line"):
        return "revenue"
    if raw in ("guidance", "outlook", "forecast"):
        return "guidance"
    if raw in ("10-k", "annual report"):
        return "filing_10k"
    if raw in ("10-q", "quarterly report"):
        return "filing_10q"
    if raw in ("8-k",):
        return "filing_8k"
    return raw


def _has_negative_context(text: str) -> str | None:
    for pat in _EARNINGS_NEGATIVES:
        if pat.search(text):
            return f"earnings_negative: {pat.pattern}"
    return None


def _find_keyword_matches(text: str) -> list[str]:
    matched: list[str] = []
    for kw, pat in zip(_EARNINGS_KEYWORDS, _PLAIN_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    for kw, pat in zip(_EARNINGS_KEYWORDS_BOUNDARY, _BOUNDARY_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    return matched


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# 1. EPS/Revenue threshold:
#    "Will Apple's EPS exceed $1.50 in Q1 2026?"
#    "Will Tesla revenue beat $25B in Q4 2025?"
_RE_EARNINGS_THRESHOLD = re.compile(
    r"(?:will\s+)?(?P<company>[A-Za-z][A-Za-z .&']+?)(?:'s?\s+)"
    r"(?P<metric>eps|earnings\s+per\s+share|revenue|revenues?|sales)\s+"
    r"(?P<comp>exceed|exceeds?|beat|beats?|miss|misses?|be\s+above|be\s+below|"
    r"surpass|fall\s+short)\s+"
    r"\$?(?P<threshold>[\d,.]+)(?P<unit>[BMK])?"
    r"(?:\s+(?:in|for|during)\s+(?P<period>[^?.]+))?",
    re.IGNORECASE,
)

# 2. Filing existence:
#    "Will Apple file its 10-K by March 2026?"
_RE_FILING = re.compile(
    r"(?:will\s+)?(?P<company>[A-Za-z][A-Za-z .&']+?)\s+"
    r"(?:file|submit|release)\s+(?:its?\s+)?(?P<filing>10-[KkQq]|8-[Kk])"
    r"(?:\s+(?:by|before|in|for)\s+(?P<period>[^?.]+))?",
    re.IGNORECASE,
)

# 3. Ticker-based: "Will $AAPL EPS beat estimates in Q1 2026?"
_RE_TICKER_EARNINGS = re.compile(
    r"(?:will\s+)?\$?(?P<ticker>[A-Z]{1,5})\s+"
    r"(?P<metric>eps|earnings|revenue)\s+"
    r"(?P<comp>beat|miss|exceed|surpass)\s+"
    r"(?:estimates?\s*)?"
    r"(?:(?:of\s+)?\$?(?P<threshold>[\d,.]+)\s*(?P<unit>[BMK])?)?"
    r"(?:\s+(?:in|for)\s+(?P<period>[^?.]+))?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# EarningsParser
# ---------------------------------------------------------------------------


class EarningsParser(ContractParser):
    """Parse earnings and SEC filing prediction markets."""

    @property
    def name(self) -> str:
        return "earnings_v1"

    @property
    def category(self) -> str:
        return "earnings"

    def can_parse(self, question: str, rules_text: str | None) -> bool:
        combined = question + " " + (rules_text or "")
        for pat in _BOUNDARY_PATTERNS:
            if pat.search(combined):
                return True
        for pat in _PLAIN_PATTERNS:
            if pat.search(combined):
                return True
        return False

    def parse(self, question: str, rules_text: str | None) -> ParseResult:
        combined = question + " " + (rules_text or "")

        negative = _has_negative_context(combined)
        if negative:
            logger.debug(
                "earnings_parse_negative_filter",
                reason=negative,
                matched_keywords=_find_keyword_matches(combined),
            )
            return ParseResult(matched=False, reject_reason=negative)

        # 1. Earnings threshold (EPS/Revenue)
        m = _RE_EARNINGS_THRESHOLD.search(combined)
        if m:
            threshold_str = m.group("threshold").replace(",", "")
            unit = (m.group("unit") or "").upper()
            spec = EarningsContractSpec(
                category="earnings",
                company=m.group("company").strip().rstrip("'"),
                metric=_normalize_metric(m.group("metric")),
                threshold=float(threshold_str),
                threshold_unit=unit or "USD",
                comparison=_normalize_comparison(m.group("comp")),
                fiscal_period=(m.group("period") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "earnings_parse_match",
                pattern="earnings_threshold",
                company=spec.company,
                metric=spec.metric,
            )
            return ParseResult(matched=True, category="earnings", spec=spec, confidence=0.90)

        # 2. Filing existence
        m = _RE_FILING.search(combined)
        if m:
            filing_type = m.group("filing").upper()
            spec = EarningsContractSpec(
                category="earnings",
                company=m.group("company").strip(),
                metric=_normalize_metric(filing_type),
                filing_type=filing_type,
                fiscal_period=(m.group("period") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "earnings_parse_match",
                pattern="filing",
                company=spec.company,
                filing_type=filing_type,
            )
            return ParseResult(matched=True, category="earnings", spec=spec, confidence=0.85)

        # 3. Ticker-based
        m = _RE_TICKER_EARNINGS.search(combined)
        if m:
            threshold_raw = m.group("threshold")
            unit = (m.group("unit") or "").upper() if m.group("unit") else ""
            spec = EarningsContractSpec(
                category="earnings",
                ticker=m.group("ticker").upper(),
                metric=_normalize_metric(m.group("metric")),
                threshold=float(threshold_raw.replace(",", "")) if threshold_raw else None,
                threshold_unit=unit or "USD",
                comparison=_normalize_comparison(m.group("comp")),
                fiscal_period=(m.group("period") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "earnings_parse_match",
                pattern="ticker",
                ticker=spec.ticker,
                metric=spec.metric,
            )
            return ParseResult(matched=True, category="earnings", spec=spec, confidence=0.85)

        # No structural pattern matched.
        matched_kws = _find_keyword_matches(combined)
        logger.debug(
            "earnings_parse_keyword_only",
            matched_keywords=matched_kws,
            reject_reason="keyword_only_no_structure",
        )
        return ParseResult(
            matched=False, reject_reason="keyword_only_no_structure"
        )

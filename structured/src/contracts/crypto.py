"""Crypto price threshold contract parser — BTC, ETH, SOL price markets."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from src.contracts.base import ContractParser, ContractSpec, ParseResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

# Multi-word keywords safe from substring issues.
_CRYPTO_KEYWORDS: list[str] = [
    "bitcoin",
    "ethereum",
    "solana",
    "price of",
    "trading above",
    "trading below",
    "crypto price",
    "all time high",
    "all-time high",
]

# Short keywords that need word-boundary matching.
_CRYPTO_KEYWORDS_BOUNDARY: list[str] = [
    "btc",
    "eth",
    "sol",
]

_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in _CRYPTO_KEYWORDS_BOUNDARY
]

_PLAIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(re.escape(kw), re.IGNORECASE)
    for kw in _CRYPTO_KEYWORDS
]

# ---------------------------------------------------------------------------
# Negative context filters
# ---------------------------------------------------------------------------

_CRYPTO_NEGATIVES: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bdao\s+vote\b",
        r"\bgovernance\s+proposal\b",
        r"\bnft\b",
        r"\bnon[- ]fungible\b",
        r"\bblockchain\s+tech(?:nology)?\b",
        r"\bsmart\s+contract\s+(?:audit|deploy|bug)\b",
        r"\bmining\s+difficulty\b",
        r"\bhash\s+rate\b",
        r"\bgas\s+fee\b",
        r"\bgas\s+price\b",
        # Ethnic false positive for "eth"
        r"\bethnicit",
        r"\bethics\b",
        r"\bethanol\b",
    ]
]

# ---------------------------------------------------------------------------
# CryptoContractSpec
# ---------------------------------------------------------------------------


@dataclass
class CryptoContractSpec(ContractSpec):
    """Structured fields for a crypto price threshold market."""

    asset: str = ""
    threshold: float | None = None
    threshold_unit: str = "USD"
    comparison: str = ""
    exchange: str = ""
    reference_price: str = "last_trade"
    resolution_timestamp: str = ""
    date_description: str = ""

    def to_json(self) -> dict[str, object]:  # type: ignore[override]
        return {
            "category": self.category,
            "asset": self.asset,
            "threshold": self.threshold,
            "threshold_unit": self.threshold_unit,
            "comparison": self.comparison,
            "exchange": self.exchange,
            "reference_price": self.reference_price,
            "resolution_timestamp": self.resolution_timestamp,
            "date_description": self.date_description,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ASSET_MAP: dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "cardano": "ADA",
    "ada": "ADA",
    "xrp": "XRP",
    "ripple": "XRP",
}


def _normalize_asset(raw: str) -> str:
    return _ASSET_MAP.get(raw.strip().lower(), raw.strip().upper())


def _normalize_comparison(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("exceed", "exceeds", "above", "over", "more than", "higher than",
               "be above", "trading above", "rise above", "reach", "hit"):
        return "above"
    if raw in ("below", "under", "less than", "lower than", "fall below",
               "drop below", "be below", "trading below"):
        return "below"
    if raw in ("at least", "at_least"):
        return "at_least"
    if raw in ("at most", "at_most"):
        return "at_most"
    return raw


def _normalize_exchange(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("coinbase", "coinbase pro", "coinbase exchange"):
        return "coinbase"
    if raw in ("binance", "binance.us", "binance spot"):
        return "binance"
    if raw in ("kraken",):
        return "kraken"
    return raw


_SUFFIX_MULTIPLIERS: dict[str, float] = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}


def _apply_suffix(value: float, suffix: str | None) -> float:
    """Multiply *value* by the suffix multiplier (k/m/b)."""
    if not suffix:
        return value
    return value * _SUFFIX_MULTIPLIERS.get(suffix.lower(), 1.0)


# Known all-time high prices (USD, approximate). Updated periodically.
_ATH_PRICES: dict[str, float] = {
    "BTC": 109_000.0,
    "ETH": 4_890.0,
    "SOL": 295.0,
    "DOGE": 0.74,
    "ADA": 3.10,
    "XRP": 3.84,
}


def _has_negative_context(text: str) -> str | None:
    for pat in _CRYPTO_NEGATIVES:
        if pat.search(text):
            return f"crypto_negative: {pat.pattern}"
    return None


def _find_keyword_matches(text: str) -> list[str]:
    matched: list[str] = []
    for kw, pat in zip(_CRYPTO_KEYWORDS, _PLAIN_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    for kw, pat in zip(_CRYPTO_KEYWORDS_BOUNDARY, _BOUNDARY_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    return matched


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# 1. Price threshold: "Will BTC exceed $100,000 on Coinbase by June 30?"
_RE_PRICE_THRESHOLD = re.compile(
    r"(?:will\s+)?(?:the\s+)?(?:price\s+of\s+)?"
    r"(?P<asset>bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ada|xrp|ripple)\s+"
    r"(?P<comp>exceed|exceeds?|be\s+above|be\s+below|fall\s+below|drop\s+below|"
    r"reach|hit|trading\s+above|trading\s+below|be\s+over|be\s+under|"
    r"rise\s+above|go\s+above|go\s+below)\s+"
    r"\$?(?P<threshold>[\d,]+(?:\.\d+)?)(?P<suffix>[kKmMbB])?\s*"
    r"(?:(?:on|at)\s+(?P<exchange>coinbase|binance|kraken)\s*)?"
    r"(?:(?:by|on|before|in|during)\s+(?P<date>.+?))?",
    re.IGNORECASE,
)

# 2. Trading above/below: "Will ETH be trading above $5,000 on December 31?"
_RE_TRADING = re.compile(
    r"(?:will\s+)?(?:the\s+)?(?:price\s+of\s+)?"
    r"(?P<asset>bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ada|xrp|ripple)\s+"
    r"(?:be\s+)?(?P<comp>trading\s+above|trading\s+below)\s+"
    r"\$?(?P<threshold>[\d,]+(?:\.\d+)?)(?P<suffix>[kKmMbB])?"
    r"(?:\s+(?:on|at)\s+(?P<exchange>coinbase|binance|kraken))?"
    r"(?:\s+(?:on|by|before|in)\s+(?P<date>.+?))?",
    re.IGNORECASE,
)


# 3. All-time high: "Ethereum all time high by March 31, 2026?"
_RE_ATH = re.compile(
    r"(?:will\s+)?(?:the\s+)?(?:price\s+of\s+)?"
    r"(?P<asset>bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ada|xrp|ripple)\s+"
    r"(?:(?:hit|reach|set|break|new)\s+)?(?:a\s+)?(?:new\s+)?"
    r"(?P<comp>all[- ]?time\s+high|ath|new\s+ath)"
    r"(?:\s+(?:on|at)\s+(?P<exchange>coinbase|binance|kraken))?"
    r"(?:\s+(?:by|on|before|in|during)\s+(?P<date>.+))?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# CryptoParser
# ---------------------------------------------------------------------------


class CryptoParser(ContractParser):
    """Parse crypto price threshold prediction markets."""

    @property
    def name(self) -> str:
        return "crypto_v1"

    @property
    def category(self) -> str:
        return "crypto"

    # -- keyword gate --------------------------------------------------------

    def can_parse(self, question: str, rules_text: str | None) -> bool:
        combined = question + " " + (rules_text or "")
        for pat in _BOUNDARY_PATTERNS:
            if pat.search(combined):
                return True
        for pat in _PLAIN_PATTERNS:
            if pat.search(combined):
                return True
        return False

    # -- full parse ----------------------------------------------------------

    def parse(self, question: str, rules_text: str | None) -> ParseResult:
        combined = question + " " + (rules_text or "")

        # Early rejection: check negative context.
        negative = _has_negative_context(combined)
        if negative:
            logger.debug(
                "crypto_parse_negative_filter",
                reason=negative,
                matched_keywords=_find_keyword_matches(combined),
            )
            return ParseResult(matched=False, reject_reason=negative)

        # 1. Price threshold
        m = _RE_PRICE_THRESHOLD.search(combined)
        if m:
            threshold_str = m.group("threshold").replace(",", "")
            threshold_val = _apply_suffix(float(threshold_str), m.group("suffix"))
            exchange_raw = m.group("exchange") or ""
            spec = CryptoContractSpec(
                category="crypto",
                asset=_normalize_asset(m.group("asset")),
                threshold=threshold_val,
                threshold_unit="USD",
                comparison=_normalize_comparison(m.group("comp")),
                exchange=_normalize_exchange(exchange_raw),
                date_description=(m.group("date") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "crypto_parse_match",
                pattern="price_threshold",
                asset=spec.asset,
                threshold=spec.threshold,
                exchange=spec.exchange,
            )
            return ParseResult(matched=True, category="crypto", spec=spec, confidence=0.90)

        # 2. Trading above/below
        m = _RE_TRADING.search(combined)
        if m:
            threshold_str = m.group("threshold").replace(",", "")
            threshold_val = _apply_suffix(float(threshold_str), m.group("suffix"))
            exchange_raw = m.group("exchange") or ""
            spec = CryptoContractSpec(
                category="crypto",
                asset=_normalize_asset(m.group("asset")),
                threshold=threshold_val,
                threshold_unit="USD",
                comparison=_normalize_comparison(m.group("comp")),
                exchange=_normalize_exchange(exchange_raw),
                date_description=(m.group("date") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "crypto_parse_match",
                pattern="trading",
                asset=spec.asset,
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="crypto", spec=spec, confidence=0.85)

        # 3. All-time high
        m = _RE_ATH.search(combined)
        if m:
            asset = _normalize_asset(m.group("asset"))
            ath_price = _ATH_PRICES.get(asset)
            if ath_price is not None:
                exchange_raw = m.group("exchange") or ""
                spec = CryptoContractSpec(
                    category="crypto",
                    asset=asset,
                    threshold=ath_price,
                    threshold_unit="USD",
                    comparison="above",
                    exchange=_normalize_exchange(exchange_raw),
                    date_description=(m.group("date") or "").strip().rstrip("?. "),
                )
                logger.debug(
                    "crypto_parse_match",
                    pattern="ath",
                    asset=spec.asset,
                    threshold=spec.threshold,
                )
                return ParseResult(matched=True, category="crypto", spec=spec, confidence=0.85)

        # No structural pattern matched.
        matched_kws = _find_keyword_matches(combined)
        logger.debug(
            "crypto_parse_keyword_only",
            matched_keywords=matched_kws,
            reject_reason="keyword_only_no_structure",
        )
        return ParseResult(
            matched=False, reject_reason="keyword_only_no_structure"
        )

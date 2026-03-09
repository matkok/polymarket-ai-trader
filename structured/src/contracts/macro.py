"""Macro-economic contract parser — CPI, unemployment, GDP, payrolls, fed rate."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from src.contracts.base import ContractParser, ContractSpec, ParseResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keywords used by can_parse()
# ---------------------------------------------------------------------------

# Multi-word keywords safe from substring false positives.
_MACRO_KEYWORDS: list[str] = [
    "consumer price index",
    "nonfarm payroll",
    "jobs report",
    "retail sales",
    "the fed",
    "fed rate",
    "federal reserve",
    "rate hike",
    "rate cut",
    "housing starts",
]

# Short keywords that need word-boundary matching to avoid false positives
# (e.g. "gdp" in "dogdp", "cpi" in a ticker, "pce" in "piece").
_MACRO_KEYWORDS_BOUNDARY: list[str] = [
    "cpi",
    "inflation",
    "unemployment",
    "gdp",
    "payrolls",
    "pce",
    "fomc",
    "ppi",
]

# Pre-compiled word-boundary patterns.
_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in _MACRO_KEYWORDS_BOUNDARY
]

# Pre-compiled plain patterns (multi-word, safe from substring issues).
_PLAIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(re.escape(kw), re.IGNORECASE)
    for kw in _MACRO_KEYWORDS
]

# ---------------------------------------------------------------------------
# Negative context filters — crypto/DeFi and other false-positive sources
# ---------------------------------------------------------------------------

_CRYPTO_NEGATIVES: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bdefi\b",
        r"\btoken\b",
        r"\bblockchain\b",
        r"\byield\s+farm",
        r"\bstaking\b",
        r"\bcrypto\s+inflation\b",
        r"\btoken(?:omic)?s?\s+inflation\b",
        r"\bmint(?:ing)?\s+rate\b",
        r"\bprotocol\s+rate\b",
        r"\bapy\b",
        r"\bapr\b",
        r"\bdao\s+vote\b",
        r"\bgovernance\s+proposal\b",
        r"\bsmart\s+contract\b",
    ]
]

# ---------------------------------------------------------------------------
# MacroContractSpec
# ---------------------------------------------------------------------------


@dataclass
class MacroContractSpec(ContractSpec):
    """Structured fields for a macro-economic market."""

    indicator: str = ""
    threshold: float | None = None
    threshold_unit: str = ""
    comparison: str = ""
    release_period: str = ""
    release_date: str = ""
    bls_series_id: str = ""
    fred_series_id: str = ""

    def to_json(self) -> dict[str, object]:  # type: ignore[override]
        return {
            "category": self.category,
            "indicator": self.indicator,
            "threshold": self.threshold,
            "threshold_unit": self.threshold_unit,
            "comparison": self.comparison,
            "release_period": self.release_period,
            "release_date": self.release_date,
            "bls_series_id": self.bls_series_id,
            "fred_series_id": self.fred_series_id,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_indicator(raw: str) -> str:
    """Map free-text indicator names to canonical keys."""
    raw = raw.strip().lower()
    if raw in ("cpi", "consumer price index"):
        return "cpi"
    if raw in ("core cpi", "core consumer price index"):
        return "core_cpi"
    if raw in ("pce", "personal consumption expenditures"):
        return "pce"
    if raw in ("core pce",):
        return "core_pce"
    if raw in ("unemployment", "unemployment rate"):
        return "unemployment"
    if raw in ("gdp", "gdp growth", "gross domestic product"):
        return "gdp"
    if raw in ("nonfarm payrolls", "nonfarm payroll", "non-farm payrolls", "non-farm payroll"):
        return "nonfarm_payrolls"
    if raw in ("ppi", "producer price index"):
        return "ppi"
    if raw in ("retail sales",):
        return "retail_sales"
    if raw in ("housing starts",):
        return "housing_starts"
    if "fed" in raw or "fomc" in raw:
        return "fed_rate"
    return raw


def _normalize_fed_comparison(raw: str) -> str:
    """Normalise Fed action verbs."""
    raw = raw.strip().lower()
    if raw in ("raise", "raises", "hike", "hikes", "increase", "increases"):
        return "raise"
    if raw in ("cut", "cuts", "lower", "lowers", "reduce", "reduces"):
        return "cut"
    if raw in ("hold", "holds", "maintain", "maintains", "pause", "pauses", "keep", "keeps"):
        return "hold"
    return raw


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# 1. Indicator threshold (CPI / unemployment / core CPI / PPI / PCE etc.)
#    "Will CPI exceed 3.0% in January 2026?"
_RE_INDICATOR = re.compile(
    r"(?:will\s+)?(?:the\s+)?(?:(?:year[- ]over[- ]year|yoy|annual|monthly)\s+)?"
    r"(?P<indicator>(?:core\s+)?(?:cpi|consumer\s+price\s+index|unemployment(?:\s+rate)?|"
    r"pce|ppi|retail\s+sales|housing\s+starts|gdp(?:\s+growth)?|inflation))\s+"
    r"(?P<comp>exceed|exceeds?|be\s+above|be\s+below|fall\s+below|drop\s+below|"
    r"be\s+over|be\s+under|be\s+at\s+least|be\s+at\s+most|rise\s+above|"
    r"come\s+in\s+(?:above|below|under|over))\s+"
    r"(?P<threshold>-?\d+(?:\.\d+)?)\s*(?P<unit>%|percent|bps)?"
    r"(?:\s+(?:in|for|during)\s+(?P<period>.+))?",
    re.IGNORECASE,
)

# 2. Nonfarm payrolls
#    "Will nonfarm payrolls exceed 200K in February 2026?"
_RE_PAYROLLS = re.compile(
    r"(?:will\s+)?(?:the\s+)?(?:us\s+)?(?P<indicator>non[- ]?farm\s+payrolls?)\s+"
    r"(?P<comp>exceed|exceeds?|be\s+above|be\s+below|be\s+over|be\s+under|"
    r"come\s+in\s+(?:above|below)|add\s+(?:more|fewer)\s+than)\s+"
    r"(?P<threshold>-?\d+(?:\.\d+)?)\s*(?P<unit>[KkMm])?"
    r"(?:\s+(?:in|for|during)\s+(?P<period>.+))?",
    re.IGNORECASE,
)

# 3. Fed rate decision
#    "Will the Fed raise rates in March 2026?"
_RE_FED = re.compile(
    r"(?:will\s+)?(?:the\s+)?(?:fed(?:eral\s+reserve)?|fomc)\s+"
    r"(?P<comp>raise|hike|cut|lower|reduce|hold|maintain|pause|keep|increase)\s+"
    r"(?:interest\s+)?rates?"
    r"(?:\s+(?:by\s+(?P<threshold>\d+(?:\.\d+)?)\s*(?P<unit>bps|basis\s+points|%|percent))?)?"
    r"(?:\s+(?:in|at|during|for)\s+(?P<period>.+))?",
    re.IGNORECASE,
)

# 4. GDP growth
#    "Will GDP growth exceed 2.5% in Q1 2026?"
_RE_GDP = re.compile(
    r"(?:will\s+)?(?:the\s+)?(?:us\s+)?(?P<indicator>gdp(?:\s+growth)?)\s+"
    r"(?P<comp>exceed|exceeds?|be\s+above|be\s+below|contract|shrink|"
    r"be\s+(?:positive|negative)|be\s+over|be\s+under)\s+"
    r"(?:(?P<threshold>-?\d+(?:\.\d+)?)\s*(?P<unit>%|percent)?)?"
    r"(?:\s+(?:in|for|during)\s+(?P<period>.+))?",
    re.IGNORECASE,
)


def _normalize_comparison(raw: str) -> str:
    """Collapse indicator comparison phrases to canonical form."""
    raw = raw.strip().lower()
    if raw in (
        "exceed", "exceeds", "above", "be above", "over", "be over",
        "rise above", "come in above", "more than",
    ):
        return "above"
    if raw in (
        "below", "be below", "under", "be under", "fall below",
        "drop below", "come in below", "come in under", "less than",
    ):
        return "below"
    if raw in ("contract", "shrink", "be negative"):
        return "below"
    if raw in ("be positive",):
        return "above"
    if raw in ("be at least", "at least"):
        return "at_least"
    if raw in ("be at most", "at most"):
        return "at_most"
    if "more than" in raw:
        return "above"
    if "fewer than" in raw:
        return "below"
    return raw


# ---------------------------------------------------------------------------
# MacroParser
# ---------------------------------------------------------------------------


def _has_negative_context(text: str) -> str | None:
    """Check for crypto/DeFi context that blocks macro classification.

    Returns the reject reason string, or ``None`` if clean.
    """
    for pat in _CRYPTO_NEGATIVES:
        if pat.search(text):
            return f"crypto_context: {pat.pattern}"
    return None


def _find_keyword_matches(text: str) -> list[str]:
    """Return list of macro keywords found (for diagnostics)."""
    matched: list[str] = []
    for kw, pat in zip(_MACRO_KEYWORDS, _PLAIN_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    for kw, pat in zip(_MACRO_KEYWORDS_BOUNDARY, _BOUNDARY_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    return matched


class MacroParser(ContractParser):
    """Parse macro-economic prediction markets."""

    @property
    def name(self) -> str:
        return "macro_v1"

    @property
    def category(self) -> str:
        return "macro"

    # -- keyword gate --------------------------------------------------------

    def can_parse(self, question: str, rules_text: str | None) -> bool:
        """Check if text contains macro keywords (word-boundary safe)."""
        combined = question + " " + (rules_text or "")

        # Check word-boundary keywords first (cpi, gdp, pce, etc.).
        for pat in _BOUNDARY_PATTERNS:
            if pat.search(combined):
                return True

        # Check multi-word keywords (consumer price index, etc.).
        for pat in _PLAIN_PATTERNS:
            if pat.search(combined):
                return True

        return False

    # -- full parse ----------------------------------------------------------

    def parse(self, question: str, rules_text: str | None) -> ParseResult:
        combined = question + " " + (rules_text or "")

        # Early rejection: check negative context before trying patterns.
        negative = _has_negative_context(combined)
        if negative:
            logger.debug(
                "macro_parse_negative_filter",
                reason=negative,
                matched_keywords=_find_keyword_matches(combined),
            )
            return ParseResult(matched=False, reject_reason=negative)

        # 1. Nonfarm payrolls (before generic indicator — more specific)
        m = _RE_PAYROLLS.search(combined)
        if m:
            raw_threshold = float(m.group("threshold"))
            unit = (m.group("unit") or "").upper()
            if unit == "K":
                threshold_unit = "K"
            elif unit == "M":
                threshold_unit = "M"
            else:
                threshold_unit = "K"  # default for payrolls
            spec = MacroContractSpec(
                category="macro",
                indicator="nonfarm_payrolls",
                threshold=raw_threshold,
                threshold_unit=threshold_unit,
                comparison=_normalize_comparison(m.group("comp")),
                release_period=(m.group("period") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "macro_parse_match",
                pattern="payrolls",
                indicator=spec.indicator,
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="macro", spec=spec, confidence=0.90)

        # 2. Fed rate decision
        m = _RE_FED.search(combined)
        if m:
            threshold_raw = m.group("threshold")
            unit_raw = m.group("unit") or ""
            spec = MacroContractSpec(
                category="macro",
                indicator="fed_rate",
                threshold=float(threshold_raw) if threshold_raw else None,
                threshold_unit="bps" if "bps" in unit_raw.lower() or "basis" in unit_raw.lower() else (
                    "%" if "%" in unit_raw or "percent" in unit_raw.lower() else ""
                ),
                comparison=_normalize_fed_comparison(m.group("comp")),
                release_period=(m.group("period") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "macro_parse_match",
                pattern="fed_rate",
                comparison=spec.comparison,
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="macro", spec=spec, confidence=0.85)

        # 3. GDP growth
        m = _RE_GDP.search(combined)
        if m:
            threshold_raw = m.group("threshold")
            spec = MacroContractSpec(
                category="macro",
                indicator="gdp",
                threshold=float(threshold_raw) if threshold_raw else None,
                threshold_unit=(m.group("unit") or "%").rstrip(),
                comparison=_normalize_comparison(m.group("comp")),
                release_period=(m.group("period") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "macro_parse_match",
                pattern="gdp",
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="macro", spec=spec, confidence=0.85)

        # 4. Generic indicator threshold (CPI, unemployment, etc.)
        m = _RE_INDICATOR.search(combined)
        if m:
            spec = MacroContractSpec(
                category="macro",
                indicator=_normalize_indicator(m.group("indicator")),
                threshold=float(m.group("threshold")),
                threshold_unit=(m.group("unit") or "%").rstrip(),
                comparison=_normalize_comparison(m.group("comp")),
                release_period=(m.group("period") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "macro_parse_match",
                pattern="indicator",
                indicator=spec.indicator,
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="macro", spec=spec, confidence=0.90)

        # No structural pattern matched — reject even if keywords present.
        matched_kws = _find_keyword_matches(combined)
        logger.debug(
            "macro_parse_keyword_only",
            matched_keywords=matched_kws,
            reject_reason="keyword_only_no_structure",
        )
        return ParseResult(
            matched=False, reject_reason="keyword_only_no_structure"
        )

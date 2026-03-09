"""Weather contract parser — temperature, precipitation, hurricane, snow."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from src.contracts.base import ContractParser, ContractSpec, ParseResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Word-boundary keyword patterns for can_parse() gate
# ---------------------------------------------------------------------------

_WEATHER_KEYWORDS: list[str] = [
    "temperature",
    "high temp",
    "low temp",
    "degrees fahrenheit",
    "degrees celsius",
    "fahrenheit",
    "celsius",
    "precipitation",
    "snowfall",
    "tropical storm",
    "heat wave",
    "frost",
    "freeze warning",
    "wind speed",
    "wind chill",
    "landfall",
]

# These require word-boundary matching to avoid substring false positives
# (e.g. "Ukraine" contains "rain", "window" contains "wind").
_WEATHER_KEYWORDS_BOUNDARY: list[str] = [
    "rain",
    "snow",
    "hurricane",
    "weather",
    "wind",
    "freeze",
]

# Pre-compiled word-boundary patterns.
_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in _WEATHER_KEYWORDS_BOUNDARY
]

# Pre-compiled plain patterns (multi-word, already safe from substring issues).
_PLAIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(re.escape(kw), re.IGNORECASE)
    for kw in _WEATHER_KEYWORDS
]

# ---------------------------------------------------------------------------
# Negative filters — sports and geopolitics blocklists
# ---------------------------------------------------------------------------

_SPORTS_NEGATIVES: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bnhl\b",
        r"\bnba\b",
        r"\bnfl\b",
        r"\bmlb\b",
        r"\bmls\b",
        r"\bfifa\b",
        r"\bstanley\s+cup\b",
        r"\bsuper\s+bowl\b",
        r"\bworld\s+cup\b",
        r"\bplayoffs?\b",
        r"\bchampionship\b",
        r"\bsemifinal\b",
        r"\bgame\s+\d",
        r"\bwin\s+the\b",
        r"\bvs\.?\b",
        r"\bgoals?\b",
        r"\bpoints?\s+per\b",
        r"\bseeding\b",
        r"\bdraft\s+pick\b",
        r"\bcoach\b",
        r"\broster\b",
        r"\btouchdown\b",
        r"\bhome\s+run\b",
        # Specific sports teams with weather-word names.
        r"\bcarolina\s+hurricanes\b",
        r"\bmiami\s+heat\b",
        r"\bokc\s+thunder\b",
        r"\bthunder\b(?=.*\b(?:nba|game|playoff|seed|win)\b)",
        r"\bqualify\s+for\b",
    ]
]

_GEOPOLITICS_NEGATIVES: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bukraine\b",
        r"\brussia\b",
        r"\bceasefire\b",
        r"\bimpeach\b",
        r"\belection\b",
        r"\bpresident\b(?!.*\btemperature\b)",
        r"\bwar\b(?!.*\b(?:temperature|rain|snow|precipitation)\b)",
        r"\bsanctions?\b",
        r"\btariff\b",
        r"\btreaty\b",
    ]
]

# ---------------------------------------------------------------------------
# WeatherContractSpec
# ---------------------------------------------------------------------------


@dataclass
class WeatherContractSpec(ContractSpec):
    """Structured fields for a weather market."""

    metric: str = ""
    location: str = ""
    threshold: float | None = None
    threshold_unit: str = ""
    comparison: str = ""
    date_start: str = ""
    date_end: str = ""
    date_description: str = ""
    nws_station_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:  # type: ignore[override]
        return {
            "category": self.category,
            "metric": self.metric,
            "location": self.location,
            "threshold": self.threshold,
            "threshold_unit": self.threshold_unit,
            "comparison": self.comparison,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "date_description": self.date_description,
            "nws_station_ids": self.nws_station_ids,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_comparison(raw: str) -> str:
    """Collapse various phrasings into a canonical comparison operator."""
    raw = raw.strip().lower()
    if raw in ("exceed", "exceeds", "above", "over", "more than", "greater than", "higher than"):
        return "above"
    if raw in ("below", "under", "less than", "lower than", "fall below", "drop below"):
        return "below"
    if raw in ("at least", "at_least", ">="):
        return "at_least"
    if raw in ("at most", "at_most", "<=", "no more than"):
        return "at_most"
    return raw


def _normalize_metric(raw: str) -> str:
    """Map free-text metric descriptions to canonical keys."""
    raw = raw.strip().lower()
    if "high" in raw and "temp" in raw:
        return "temperature_high"
    if "low" in raw and "temp" in raw:
        return "temperature_low"
    if raw in ("temperature", "temp"):
        return "temperature_high"
    if raw in ("precipitation", "rain", "rainfall"):
        return "precipitation"
    if raw in ("snowfall", "snow accumulation"):
        return "snowfall"
    if raw in ("hurricane", "tropical storm", "cyclone"):
        return "hurricane"
    if raw in ("snow",):
        return "snowfall"
    if raw in ("snow occurrence",):
        return "snow_occurrence"
    return raw


def _has_negative_context(text: str) -> str | None:
    """Check for sports or geopolitics context that blocks weather.

    Returns the reject reason string, or ``None`` if clean.
    """
    for pat in _SPORTS_NEGATIVES:
        if pat.search(text):
            return f"sports_context: {pat.pattern}"
    for pat in _GEOPOLITICS_NEGATIVES:
        if pat.search(text):
            return f"geopolitics_context: {pat.pattern}"
    return None


def _find_keyword_matches(text: str) -> list[str]:
    """Return list of weather keywords found (for diagnostics)."""
    matched: list[str] = []
    for kw, pat in zip(_WEATHER_KEYWORDS, _PLAIN_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    for kw, pat in zip(_WEATHER_KEYWORDS_BOUNDARY, _BOUNDARY_PATTERNS):
        if pat.search(text):
            matched.append(kw)
    return matched


# ---------------------------------------------------------------------------
# Regex patterns (tried in priority order)
# ---------------------------------------------------------------------------

# 1. Temperature threshold
#    "Will the high temperature in Dallas exceed 100°F on June 15?"
_RE_TEMP = re.compile(
    r"(?:will\s+)?(?:the\s+)?"
    r"(?P<metric>(?:high|low)?\s*temp(?:erature)?)\s+"
    r"(?:in|for|at)\s+(?P<location>[A-Za-z][A-Za-z ,.'()-]+?)\s+"
    r"(?P<comp>exceed|exceeds?|be\s+above|be\s+below|fall\s+below|drop\s+below|be\s+over|be\s+under|be\s+at\s+least|be\s+at\s+most)\s+"
    r"(?P<threshold>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[FCfc])",
    re.IGNORECASE,
)

# 2. Precipitation / snowfall amount
#    "Will New York receive more than 2 inches of rain in January 2026?"
_RE_PRECIP = re.compile(
    r"(?:will\s+)?(?P<location>[A-Za-z][A-Za-z ,.'()-]+?)\s+"
    r"(?:receive|get|see|have|record)\s+"
    r"(?P<comp>more than|less than|at least|at most|over|under|exceed(?:ing)?)\s+"
    r"(?P<threshold>\d+(?:\.\d+)?)\s*(?P<unit>inches?|in\.?|cm|mm)\s+"
    r"of\s+(?P<metric>rain|snow|precipitation|snowfall|rainfall)"
    r"(?:\s+(?:in|on|during|for)\s+(?P<date>.+?))?",
    re.IGNORECASE,
)

# 3. Hurricane — requires word boundary to avoid matching "Hurricanes" (sports)
#    "Will there be a Category 3+ hurricane in the Atlantic in 2026?"
_RE_HURRICANE = re.compile(
    r"(?:will\s+there\s+be\s+)?(?:a\s+)?"
    r"(?:category\s+(?P<threshold>\d)\+?\s+)?"
    r"(?P<metric>hurricane|tropical\s+storm)\b"
    r"(?:\s+(?:in|hitting|making\s+landfall\s+in|affect(?:ing)?)\s+"
    r"(?P<location>[A-Za-z][A-Za-z ,.'()-]+?)"
    r"(?=\s+(?:in|during|by)\s|\s*[?.]|\s*$))?"
    r"(?:\s+(?:in|during|by)\s+(?P<date>.+))?",
    re.IGNORECASE,
)

# 4. Snow occurrence
#    "Will it snow in Chicago on Christmas Day?"
_RE_SNOW_OCCUR = re.compile(
    r"will\s+it\s+snow\s+"
    r"(?:in|at)\s+(?P<location>[A-Za-z][A-Za-z ,.'()-]+?)"
    r"(?=\s+(?:on|in|during|before|after|by)\s|\s*[?.]|\s*$)"
    r"(?:\s+(?:on|in|during|before|after|by)\s+(?P<date>.+))?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# WeatherParser
# ---------------------------------------------------------------------------


class WeatherParser(ContractParser):
    """Parse weather-related prediction markets."""

    @property
    def name(self) -> str:
        return "weather_v1"

    @property
    def category(self) -> str:
        return "weather"

    # -- keyword gate --------------------------------------------------------

    def can_parse(self, question: str, rules_text: str | None) -> bool:
        """Check if text contains weather keywords (word-boundary safe)."""
        combined = question + " " + (rules_text or "")

        # Check word-boundary keywords first (rain, snow, hurricane, etc.).
        for pat in _BOUNDARY_PATTERNS:
            if pat.search(combined):
                return True

        # Check multi-word keywords (temperature, tropical storm, etc.).
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
                "weather_parse_negative_filter",
                reason=negative,
                matched_keywords=_find_keyword_matches(combined),
            )
            return ParseResult(matched=False, reject_reason=negative)

        # 1. Temperature
        m = _RE_TEMP.search(combined)
        if m:
            spec = WeatherContractSpec(
                category="weather",
                metric=_normalize_metric(m.group("metric")),
                location=m.group("location").strip().rstrip(","),
                threshold=float(m.group("threshold")),
                threshold_unit=m.group("unit").upper(),
                comparison=_normalize_comparison(m.group("comp")),
            )
            # Try to grab a date from surrounding text.
            spec.date_description = _extract_date_tail(combined, m.end())
            logger.debug(
                "weather_parse_match",
                pattern="temperature",
                location=spec.location,
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="weather", spec=spec, confidence=0.90)

        # 2. Precipitation / snowfall amount
        m = _RE_PRECIP.search(combined)
        if m:
            spec = WeatherContractSpec(
                category="weather",
                metric=_normalize_metric(m.group("metric")),
                location=m.group("location").strip().rstrip(","),
                threshold=float(m.group("threshold")),
                threshold_unit="inches",
                comparison=_normalize_comparison(m.group("comp")),
                date_description=(m.group("date") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "weather_parse_match",
                pattern="precipitation",
                location=spec.location,
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="weather", spec=spec, confidence=0.85)

        # 3. Hurricane
        m = _RE_HURRICANE.search(combined)
        if m:
            threshold_raw = m.group("threshold")
            spec = WeatherContractSpec(
                category="weather",
                metric="hurricane",
                location=(m.group("location") or "").strip().rstrip("?,. "),
                threshold=float(threshold_raw) if threshold_raw else None,
                threshold_unit="category",
                comparison="at_least" if threshold_raw else "",
                date_description=(m.group("date") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "weather_parse_match",
                pattern="hurricane",
                location=spec.location,
                threshold=spec.threshold,
            )
            return ParseResult(matched=True, category="weather", spec=spec, confidence=0.85)

        # 4. Snow occurrence
        m = _RE_SNOW_OCCUR.search(combined)
        if m:
            spec = WeatherContractSpec(
                category="weather",
                metric="snow_occurrence",
                location=m.group("location").strip().rstrip(","),
                date_description=(m.group("date") or "").strip().rstrip("?. "),
            )
            logger.debug(
                "weather_parse_match",
                pattern="snow_occurrence",
                location=spec.location,
            )
            return ParseResult(matched=True, category="weather", spec=spec, confidence=0.80)

        # No structural pattern matched — reject even if keywords present.
        # The generic fallback was the source of most false positives.
        matched_kws = _find_keyword_matches(combined)
        logger.debug(
            "weather_parse_keyword_only",
            matched_keywords=matched_kws,
            reject_reason="keyword_only_no_structure",
        )
        return ParseResult(
            matched=False, reject_reason="keyword_only_no_structure"
        )


def _extract_date_tail(text: str, start: int) -> str:
    """Try to grab a date fragment after the regex match end."""
    tail = text[start:].strip()
    m = re.match(
        r"(?:on|in|during|for|by)\s+(.+?)(?:\?|$)",
        tail,
        re.IGNORECASE,
    )
    return m.group(1).strip().rstrip("?. ") if m else ""

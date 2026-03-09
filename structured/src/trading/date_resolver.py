"""Resolve date descriptions from contract specs into datetime ranges.

Handles patterns like:
- "June 15, 2026", "January 2026"
- "Christmas Day", "New Year's Day", "Thanksgiving"
- "Q1 2026", "Q2 2026"
- "2026 Atlantic hurricane season"
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timezone

from dateutil import parser as dateutil_parser

# ---------------------------------------------------------------------------
# Named date patterns
# ---------------------------------------------------------------------------

_NAMED_DATES: dict[str, tuple[int, int]] = {
    "new year's day": (1, 1),
    "new years day": (1, 1),
    "new year": (1, 1),
    "valentine's day": (2, 14),
    "valentines day": (2, 14),
    "independence day": (7, 4),
    "july 4th": (7, 4),
    "fourth of july": (7, 4),
    "halloween": (10, 31),
    "christmas day": (12, 25),
    "christmas eve": (12, 24),
    "christmas": (12, 25),
    "new year's eve": (12, 31),
    "new years eve": (12, 31),
}

# Quarter definitions: (start_month, end_month)
_QUARTERS: dict[str, tuple[int, int]] = {
    "q1": (1, 3),
    "q2": (4, 6),
    "q3": (7, 9),
    "q4": (10, 12),
}

# Hurricane season
_RE_HURRICANE_SEASON = re.compile(
    r"(\d{4})\s+(?:atlantic\s+)?hurricane\s+season", re.IGNORECASE
)

# Quarter: "Q1 2026" or "2026 Q1"
_RE_QUARTER = re.compile(
    r"(?:([Qq][1-4])\s+(\d{4}))|(?:(\d{4})\s+([Qq][1-4]))", re.IGNORECASE
)

# Month year: "January 2026"
_RE_MONTH_YEAR = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _last_day_of_month(year: int, month: int) -> int:
    """Return the last day of the given month."""
    import calendar
    return calendar.monthrange(year, month)[1]


def resolve_date_range(
    date_description: str,
    reference_date: date | None = None,
) -> tuple[datetime, datetime] | None:
    """Resolve a date description to a ``(start, end)`` UTC datetime range.

    Returns ``None`` if the description cannot be parsed.
    """
    if not date_description or not date_description.strip():
        return None

    desc = date_description.strip()
    ref = reference_date or date.today()

    # 1. Named dates (Christmas Day, etc.)
    desc_lower = desc.lower()
    for name, (month, day) in _NAMED_DATES.items():
        if name in desc_lower:
            # Extract year if present, else use reference year.
            year_match = re.search(r"\d{4}", desc)
            year = int(year_match.group()) if year_match else ref.year
            start = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
            return (start, end)

    # 2. Hurricane season.
    m = _RE_HURRICANE_SEASON.search(desc)
    if m:
        year = int(m.group(1))
        start = datetime(year, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(year, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
        return (start, end)

    # 3. Quarter.
    m = _RE_QUARTER.search(desc)
    if m:
        quarter = (m.group(1) or m.group(4)).lower()
        year = int(m.group(2) or m.group(3))
        start_month, end_month = _QUARTERS[quarter]
        last_day = _last_day_of_month(year, end_month)
        start = datetime(year, start_month, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        return (start, end)

    # 4. Month + year: "January 2026".
    m = _RE_MONTH_YEAR.search(desc)
    if m:
        month = _MONTH_NAMES[m.group(1).lower()]
        year = int(m.group(2))
        last_day = _last_day_of_month(year, month)
        start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        return (start, end)

    # 5. Generic dateutil parse (handles "June 15, 2026", "2026-06-15", etc.)
    try:
        parsed = dateutil_parser.parse(desc, default=datetime(ref.year, 1, 1))
        start = datetime(
            parsed.year, parsed.month, parsed.day, 0, 0, 0, tzinfo=timezone.utc
        )
        end = datetime(
            parsed.year, parsed.month, parsed.day, 23, 59, 59, tzinfo=timezone.utc
        )
        return (start, end)
    except (ValueError, OverflowError):
        pass

    return None

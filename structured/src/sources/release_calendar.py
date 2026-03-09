"""Economic release calendar — resolves indicator + period to release dates.

Covers BLS (CPI, payrolls, PPI), BEA/FRED (GDP, PCE), and FOMC meetings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Static FOMC meeting dates (2025-2026)
# ---------------------------------------------------------------------------

_FOMC_DATES: list[datetime] = [
    # 2025
    datetime(2025, 1, 29, tzinfo=timezone.utc),
    datetime(2025, 3, 19, tzinfo=timezone.utc),
    datetime(2025, 5, 7, tzinfo=timezone.utc),
    datetime(2025, 6, 18, tzinfo=timezone.utc),
    datetime(2025, 7, 30, tzinfo=timezone.utc),
    datetime(2025, 9, 17, tzinfo=timezone.utc),
    datetime(2025, 10, 29, tzinfo=timezone.utc),
    datetime(2025, 12, 17, tzinfo=timezone.utc),
    # 2026
    datetime(2026, 1, 28, tzinfo=timezone.utc),
    datetime(2026, 3, 18, tzinfo=timezone.utc),
    datetime(2026, 4, 29, tzinfo=timezone.utc),
    datetime(2026, 6, 17, tzinfo=timezone.utc),
    datetime(2026, 7, 29, tzinfo=timezone.utc),
    datetime(2026, 9, 16, tzinfo=timezone.utc),
    datetime(2026, 11, 4, tzinfo=timezone.utc),
    datetime(2026, 12, 16, tzinfo=timezone.utc),
]

# ---------------------------------------------------------------------------
# Release day heuristics per indicator
# ---------------------------------------------------------------------------

# BLS CPI: typically released on the 10th-13th of the month for the prior month.
# BLS Jobs (NFP): first Friday of the month.
# BLS PPI: typically mid-month.
# BEA GDP: ~28th of month (advance estimate month+1 after quarter end).
# BEA PCE: last weekday of month for prior month.

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_QUARTER_MAP: dict[str, tuple[int, int]] = {
    "q1": (1, 4),  # Q1 data released ~late April
    "q2": (4, 7),  # Q2 data released ~late July
    "q3": (7, 10),  # Q3 data released ~late October
    "q4": (10, 1),  # Q4 data released ~late January next year
}


def _parse_period(release_period: str) -> tuple[int | None, int | None]:
    """Parse a release_period string into (month, year).

    Returns (None, None) if parsing fails.
    """
    parts = release_period.strip().lower().split()
    month = None
    year = None

    for part in parts:
        if part in _MONTH_NAMES:
            month = _MONTH_NAMES[part]
        elif part.isdigit() and len(part) == 4:
            year = int(part)
        elif part.startswith("q") and len(part) == 2 and part[1].isdigit():
            quarter_info = _QUARTER_MAP.get(part)
            if quarter_info:
                month = quarter_info[1]

    return month, year


def _first_friday(year: int, month: int) -> datetime:
    """Return the first Friday of the given month."""
    d = datetime(year, month, 1, 8, 30, tzinfo=timezone.utc)
    # weekday(): Monday=0, Friday=4
    days_until_friday = (4 - d.weekday()) % 7
    return d + timedelta(days=days_until_friday)


def resolve_release_date(
    indicator: str,
    release_period: str,
    reference_date: datetime | None = None,
) -> datetime | None:
    """Resolve an indicator + release period to an approximate release date.

    Returns ``None`` if the period cannot be parsed.
    """
    month, year = _parse_period(release_period)
    if month is None:
        return None
    if year is None:
        now = reference_date or datetime.now(timezone.utc)
        year = now.year
        if month < now.month:
            year += 1

    indicator = indicator.lower()

    if indicator == "fed_rate":
        return _next_fomc_for_period(month, year)

    if indicator in ("cpi", "core_cpi"):
        # CPI for month M is released ~13th of month M+1.
        rel_month = month + 1
        rel_year = year
        if rel_month > 12:
            rel_month = 1
            rel_year += 1
        return datetime(rel_year, rel_month, 13, 8, 30, tzinfo=timezone.utc)

    if indicator in ("nonfarm_payrolls",):
        # Jobs report for month M is released first Friday of M+1.
        rel_month = month + 1
        rel_year = year
        if rel_month > 12:
            rel_month = 1
            rel_year += 1
        return _first_friday(rel_year, rel_month)

    if indicator in ("ppi",):
        # PPI for month M is released ~15th of M+1.
        rel_month = month + 1
        rel_year = year
        if rel_month > 12:
            rel_month = 1
            rel_year += 1
        return datetime(rel_year, rel_month, 15, 8, 30, tzinfo=timezone.utc)

    if indicator in ("gdp",):
        # GDP advance estimate: ~28th of month after quarter end.
        return datetime(year, month, 28, 8, 30, tzinfo=timezone.utc)

    if indicator in ("pce", "core_pce"):
        # PCE for month M released last weekday of M+1.
        rel_month = month + 1
        rel_year = year
        if rel_month > 12:
            rel_month = 1
            rel_year += 1
        # Last day of rel_month.
        if rel_month == 12:
            last_day = datetime(rel_year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            last_day = datetime(rel_year, rel_month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        # Back up to weekday.
        while last_day.weekday() >= 5:
            last_day -= timedelta(days=1)
        return last_day.replace(hour=8, minute=30)

    if indicator in ("unemployment", "retail_sales", "housing_starts"):
        # Generic: ~15th of month after reference.
        rel_month = month + 1
        rel_year = year
        if rel_month > 12:
            rel_month = 1
            rel_year += 1
        return datetime(rel_year, rel_month, 15, 8, 30, tzinfo=timezone.utc)

    return None


def _next_fomc_for_period(month: int, year: int) -> datetime | None:
    """Find the FOMC meeting closest to the given month/year."""
    target = datetime(year, month, 15, tzinfo=timezone.utc)
    best: datetime | None = None
    best_dist = timedelta(days=9999)
    for dt in _FOMC_DATES:
        dist = abs(dt - target)
        if dist < best_dist:
            best = dt
            best_dist = dist
    return best


def next_release(
    indicator: str,
    after: datetime | None = None,
) -> datetime | None:
    """Find the next release date for an indicator after a given time."""
    now = after or datetime.now(timezone.utc)

    if indicator.lower() == "fed_rate":
        for dt in _FOMC_DATES:
            if dt > now:
                return dt
        return None

    # For other indicators, iterate forward month by month.
    for offset in range(0, 14):
        m = now.month + offset
        y = now.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        period_name = datetime(y, m, 1).strftime("%B %Y")
        release = resolve_release_date(indicator, period_name)
        if release is not None and release > now:
            return release
    return None


def is_near_release(
    indicator: str,
    release_period: str,
    hours_threshold: float = 24.0,
    now: datetime | None = None,
) -> bool:
    """Return True if the release date is within ``hours_threshold`` hours."""
    now = now or datetime.now(timezone.utc)
    release = resolve_release_date(indicator, release_period)
    if release is None:
        return False
    return abs((release - now).total_seconds()) / 3600 <= hours_threshold

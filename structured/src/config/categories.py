"""Category definitions and portfolio isolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    """Supported market categories."""

    WEATHER = "weather"
    MACRO = "macro"


@dataclass
class CategoryPortfolio:
    """In-memory portfolio state for a single category."""

    category: Category
    bankroll_eur: float
    exposure_eur: float = 0.0
    realized_pnl_eur: float = 0.0
    unrealized_pnl_eur: float = 0.0
    daily_realized_pnl_eur: float = 0.0

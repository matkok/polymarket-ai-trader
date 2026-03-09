"""Base types for pricing engines."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PriceEstimate:
    """Output of a pricing engine computation."""

    p_yes: float
    confidence: float
    source_confidence: float = 1.0
    model_details: dict[str, Any] = field(default_factory=dict)

    def to_engine_price_dict(
        self,
        market_id: str,
        category: str,
        engine_version: str,
        ts_utc: Any,
        p_market: float | None = None,
    ) -> dict[str, Any]:
        """Convert to a dict suitable for ``repo.add_engine_price()``."""
        edge = None
        if p_market is not None:
            edge = abs(self.p_yes - p_market)
        return {
            "market_id": market_id,
            "category": category,
            "ts_utc": ts_utc,
            "engine_version": engine_version,
            "p_yes": self.p_yes,
            "confidence": self.confidence,
            "source_confidence": self.source_confidence,
            "edge_before_costs": edge,
            "price_json": self.model_details,
        }


class PricingEngine(abc.ABC):
    """Abstract base class for deterministic pricing engines."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Engine identifier, e.g. ``'weather_v1'``."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Semantic version string."""

    @abc.abstractmethod
    def compute(self, spec: Any, observation: Any) -> PriceEstimate:
        """Compute a price estimate from a contract spec and observation data."""

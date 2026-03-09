"""Base types for data source adapters."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FetchResult:
    """Result of a data source fetch."""

    source_name: str
    source_key: str
    ts_source: datetime
    raw_json: dict[str, Any]
    normalized_json: dict[str, Any]
    quality_score: float = 1.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_observation_dict(self, category: str) -> dict[str, Any]:
        """Convert to a dict suitable for ``repo.add_source_observation()``."""
        return {
            "category": category,
            "source_name": self.source_name,
            "source_key": self.source_key,
            "ts_source": self.ts_source,
            "raw_json": self.raw_json,
            "normalized_json": self.normalized_json,
            "quality_score": self.quality_score,
        }


class SourceAdapter(abc.ABC):
    """Abstract base class for data source adapters."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique adapter identifier, e.g. ``'nws'``."""

    @abc.abstractmethod
    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch data for the given contract spec."""

    async def health_check(self) -> bool:
        """Return *True* if the source is reachable."""
        return True

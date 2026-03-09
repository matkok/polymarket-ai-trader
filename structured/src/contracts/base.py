"""Base types for contract parsing: ABC, ContractSpec, ParseResult."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContractSpec:
    """Structured specification extracted from a market's question/rules."""

    category: str
    raw_fields: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialise to dict for storage in ``contract_spec_json``."""
        return {"category": self.category, **self.raw_fields}


@dataclass
class ParseResult:
    """Outcome of attempting to parse a market."""

    matched: bool
    category: str = ""
    spec: ContractSpec | None = None
    confidence: float = 0.0
    reject_reason: str = ""


class ContractParser(abc.ABC):
    """Abstract base class for category-specific market parsers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique parser identifier, e.g. ``'weather_v1'``."""

    @property
    @abc.abstractmethod
    def category(self) -> str:
        """Category slug, e.g. ``'weather'``."""

    @abc.abstractmethod
    def can_parse(self, question: str, rules_text: str | None) -> bool:
        """Return *True* if this parser might handle the market."""

    @abc.abstractmethod
    def parse(self, question: str, rules_text: str | None) -> ParseResult:
        """Attempt to extract a structured :class:`ContractSpec`."""

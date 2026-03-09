"""Contract parsing — market classification and spec extraction."""

from __future__ import annotations

from src.contracts.base import ContractParser, ContractSpec, ParseResult
from src.contracts.macro import MacroContractSpec, MacroParser
from src.contracts.registry import ParserRegistry, classify_markets_batch
from src.contracts.weather import WeatherContractSpec, WeatherParser

__all__ = [
    "ContractParser",
    "ContractSpec",
    "ParseResult",
    "WeatherContractSpec",
    "WeatherParser",
    "MacroContractSpec",
    "MacroParser",
    "ParserRegistry",
    "classify_markets_batch",
]

"""Exchange router — routes spec.exchange to the correct adapter."""

from __future__ import annotations

from typing import Any

import structlog

from src.sources.base import FetchResult, SourceAdapter
from src.sources.binance import BinanceAdapter
from src.sources.coinbase import CoinbaseAdapter
from src.sources.kraken import KrakenAdapter

logger = structlog.get_logger(__name__)


class ExchangeRouter(SourceAdapter):
    """Route crypto specs to the correct exchange adapter.

    Falls back to Coinbase if no exchange specified.
    """

    def __init__(
        self,
        coinbase: CoinbaseAdapter | None = None,
        binance: BinanceAdapter | None = None,
        kraken: KrakenAdapter | None = None,
    ) -> None:
        self._coinbase = coinbase or CoinbaseAdapter()
        self._binance = binance or BinanceAdapter()
        self._kraken = kraken or KrakenAdapter()

        self._adapters: dict[str, SourceAdapter] = {
            "coinbase": self._coinbase,
            "binance": self._binance,
            "kraken": self._kraken,
        }

    @property
    def name(self) -> str:
        return "exchange_router"

    def get_adapter(self, exchange: str) -> SourceAdapter:
        """Return the adapter for the given exchange name.

        Falls back to Coinbase for unknown or empty exchange.
        """
        return self._adapters.get(exchange.lower(), self._coinbase)

    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch from the exchange specified in the spec."""
        exchange = getattr(spec, "exchange", "") or ""
        adapter = self.get_adapter(exchange)
        logger.debug(
            "exchange_router_dispatch",
            exchange=exchange,
            adapter=adapter.name,
            asset=getattr(spec, "asset", ""),
        )
        return await adapter.fetch(spec)

    async def health_check(self) -> bool:
        """Check if at least one exchange is reachable."""
        for adapter in self._adapters.values():
            if await adapter.health_check():
                return True
        return False

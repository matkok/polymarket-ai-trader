"""Async client for Polymarket Gamma API (market discovery)."""

from __future__ import annotations

import httpx
import structlog

from src.polymarket.schemas import GammaEvent, GammaMarket


class GammaClient:
    """Async client for Polymarket Gamma API (market discovery).

    The Gamma API provides market metadata: questions, categories,
    outcome prices, liquidity, and volume.
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.logger = structlog.get_logger(__name__)

    async def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
    ) -> list[GammaMarket]:
        """Fetch markets from Gamma API with pagination."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            return [GammaMarket.model_validate(m) for m in data]

    async def get_all_active_markets(self, max_pages: int = 300) -> list[GammaMarket]:
        """Paginate through active markets up to *max_pages*."""
        all_markets: list[GammaMarket] = []
        for page in range(max_pages):
            markets = await self.get_markets(limit=100, offset=page * 100)
            if not markets:
                break
            all_markets.extend(markets)
            self.logger.info("gamma_markets_fetched", page=page, count=len(markets))
        return all_markets

    async def get_events(self, limit: int = 50) -> list[GammaEvent]:
        """Fetch events from Gamma API."""
        params = {"limit": limit}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}/events", params=params)
            resp.raise_for_status()
            data = resp.json()
            return [GammaEvent.model_validate(e) for e in data]

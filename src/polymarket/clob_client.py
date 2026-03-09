"""Async client for Polymarket CLOB REST API (prices, order books)."""

from __future__ import annotations

import httpx
import structlog

from src.polymarket.schemas import CLOBOrderBook


class CLOBClient:
    """Async client for Polymarket CLOB REST API (prices, order books).

    The CLOB API provides real-time order book snapshots, prices, and
    midpoints for individual tokens.
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.logger = structlog.get_logger(__name__)

    async def get_order_book(self, token_id: str) -> CLOBOrderBook:
        """Fetch order book for a specific token."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self.base_url}/book", params={"token_id": token_id}
            )
            resp.raise_for_status()
            return CLOBOrderBook.model_validate(resp.json())

    async def get_price(self, token_id: str) -> float | None:
        """Get latest price for a token.  Returns ``None`` on error."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self.base_url}/price", params={"token_id": token_id}
                )
                resp.raise_for_status()
                data = resp.json()
                return float(data.get("price", 0))
        except Exception:
            self.logger.warning("clob_price_error", token_id=token_id)
            return None

    async def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token.  Returns ``None`` on error."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self.base_url}/midpoint", params={"token_id": token_id}
                )
                resp.raise_for_status()
                data = resp.json()
                return float(data.get("mid", 0))
        except Exception:
            self.logger.warning("clob_midpoint_error", token_id=token_id)
            return None

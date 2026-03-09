"""Coinbase Exchange API adapter — public ticker data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.sources.base import FetchResult, SourceAdapter

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.exchange.coinbase.com"

# Pair name mapping: asset → Coinbase product ID.
_PAIR_MAP: dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "XRP": "XRP-USD",
}


class CoinbaseAdapter(SourceAdapter):
    """Fetch public ticker data from the Coinbase Exchange API."""

    @property
    def name(self) -> str:
        return "coinbase"

    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch latest ticker for a crypto asset."""
        now = datetime.now(timezone.utc)
        asset = getattr(spec, "asset", None)
        if asset is None:
            return FetchResult(
                source_name=self.name, source_key="",
                ts_source=now, raw_json={}, normalized_json={},
                error="invalid_spec",
            )

        pair = _PAIR_MAP.get(asset.upper())
        if pair is None:
            return FetchResult(
                source_name=self.name, source_key=asset,
                ts_source=now, raw_json={}, normalized_json={},
                error=f"unsupported_asset: {asset}",
            )

        url = f"{_BASE_URL}/products/{pair}/ticker"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            logger.exception("coinbase_fetch_error", asset=asset)
            return FetchResult(
                source_name=self.name, source_key=pair,
                ts_source=now, raw_json={}, normalized_json={},
                error=f"api_error: {exc}",
            )

        try:
            price = float(raw["price"])
            bid = float(raw.get("bid", 0))
            ask = float(raw.get("ask", 0))
            volume = float(raw.get("volume", 0))
        except (KeyError, ValueError, TypeError) as exc:
            return FetchResult(
                source_name=self.name, source_key=pair,
                ts_source=now, raw_json=raw, normalized_json={},
                error=f"parse_error: {exc}",
            )

        normalized = {
            "asset": asset.upper(),
            "price": price,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "exchange": "coinbase",
            "pair": pair,
        }

        return FetchResult(
            source_name=self.name, source_key=pair,
            ts_source=now, raw_json=raw, normalized_json=normalized,
            quality_score=1.0,
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_BASE_URL}/products")
                return resp.status_code == 200
        except Exception:
            return False

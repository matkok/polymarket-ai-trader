"""Binance Spot API adapter — public ticker data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.sources.base import FetchResult, SourceAdapter

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.binance.com/api/v3"

_PAIR_MAP: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT",
    "XRP": "XRPUSDT",
}


class BinanceAdapter(SourceAdapter):
    """Fetch public ticker data from the Binance Spot API."""

    @property
    def name(self) -> str:
        return "binance"

    async def fetch(self, spec: Any) -> FetchResult:
        now = datetime.now(timezone.utc)
        asset = getattr(spec, "asset", None)
        if asset is None:
            return FetchResult(
                source_name=self.name, source_key="",
                ts_source=now, raw_json={}, normalized_json={},
                error="invalid_spec",
            )

        symbol = _PAIR_MAP.get(asset.upper())
        if symbol is None:
            return FetchResult(
                source_name=self.name, source_key=asset,
                ts_source=now, raw_json={}, normalized_json={},
                error=f"unsupported_asset: {asset}",
            )

        url = f"{_BASE_URL}/ticker/price"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params={"symbol": symbol})
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            logger.exception("binance_fetch_error", asset=asset)
            return FetchResult(
                source_name=self.name, source_key=symbol,
                ts_source=now, raw_json={}, normalized_json={},
                error=f"api_error: {exc}",
            )

        try:
            price = float(raw["price"])
        except (KeyError, ValueError, TypeError) as exc:
            return FetchResult(
                source_name=self.name, source_key=symbol,
                ts_source=now, raw_json=raw, normalized_json={},
                error=f"parse_error: {exc}",
            )

        normalized = {
            "asset": asset.upper(),
            "price": price,
            "exchange": "binance",
            "pair": symbol,
        }

        return FetchResult(
            source_name=self.name, source_key=symbol,
            ts_source=now, raw_json=raw, normalized_json=normalized,
            quality_score=1.0,
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_BASE_URL}/ping")
                return resp.status_code == 200
        except Exception:
            return False

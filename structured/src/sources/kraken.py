"""Kraken REST API adapter — public ticker data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.sources.base import FetchResult, SourceAdapter

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.kraken.com/0/public"

_PAIR_MAP: dict[str, str] = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "DOGE": "DOGEUSD",
    "ADA": "ADAUSD",
    "XRP": "XRPUSD",
}


class KrakenAdapter(SourceAdapter):
    """Fetch public ticker data from the Kraken REST API."""

    @property
    def name(self) -> str:
        return "kraken"

    async def fetch(self, spec: Any) -> FetchResult:
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

        url = f"{_BASE_URL}/Ticker"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params={"pair": pair})
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            logger.exception("kraken_fetch_error", asset=asset)
            return FetchResult(
                source_name=self.name, source_key=pair,
                ts_source=now, raw_json={}, normalized_json={},
                error=f"api_error: {exc}",
            )

        errors = raw.get("error", [])
        if errors:
            return FetchResult(
                source_name=self.name, source_key=pair,
                ts_source=now, raw_json=raw, normalized_json={},
                error=f"kraken_error: {errors}",
            )

        result_data = raw.get("result", {})
        # Kraken uses varying key names (e.g. "XXBTZUSD" for BTC).
        ticker = None
        for key, val in result_data.items():
            ticker = val
            break

        if ticker is None:
            return FetchResult(
                source_name=self.name, source_key=pair,
                ts_source=now, raw_json=raw, normalized_json={},
                error="no_ticker_data",
            )

        try:
            # "c" = last trade [price, lot_volume].
            price = float(ticker["c"][0])
            bid = float(ticker["b"][0])
            ask = float(ticker["a"][0])
        except (KeyError, ValueError, TypeError, IndexError) as exc:
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
            "exchange": "kraken",
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
                resp = await client.get(f"{_BASE_URL}/Time")
                return resp.status_code == 200
        except Exception:
            return False

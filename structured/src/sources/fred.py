"""FRED API adapter — GDP, PCE, Fed Funds Rate, historical series."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.contracts.macro import MacroContractSpec
from src.sources.base import FetchResult, SourceAdapter

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Series ID mapping
# ---------------------------------------------------------------------------

_SERIES_MAP: dict[str, str] = {
    "gdp": "GDPC1",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "fed_rate": "FEDFUNDS",
    # Fallback aliases for BLS indicators (FRED mirrors them).
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "ppi": "PPIACO",
}

_API_URL = "https://api.stlouisfed.org/fred/series/observations"


class FREDAdapter(SourceAdapter):
    """Fetch economic data from the Federal Reserve Economic Data API."""

    def __init__(self, api_key: str = "", lookback_months: int = 24) -> None:
        self._api_key = api_key
        self._lookback_months = lookback_months

    @property
    def name(self) -> str:
        return "fred"

    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch observations for a macro contract spec from FRED."""
        now = datetime.now(timezone.utc)

        if not isinstance(spec, MacroContractSpec):
            return FetchResult(
                source_name=self.name,
                source_key="",
                ts_source=now,
                raw_json={},
                normalized_json={},
                error="invalid_spec_type",
            )

        series_id = _SERIES_MAP.get(spec.indicator)
        if series_id is None:
            return FetchResult(
                source_name=self.name,
                source_key=spec.indicator,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error=f"unsupported_indicator: {spec.indicator}",
            )

        if not self._api_key:
            return FetchResult(
                source_name=self.name,
                source_key=series_id,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error="no_api_key",
            )

        # Calculate observation window.
        lookback_days = self._lookback_months * 31
        obs_start = now - __import__("datetime").timedelta(days=lookback_days)

        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": obs_start.strftime("%Y-%m-%d"),
            "sort_order": "desc",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(_API_URL, params=params)
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            logger.exception("fred_fetch_error", indicator=spec.indicator)
            return FetchResult(
                source_name=self.name,
                source_key=series_id,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error=f"api_error: {exc}",
            )

        normalized = self._parse_response(raw, spec.indicator, series_id)
        if "error" in normalized:
            return FetchResult(
                source_name=self.name,
                source_key=series_id,
                ts_source=now,
                raw_json=raw,
                normalized_json=normalized,
                error=normalized["error"],
            )

        return FetchResult(
            source_name=self.name,
            source_key=series_id,
            ts_source=now,
            raw_json=raw,
            normalized_json=normalized,
            quality_score=1.0,
        )

    @staticmethod
    def _parse_response(
        raw: dict[str, Any],
        indicator: str,
        series_id: str,
    ) -> dict[str, Any]:
        """Parse FRED API response into normalized dict."""
        observations = raw.get("observations", [])
        if not observations:
            return {"error": "no_observations"}

        # Filter out missing values (".").
        valid_obs: list[dict[str, Any]] = []
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                try:
                    valid_obs.append({
                        "date": obs["date"],
                        "value": float(val),
                    })
                except (ValueError, KeyError):
                    continue

        if not valid_obs:
            return {"error": "no_valid_observations"}

        # Response is sorted desc — first is latest.
        latest = valid_obs[0]
        previous = valid_obs[1] if len(valid_obs) > 1 else None

        result: dict[str, Any] = {
            "indicator": indicator,
            "series_id": series_id,
            "latest_value": latest["value"],
            "latest_date": latest["date"],
        }

        if previous is not None:
            result["previous_value"] = previous["value"]
            result["previous_date"] = previous["date"]
            change = latest["value"] - previous["value"]
            result["change"] = change
            if previous["value"] != 0:
                result["change_pct"] = change / previous["value"] * 100.0

        # Historical values for distribution fitting.
        result["history"] = valid_obs

        return result

    async def health_check(self) -> bool:
        """Check if FRED API is reachable."""
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.stlouisfed.org/fred/series",
                    params={
                        "series_id": "GNPCA",
                        "api_key": self._api_key,
                        "file_type": "json",
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False

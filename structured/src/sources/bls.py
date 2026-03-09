"""BLS Public Data API v2 adapter — CPI, unemployment, payrolls, PPI."""

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
    "cpi": "CUSR0000SA0",
    "core_cpi": "CUSR0000SA0L1E",
    "unemployment": "LNS14000000",
    "nonfarm_payrolls": "CES0000000001",
    "ppi": "WPSFD4",
}

_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


class BLSAdapter(SourceAdapter):
    """Fetch economic data from the Bureau of Labor Statistics API."""

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "bls"

    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch latest data for a macro contract spec from BLS."""
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

        # Build request payload.
        end_year = now.year
        start_year = end_year - 2  # 3 years of history.

        payload: dict[str, Any] = {
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self._api_key:
            payload["registrationkey"] = self._api_key

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    _API_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            logger.exception("bls_fetch_error", indicator=spec.indicator)
            return FetchResult(
                source_name=self.name,
                source_key=series_id,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error=f"api_error: {exc}",
            )

        # Parse response.
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

        # Quality: 1.0 for final data, 0.5 for preliminary.
        is_preliminary = normalized.get("is_preliminary", False)
        quality = 0.5 if is_preliminary else 1.0

        return FetchResult(
            source_name=self.name,
            source_key=series_id,
            ts_source=now,
            raw_json=raw,
            normalized_json=normalized,
            quality_score=quality,
        )

    @staticmethod
    def _parse_response(
        raw: dict[str, Any],
        indicator: str,
        series_id: str,
    ) -> dict[str, Any]:
        """Parse BLS API response into normalized dict."""
        status = raw.get("status", "")
        if status != "REQUEST_SUCCEEDED":
            return {"error": f"bls_status: {status}"}

        results = raw.get("Results", {})
        series_list = results.get("series", [])
        if not series_list:
            return {"error": "no_series_data"}

        data = series_list[0].get("data", [])
        if not data:
            return {"error": "no_data_points"}

        # BLS returns data newest-first.
        latest = data[0]
        previous = data[1] if len(data) > 1 else None

        latest_value = float(latest["value"])
        latest_year = int(latest["year"])
        latest_period = latest.get("periodName", "")

        # Check for preliminary flag.
        footnotes = latest.get("footnotes", [])
        is_preliminary = any(
            "preliminary" in (fn.get("text", "") or "").lower()
            for fn in footnotes
        )

        result: dict[str, Any] = {
            "indicator": indicator,
            "series_id": series_id,
            "latest_value": latest_value,
            "latest_year": latest_year,
            "latest_period": latest_period,
            "is_preliminary": is_preliminary,
        }

        if previous is not None:
            prev_value = float(previous["value"])
            result["previous_value"] = prev_value
            result["month_over_month_change"] = latest_value - prev_value
            if prev_value != 0:
                result["month_over_month_pct"] = (
                    (latest_value - prev_value) / prev_value * 100.0
                )

        # Build historical series for distribution fitting.
        history: list[dict[str, Any]] = []
        for pt in data:
            try:
                history.append({
                    "year": int(pt["year"]),
                    "period": pt.get("periodName", ""),
                    "value": float(pt["value"]),
                })
            except (ValueError, KeyError):
                continue
        result["history"] = history

        return result

    async def health_check(self) -> bool:
        """Check if BLS API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://api.bls.gov/publicAPI/v2/")
                return resp.status_code < 500
        except Exception:
            return False

"""Aviation Weather Center adapter — METAR observation data.

Fetches actual observations from aviationweather.gov for station IDs
near the contract location.  Used near resolution windows to validate
forecasts against reality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.contracts.weather import WeatherContractSpec
from src.sources.base import FetchResult, SourceAdapter

logger = structlog.get_logger(__name__)

_AWC_BASE = "https://aviationweather.gov/api/data/metar"


class AWCAdapter(SourceAdapter):
    """Fetch METAR observations from Aviation Weather Center."""

    @property
    def name(self) -> str:
        return "awc"

    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch latest METAR observation for the contract's station IDs."""
        if not isinstance(spec, WeatherContractSpec):
            return FetchResult(
                source_name=self.name,
                source_key="",
                ts_source=datetime.now(timezone.utc),
                raw_json={},
                normalized_json={},
                error="invalid_spec_type",
            )

        now = datetime.now(timezone.utc)
        station_ids = spec.nws_station_ids
        if not station_ids:
            return FetchResult(
                source_name=self.name,
                source_key=spec.location,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error="no_station_ids",
            )

        # Fetch METAR for all stations.
        ids_str = ",".join(station_ids)
        params = {"ids": ids_str, "format": "json"}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(_AWC_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("awc_fetch_error", stations=ids_str)
            return FetchResult(
                source_name=self.name,
                source_key=ids_str,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error="fetch_failed",
            )

        # data is a list of METAR observations.
        if not isinstance(data, list) or not data:
            return FetchResult(
                source_name=self.name,
                source_key=ids_str,
                ts_source=now,
                raw_json={"response": data},
                normalized_json={},
                error="no_observations",
            )

        # Normalize the most recent observation.
        obs = data[0]
        normalized = _normalize_metar(obs)

        return FetchResult(
            source_name=self.name,
            source_key=ids_str,
            ts_source=now,
            raw_json={"observations": data},
            normalized_json=normalized,
            quality_score=1.0,  # Observations are direct measurements.
        )

    async def health_check(self) -> bool:
        """Check if aviationweather.gov METAR endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    _AWC_BASE, params={"ids": "KJFK", "format": "json"}
                )
                return resp.status_code == 200
        except Exception:
            return False


def _normalize_metar(obs: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized values from a single METAR observation dict."""
    result: dict[str, Any] = {}

    # Temperature (METAR reports in °C, convert to °F).
    temp_c = obs.get("temp")
    if temp_c is not None:
        try:
            temp_c = float(temp_c)
            result["temperature_f"] = temp_c * 9.0 / 5.0 + 32.0
            result["temperature_c"] = temp_c
        except (ValueError, TypeError):
            pass

    # Dewpoint.
    dewp = obs.get("dewp")
    if dewp is not None:
        try:
            result["dewpoint_c"] = float(dewp)
        except (ValueError, TypeError):
            pass

    # Wind speed (knots).
    wspd = obs.get("wspd")
    if wspd is not None:
        try:
            result["wind_speed_kt"] = float(wspd)
        except (ValueError, TypeError):
            pass

    # Visibility (statute miles).
    visib = obs.get("visib")
    if visib is not None:
        try:
            result["visibility_sm"] = float(visib)
        except (ValueError, TypeError):
            pass

    # Weather string (e.g., "RA", "SN", "-RA").
    wxstring = obs.get("wxString")
    if wxstring:
        result["wx_string"] = wxstring
        result["has_precipitation"] = any(
            code in wxstring.upper()
            for code in ("RA", "SN", "DZ", "TS", "GR", "PL", "SG")
        )
        result["has_snow"] = any(
            code in wxstring.upper() for code in ("SN", "SG")
        )
    else:
        result["has_precipitation"] = False
        result["has_snow"] = False

    # Station ID.
    result["station_id"] = obs.get("icaoId", "")

    # Observation time.
    obs_time = obs.get("obsTime") or obs.get("reportTime")
    if obs_time:
        result["observation_time"] = obs_time

    return result

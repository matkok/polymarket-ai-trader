"""NWS API adapter — fetches hourly forecasts and QPF from api.weather.gov."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from src.contracts.weather import WeatherContractSpec
from src.sources.base import FetchResult, SourceAdapter
from src.sources.geocoder import geocode

logger = structlog.get_logger(__name__)

# Grid lookup cache: (lat, lon) → (office, grid_x, grid_y)
_grid_cache: dict[tuple[float, float], tuple[str, int, int]] = {}

_DEFAULT_USER_AGENT = "agent-trader-structured/0.1 (weather research)"

# Max retries for transient NWS errors.
_MAX_RETRIES = 3


class NWSAdapter(SourceAdapter):
    """Fetch hourly forecasts from the National Weather Service API."""

    def __init__(self, user_agent: str = "") -> None:
        self._user_agent = user_agent or _DEFAULT_USER_AGENT

    @property
    def name(self) -> str:
        return "nws"

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._user_agent, "Accept": "application/geo+json"}

    # ------------------------------------------------------------------
    # Grid resolution
    # ------------------------------------------------------------------

    async def _resolve_grid(
        self, lat: float, lon: float
    ) -> tuple[str, int, int] | None:
        """Resolve (lat, lon) to NWS grid (office, x, y).

        Results are cached in the module-level ``_grid_cache``.
        """
        key = (round(lat, 4), round(lon, 4))
        if key in _grid_cache:
            return _grid_cache[key]

        url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()

            props = data["properties"]
            office = props["gridId"]
            grid_x = int(props["gridX"])
            grid_y = int(props["gridY"])
            _grid_cache[key] = (office, grid_x, grid_y)
            logger.info(
                "nws_grid_resolved",
                lat=lat, lon=lon, office=office, x=grid_x, y=grid_y,
            )
            return (office, grid_x, grid_y)
        except Exception:
            logger.exception("nws_grid_error", lat=lat, lon=lon)
            return None

    # ------------------------------------------------------------------
    # Hourly forecast fetch
    # ------------------------------------------------------------------

    async def _fetch_hourly(
        self, office: str, grid_x: int, grid_y: int
    ) -> dict[str, Any] | None:
        """Fetch hourly forecast data from NWS gridpoints endpoint."""
        url = (
            f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}"
            f"/forecast/hourly"
        )
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url, headers=self._headers())
                    if resp.status_code in (500, 503) and attempt < _MAX_RETRIES:
                        logger.warning(
                            "nws_transient_error",
                            status=resp.status_code, attempt=attempt,
                        )
                        continue
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError:
                if attempt < _MAX_RETRIES:
                    continue
                logger.exception("nws_fetch_error", url=url)
                return None
            except Exception:
                logger.exception("nws_fetch_error", url=url)
                return None
        return None

    # ------------------------------------------------------------------
    # Gridpoints (raw QPF) fetch
    # ------------------------------------------------------------------

    async def _fetch_gridpoints(
        self, office: str, grid_x: int, grid_y: int
    ) -> dict[str, Any] | None:
        """Fetch raw gridpoints data (includes quantitativePrecipitation)."""
        url = (
            f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}"
        )
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url, headers=self._headers())
                    if resp.status_code in (500, 503) and attempt < _MAX_RETRIES:
                        logger.warning(
                            "nws_gridpoints_transient_error",
                            status=resp.status_code, attempt=attempt,
                        )
                        continue
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError:
                if attempt < _MAX_RETRIES:
                    continue
                logger.exception("nws_gridpoints_fetch_error", url=url)
                return None
            except Exception:
                logger.exception("nws_gridpoints_fetch_error", url=url)
                return None
        return None

    # ------------------------------------------------------------------
    # QPF extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_iso_duration(duration: str) -> float:
        """Parse an ISO 8601 duration string (e.g. ``PT6H``, ``PT1H``) to hours."""
        m = re.match(r"PT?(?:(\d+)H)?(?:(\d+)M)?", duration, re.IGNORECASE)
        if not m:
            return 0.0
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        return hours + minutes / 60.0

    @staticmethod
    def _extract_qpf(
        raw: dict[str, Any],
        target_start: datetime | None,
        target_end: datetime | None,
    ) -> dict[str, Any]:
        """Extract quantitative precipitation forecast from gridpoints data.

        Returns a dict with:
        - ``qpf_hourly_inches``: list of (iso_timestamp, amount_inches) tuples
        - ``qpf_total_inches``: sum of all forecast amounts in inches
        - ``qpf_coverage_hours``: total hours covered by QPF data
        - ``qpf_periods``: count of QPF periods in range
        """
        props = raw.get("properties", {})
        qpf_data = props.get("quantitativePrecipitation", {})
        values = qpf_data.get("values", [])

        if not values:
            return {}

        hourly_inches: list[tuple[str, float]] = []
        total_mm = 0.0
        coverage_hours = 0.0

        for entry in values:
            valid_time = entry.get("validTime", "")
            value_mm = entry.get("value")
            if value_mm is None:
                continue

            # Parse "2026-02-24T00:00:00+00:00/PT6H" format.
            parts = valid_time.split("/")
            if len(parts) != 2:
                continue

            ts_str, duration_str = parts
            try:
                period_start = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue

            period_hours = NWSAdapter._parse_iso_duration(duration_str)
            if period_hours <= 0:
                continue

            period_end = period_start + timedelta(hours=period_hours)

            # Filter to target date range if provided.
            if target_start and period_end <= target_start:
                continue
            if target_end and period_start >= target_end:
                continue

            amount_inches = float(value_mm) / 25.4
            hourly_inches.append((ts_str, amount_inches))
            total_mm += float(value_mm)
            coverage_hours += period_hours

        if not hourly_inches:
            return {}

        return {
            "qpf_hourly_inches": hourly_inches,
            "qpf_total_inches": total_mm / 25.4,
            "qpf_coverage_hours": coverage_hours,
            "qpf_periods": len(hourly_inches),
        }

    # ------------------------------------------------------------------
    # Extract relevant forecast values
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_forecast_values(
        raw: dict[str, Any],
        spec: WeatherContractSpec,
        target_start: datetime | None,
        target_end: datetime | None,
    ) -> dict[str, Any]:
        """Extract normalized forecast values from NWS hourly data.

        Returns a dict with metric-specific values (temperature, precip chance, etc.)
        plus ``lead_hours`` (hours until the forecast period).
        """
        periods = raw.get("properties", {}).get("periods", [])
        if not periods:
            return {"error": "no_periods"}

        now = datetime.now(timezone.utc)
        relevant: list[dict] = []

        for p in periods:
            start_iso = p.get("startTime", "")
            try:
                p_start = datetime.fromisoformat(start_iso)
            except (ValueError, TypeError):
                continue

            if target_start and target_end:
                if p_start < target_start or p_start > target_end:
                    continue

            relevant.append(p)

        if not relevant:
            # Fall back to the closest 24 hours of data.
            relevant = periods[:24]

        # Compute lead time from now to the middle of the relevant window.
        first_start = datetime.fromisoformat(relevant[0]["startTime"])
        lead_hours = max(0.0, (first_start - now).total_seconds() / 3600)

        metric = spec.metric
        result: dict[str, Any] = {"lead_hours": lead_hours, "n_periods": len(relevant)}

        if metric in ("temperature_high", "temperature_low", "temperature"):
            temps = [p["temperature"] for p in relevant if "temperature" in p]
            if temps:
                result["forecast_max"] = max(temps)
                result["forecast_min"] = min(temps)
                result["forecast_mean"] = sum(temps) / len(temps)
                # NWS returns °F by default.
                result["unit"] = relevant[0].get("temperatureUnit", "F")

        if metric in ("precipitation", "snowfall", "snow_occurrence"):
            probs = []
            for p in relevant:
                prob = p.get("probabilityOfPrecipitation", {})
                if isinstance(prob, dict):
                    val = prob.get("value")
                    if val is not None:
                        probs.append(float(val) / 100.0)
                elif isinstance(prob, (int, float)):
                    probs.append(float(prob) / 100.0)

            if probs:
                result["precip_prob_max"] = max(probs)
                result["precip_prob_mean"] = sum(probs) / len(probs)
            else:
                result["precip_prob_max"] = 0.0
                result["precip_prob_mean"] = 0.0

        if metric == "snow_occurrence":
            temps = [p["temperature"] for p in relevant if "temperature" in p]
            if temps:
                result["forecast_min"] = min(temps)

        return result

    # ------------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------------

    @staticmethod
    def _quality_score(lead_hours: float, forecast_horizon_hours: float = 168.0) -> float:
        """Compute quality score that decays with lead time.

        ``exp(-lead_hours / (2 * forecast_horizon_hours))``
        """
        return math.exp(-lead_hours / (2.0 * forecast_horizon_hours))

    # ------------------------------------------------------------------
    # Public fetch interface
    # ------------------------------------------------------------------

    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch NWS hourly forecast for the given weather contract spec."""
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

        # Geocode the location.
        coords = await geocode(spec.location)
        if coords is None:
            return FetchResult(
                source_name=self.name,
                source_key=spec.location,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error=f"geocode_failed: {spec.location}",
            )

        lat, lon = coords

        # Resolve grid.
        grid = await self._resolve_grid(lat, lon)
        if grid is None:
            return FetchResult(
                source_name=self.name,
                source_key=spec.location,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error="grid_resolution_failed",
            )

        office, gx, gy = grid

        # Fetch hourly forecast.
        raw = await self._fetch_hourly(office, gx, gy)
        if raw is None:
            return FetchResult(
                source_name=self.name,
                source_key=spec.location,
                ts_source=now,
                raw_json={},
                normalized_json={},
                error="forecast_fetch_failed",
            )

        # Parse target dates from spec (if available as datetimes).
        target_start = None
        target_end = None
        if spec.date_start:
            try:
                target_start = datetime.fromisoformat(spec.date_start)
            except (ValueError, TypeError):
                pass
        if spec.date_end:
            try:
                target_end = datetime.fromisoformat(spec.date_end)
            except (ValueError, TypeError):
                pass

        # Extract normalized values.
        normalized = self._extract_forecast_values(raw, spec, target_start, target_end)
        if "error" in normalized:
            return FetchResult(
                source_name=self.name,
                source_key=spec.location,
                ts_source=now,
                raw_json=raw,
                normalized_json=normalized,
                error=normalized["error"],
            )

        # Fetch QPF data for precipitation/snowfall metrics.
        if spec.metric in ("precipitation", "snowfall", "snow_occurrence"):
            gridpoints_raw = await self._fetch_gridpoints(office, gx, gy)
            if gridpoints_raw is not None:
                qpf = self._extract_qpf(gridpoints_raw, target_start, target_end)
                if qpf:
                    normalized.update(qpf)
                    logger.info(
                        "nws_qpf_extracted",
                        location=spec.location,
                        qpf_total_inches=qpf.get("qpf_total_inches"),
                        qpf_coverage_hours=qpf.get("qpf_coverage_hours"),
                        qpf_periods=qpf.get("qpf_periods"),
                    )

        lead_hours = normalized.get("lead_hours", 0.0)
        quality = self._quality_score(lead_hours)

        return FetchResult(
            source_name=self.name,
            source_key=f"nws:{office}/{gx},{gy}",
            ts_source=now,
            raw_json=raw,
            normalized_json=normalized,
            quality_score=quality,
        )

    async def health_check(self) -> bool:
        """Check if api.weather.gov is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.weather.gov", headers=self._headers()
                )
                return resp.status_code == 200
        except Exception:
            return False


def clear_grid_cache() -> None:
    """Clear the NWS grid cache (for testing)."""
    _grid_cache.clear()

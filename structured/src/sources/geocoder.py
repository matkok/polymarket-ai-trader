"""Location geocoding: city name → (lat, lon).

Static dictionary for ~50 common US cities with Census geocoder fallback.
"""

from __future__ import annotations

import structlog
import httpx

logger = structlog.get_logger(__name__)

# Module-level cache for geocode results (avoids repeated API calls).
_cache: dict[str, tuple[float, float]] = {}

# ---------------------------------------------------------------------------
# Static dictionary — covers the most common Polymarket weather locations
# ---------------------------------------------------------------------------

_STATIC: dict[str, tuple[float, float]] = {
    "new york": (40.7128, -74.0060),
    "new york city": (40.7128, -74.0060),
    "nyc": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611),
    "dallas": (32.7767, -96.7970),
    "san jose": (37.3382, -121.8863),
    "austin": (30.2672, -97.7431),
    "jacksonville": (30.3322, -81.6557),
    "fort worth": (32.7555, -97.3308),
    "columbus": (39.9612, -82.9988),
    "charlotte": (35.2271, -80.8431),
    "indianapolis": (39.7684, -86.1581),
    "san francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "washington": (38.9072, -77.0369),
    "washington dc": (38.9072, -77.0369),
    "washington, dc": (38.9072, -77.0369),
    "nashville": (36.1627, -86.7816),
    "oklahoma city": (35.4676, -97.5164),
    "el paso": (31.7619, -106.4850),
    "boston": (42.3601, -71.0589),
    "portland": (45.5155, -122.6789),
    "las vegas": (36.1699, -115.1398),
    "memphis": (35.1495, -90.0490),
    "louisville": (38.2527, -85.7585),
    "baltimore": (39.2904, -76.6122),
    "milwaukee": (43.0389, -87.9065),
    "albuquerque": (35.0844, -106.6504),
    "tucson": (32.2226, -110.9747),
    "fresno": (36.7378, -119.7871),
    "sacramento": (38.5816, -121.4944),
    "mesa": (33.4152, -111.8315),
    "kansas city": (39.0997, -94.5786),
    "atlanta": (33.7490, -84.3880),
    "omaha": (41.2565, -95.9345),
    "colorado springs": (38.8339, -104.8214),
    "raleigh": (35.7796, -78.6382),
    "miami": (25.7617, -80.1918),
    "minneapolis": (44.9778, -93.2650),
    "tampa": (27.9506, -82.4572),
    "new orleans": (29.9511, -90.0715),
    "cleveland": (41.4993, -81.6944),
    "detroit": (42.3314, -83.0458),
    "pittsburgh": (40.4406, -79.9959),
    "st louis": (38.6270, -90.1994),
    "st. louis": (38.6270, -90.1994),
    "saint louis": (38.6270, -90.1994),
    "honolulu": (21.3069, -157.8583),
    "anchorage": (61.2181, -149.9003),
    "atlantic": (30.0, -50.0),
    "gulf of mexico": (25.0, -90.0),
}


async def geocode(location: str) -> tuple[float, float] | None:
    """Resolve a location name to (lat, lon).

    Checks the static dictionary first, then the module-level cache,
    and finally falls back to the US Census Bureau geocoder API.

    Returns ``None`` if the location cannot be resolved.
    """
    key = location.strip().lower()

    # 1. Static lookup.
    if key in _STATIC:
        return _STATIC[key]

    # 2. Module-level cache.
    if key in _cache:
        return _cache[key]

    # 3. Census geocoder fallback.
    result = await _census_geocode(location)
    if result is not None:
        _cache[key] = result
    return result


async def _census_geocode(location: str) -> tuple[float, float] | None:
    """Query the US Census Bureau geocoder for *location*."""
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {
        "address": location,
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        matches = data.get("result", {}).get("addressMatches", [])
        if not matches:
            logger.warning("geocode_no_match", location=location)
            return None

        coords = matches[0]["coordinates"]
        lat = float(coords["y"])
        lon = float(coords["x"])
        logger.info("geocode_census_hit", location=location, lat=lat, lon=lon)
        return (lat, lon)
    except Exception:
        logger.exception("geocode_census_error", location=location)
        return None


def clear_cache() -> None:
    """Clear the module-level geocode cache (for testing)."""
    _cache.clear()

"""Tests for S2: Weather Engine v1.

Covers: sources/base, sources/geocoder, sources/nws, sources/awc,
        trading/date_resolver, engines/base, engines/weather,
        trading/weather_pipeline, scheduler weather wiring.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.policy import Policy
from src.contracts.weather import WeatherContractSpec
from src.engines.base import PriceEstimate, PricingEngine
from src.engines.weather import WeatherEngine
from src.execution.paper_executor import PaperFill
from src.portfolio.risk_manager import RiskCheck
from src.sources.base import FetchResult, SourceAdapter
from src.sources.geocoder import clear_cache, geocode
from src.sources.nws import NWSAdapter, clear_grid_cache
from src.sources.awc import AWCAdapter
from src.trading.date_resolver import resolve_date_range
from src.trading.weather_pipeline import WeatherPipeline, _reconstruct_spec


# ===========================================================================
# sources/base
# ===========================================================================


class TestFetchResult:
    """FetchResult dataclass tests."""

    def test_ok_when_no_error(self) -> None:
        fr = FetchResult(
            source_name="test", source_key="k",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={},
        )
        assert fr.ok is True

    def test_not_ok_when_error(self) -> None:
        fr = FetchResult(
            source_name="test", source_key="k",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={},
            error="something failed",
        )
        assert fr.ok is False

    def test_to_observation_dict(self) -> None:
        now = datetime.now(timezone.utc)
        fr = FetchResult(
            source_name="nws", source_key="loc",
            ts_source=now,
            raw_json={"a": 1}, normalized_json={"b": 2},
            quality_score=0.8,
        )
        d = fr.to_observation_dict("weather")
        assert d["category"] == "weather"
        assert d["source_name"] == "nws"
        assert d["quality_score"] == 0.8

    def test_quality_score_default(self) -> None:
        fr = FetchResult(
            source_name="test", source_key="k",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={},
        )
        assert fr.quality_score == 1.0


class TestSourceAdapterABC:
    """SourceAdapter ABC cannot be instantiated directly."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            SourceAdapter()  # type: ignore[abstract]


# ===========================================================================
# sources/geocoder
# ===========================================================================


class TestGeocoderStatic:
    """Static dictionary lookups."""

    async def test_known_city(self) -> None:
        coords = await geocode("Dallas")
        assert coords is not None
        lat, lon = coords
        assert abs(lat - 32.7767) < 0.01
        assert abs(lon - (-96.7970)) < 0.01

    async def test_known_city_case_insensitive(self) -> None:
        coords = await geocode("NEW YORK")
        assert coords is not None
        assert abs(coords[0] - 40.7128) < 0.01

    async def test_known_city_with_whitespace(self) -> None:
        coords = await geocode("  Chicago  ")
        assert coords is not None

    async def test_alias_nyc(self) -> None:
        coords = await geocode("NYC")
        assert coords is not None


class TestGeocoderFallback:
    """Census geocoder fallback (mocked)."""

    async def test_census_fallback(self) -> None:
        clear_cache()
        with patch("src.sources.geocoder._census_geocode", new_callable=AsyncMock) as mock:
            mock.return_value = (35.0, -97.0)
            coords = await geocode("Smalltown, OK")
            assert coords == (35.0, -97.0)

    async def test_census_failure_returns_none(self) -> None:
        clear_cache()
        with patch("src.sources.geocoder._census_geocode", new_callable=AsyncMock) as mock:
            mock.return_value = None
            coords = await geocode("Nonexistent Place XYZ123")
            assert coords is None


# ===========================================================================
# trading/date_resolver
# ===========================================================================


class TestDateResolverNamedDates:
    """Named date patterns."""

    def test_christmas_day(self) -> None:
        result = resolve_date_range("Christmas Day")
        assert result is not None
        start, end = result
        assert start.month == 12
        assert start.day == 25

    def test_christmas_day_with_year(self) -> None:
        result = resolve_date_range("Christmas Day 2026")
        assert result is not None
        start, end = result
        assert start.year == 2026
        assert start.month == 12
        assert start.day == 25

    def test_new_years_day(self) -> None:
        result = resolve_date_range("New Year's Day")
        assert result is not None
        start, _ = result
        assert start.month == 1
        assert start.day == 1

    def test_independence_day(self) -> None:
        result = resolve_date_range("Independence Day")
        assert result is not None
        start, _ = result
        assert start.month == 7
        assert start.day == 4


class TestDateResolverQuarters:
    """Quarter patterns."""

    def test_q1_2026(self) -> None:
        result = resolve_date_range("Q1 2026")
        assert result is not None
        start, end = result
        assert start.month == 1
        assert start.day == 1
        assert end.month == 3
        assert end.day == 31

    def test_2026_q3(self) -> None:
        result = resolve_date_range("2026 Q3")
        assert result is not None
        start, end = result
        assert start.month == 7
        assert end.month == 9

    def test_q4_end_december(self) -> None:
        result = resolve_date_range("Q4 2026")
        assert result is not None
        _, end = result
        assert end.month == 12
        assert end.day == 31


class TestDateResolverMonthYear:
    """Month + year patterns."""

    def test_january_2026(self) -> None:
        result = resolve_date_range("January 2026")
        assert result is not None
        start, end = result
        assert start.month == 1
        assert start.year == 2026
        assert end.month == 1
        assert end.day == 31

    def test_february_2026_leap(self) -> None:
        result = resolve_date_range("February 2028")
        assert result is not None
        _, end = result
        assert end.day == 29  # 2028 is a leap year.


class TestDateResolverHurricaneSeason:
    """Hurricane season patterns."""

    def test_hurricane_season_2026(self) -> None:
        result = resolve_date_range("2026 Atlantic hurricane season")
        assert result is not None
        start, end = result
        assert start.month == 6
        assert end.month == 11
        assert end.day == 30


class TestDateResolverGeneric:
    """Generic dateutil-based parsing."""

    def test_specific_date(self) -> None:
        result = resolve_date_range("June 15, 2026")
        assert result is not None
        start, end = result
        assert start.month == 6
        assert start.day == 15

    def test_empty_string(self) -> None:
        assert resolve_date_range("") is None

    def test_none_like(self) -> None:
        assert resolve_date_range("   ") is None

    def test_unparseable(self) -> None:
        assert resolve_date_range("xyzzy gibberish") is None


# ===========================================================================
# engines/base
# ===========================================================================


class TestPriceEstimate:
    """PriceEstimate dataclass tests."""

    def test_to_engine_price_dict(self) -> None:
        now = datetime.now(timezone.utc)
        pe = PriceEstimate(p_yes=0.70, confidence=0.85, source_confidence=0.9)
        d = pe.to_engine_price_dict(
            market_id="mkt-1",
            category="weather",
            engine_version="weather_v1",
            ts_utc=now,
            p_market=0.50,
        )
        assert d["p_yes"] == 0.70
        assert d["confidence"] == 0.85
        assert abs(d["edge_before_costs"] - 0.20) < 0.001

    def test_edge_none_when_no_market_price(self) -> None:
        now = datetime.now(timezone.utc)
        pe = PriceEstimate(p_yes=0.70, confidence=0.85)
        d = pe.to_engine_price_dict(
            market_id="mkt-1", category="weather",
            engine_version="v1", ts_utc=now,
        )
        assert d["edge_before_costs"] is None


class TestPricingEngineABC:
    """PricingEngine ABC cannot be instantiated."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            PricingEngine()  # type: ignore[abstract]


# ===========================================================================
# engines/weather — WeatherEngine
# ===========================================================================


class TestWeatherEngineTemperature:
    """Temperature metric — Gaussian CDF."""

    def test_high_above_threshold(self) -> None:
        """Forecast max well above threshold → high p_yes."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=100.0, threshold_unit="F",
            comparison="above",
        )
        obs = {"forecast_max": 105.0, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert result.p_yes > 0.80

    def test_high_below_threshold(self) -> None:
        """Forecast max well below threshold → low p_yes."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=100.0, threshold_unit="F",
            comparison="above",
        )
        obs = {"forecast_max": 90.0, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert result.p_yes < 0.20

    def test_temperature_low_uses_forecast_min(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_low",
            location="Chicago", threshold=20.0, threshold_unit="F",
            comparison="below",
        )
        obs = {"forecast_min": 15.0, "lead_hours": 24.0}
        result = engine.compute(spec, obs)
        assert result.p_yes > 0.70

    def test_celsius_conversion(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=38.0, threshold_unit="C",
            comparison="above",
        )
        # 38°C = 100.4°F; forecast 105°F
        obs = {"forecast_max": 105.0, "lead_hours": 6.0}
        result = engine.compute(spec, obs)
        assert result.p_yes > 0.70

    def test_confidence_decays_with_lead_time(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=100.0, threshold_unit="F",
            comparison="above",
        )
        obs_near = {"forecast_max": 105.0, "lead_hours": 6.0}
        obs_far = {"forecast_max": 105.0, "lead_hours": 120.0}
        near = engine.compute(spec, obs_near)
        far = engine.compute(spec, obs_far)
        assert near.confidence > far.confidence

    def test_no_forecast_value_returns_low_confidence(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=100.0, threshold_unit="F",
            comparison="above",
        )
        result = engine.compute(spec, {})
        assert result.confidence == 0.1
        assert result.p_yes == 0.5

    def test_no_threshold_returns_low_confidence(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=None, threshold_unit="F",
            comparison="above",
        )
        obs = {"forecast_max": 105.0, "lead_hours": 6.0}
        result = engine.compute(spec, obs)
        assert result.confidence == 0.1

    def test_p_yes_clamped(self) -> None:
        """p_yes is clamped to [0.01, 0.99]."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=50.0, threshold_unit="F",
            comparison="above",
        )
        obs = {"forecast_max": 105.0, "lead_hours": 0.0}
        result = engine.compute(spec, obs)
        assert 0.01 <= result.p_yes <= 0.99


class TestWeatherEnginePrecipitation:
    """Precipitation metric — zero-inflated model."""

    def test_high_precip_prob_above_threshold(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="NYC", threshold=0.5, threshold_unit="inches",
            comparison="above",
        )
        obs = {"precip_prob_mean": 0.80, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert result.p_yes > 0.20
        assert result.confidence <= 0.70

    def test_zero_precip_prob(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="Phoenix", threshold=1.0, threshold_unit="inches",
            comparison="above",
        )
        obs = {"precip_prob_mean": 0.0, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert result.p_yes < 0.05

    def test_confidence_capped_at_070(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="NYC", threshold=0.1, threshold_unit="inches",
            comparison="above",
        )
        obs = {"precip_prob_mean": 0.90, "lead_hours": 0.0}
        result = engine.compute(spec, obs)
        assert result.confidence <= 0.70

    def test_no_threshold(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="NYC", threshold=None, threshold_unit="inches",
            comparison="above",
        )
        obs = {"precip_prob_mean": 0.50, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert 0.01 <= result.p_yes <= 0.99


class TestWeatherEngineSnowOccurrence:
    """Snow occurrence — joint probability."""

    def test_cold_and_wet(self) -> None:
        """Cold temp + high precip → higher snow probability."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="snow_occurrence",
            location="Chicago",
        )
        obs = {"forecast_min": 25.0, "precip_prob_max": 0.80, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert result.p_yes > 0.30

    def test_warm_weather(self) -> None:
        """Warm temp → low snow probability."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="snow_occurrence",
            location="Miami",
        )
        obs = {"forecast_min": 60.0, "precip_prob_max": 0.80, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert result.p_yes < 0.05

    def test_confidence_capped_at_075(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="snow_occurrence",
            location="Chicago",
        )
        obs = {"forecast_min": 20.0, "precip_prob_max": 0.90, "lead_hours": 0.0}
        result = engine.compute(spec, obs)
        assert result.confidence <= 0.75


class TestWeatherEngineHurricane:
    """Hurricane — seasonal base rate."""

    def test_base_rate_cat3(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="hurricane",
            location="Atlantic", threshold=3.0,
            threshold_unit="category", comparison="at_least",
        )
        result = engine.compute(spec, {})
        assert abs(result.p_yes - 0.45) < 0.01
        assert result.confidence == 0.30

    def test_base_rate_cat5(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="hurricane",
            location="Atlantic", threshold=5.0,
        )
        result = engine.compute(spec, {})
        assert abs(result.p_yes - 0.05) < 0.01

    def test_base_rate_any(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="hurricane",
            location="Atlantic",
        )
        result = engine.compute(spec, {})
        assert result.p_yes == 0.70


class TestWeatherEngineClimatology:
    """Climatology lookup tests."""

    def test_known_city_exact(self) -> None:
        normal = WeatherEngine._get_monthly_normal("New York", 2)
        assert abs(normal - 3.19) < 0.01

    def test_known_city_case_insensitive(self) -> None:
        normal = WeatherEngine._get_monthly_normal("NEW YORK", 2)
        assert abs(normal - 3.19) < 0.01

    def test_known_city_substring(self) -> None:
        """'New York City' matches 'new york' via substring."""
        normal = WeatherEngine._get_monthly_normal("New York City", 2)
        assert abs(normal - 3.19) < 0.01

    def test_unknown_city_returns_default(self) -> None:
        normal = WeatherEngine._get_monthly_normal("Anchorage", 1)
        assert normal == 3.0

    def test_chicago_july(self) -> None:
        normal = WeatherEngine._get_monthly_normal("Chicago", 7)
        assert abs(normal - 3.96) < 0.01

    def test_phoenix_june(self) -> None:
        """Phoenix June is near-zero."""
        normal = WeatherEngine._get_monthly_normal("Phoenix", 6)
        assert normal < 0.1


class TestWeatherEngineCumulativePrecipitation:
    """Cumulative precipitation model for multi-day contracts."""

    def test_nyc_feb_below_2_inches(self) -> None:
        """NYC February: monthly normal 3.19" → P(< 2") should be low."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="New York", threshold=2.0, threshold_unit="inches",
            comparison="below",
        )
        obs = {
            "precip_prob_mean": 0.60,
            "lead_hours": 12.0,
            "contract_span_days": 28.0,
            "current_month": 2,
            "qpf_total_inches": 1.2,
            "qpf_coverage_hours": 168.0,  # 7 days
        }
        result = engine.compute(spec, obs)
        # P(< 2") in a month with 3.19" normal should be LOW.
        assert result.p_yes < 0.30
        # Confidence should be low — only 25% coverage.
        assert result.confidence < 0.40
        assert result.model_details["model"] == "cumulative"
        assert result.model_details["clim_monthly_normal"] == 3.19

    def test_cumulative_high_coverage_higher_confidence(self) -> None:
        """When forecast covers most of the contract → higher confidence."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="Phoenix", threshold=1.0, threshold_unit="inches",
            comparison="below",
        )
        obs = {
            "precip_prob_mean": 0.10,
            "lead_hours": 6.0,
            "contract_span_days": 7.0,
            "current_month": 6,
            "qpf_total_inches": 0.02,
            "qpf_coverage_hours": 168.0,  # Full coverage.
        }
        result = engine.compute(spec, obs)
        # Phoenix June: near zero rain. P(< 1") should be very high.
        assert result.p_yes > 0.80
        # Full coverage → higher confidence.
        assert result.confidence > 0.50

    def test_cumulative_above_threshold(self) -> None:
        """'above' comparison inverts the CDF."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="Miami", threshold=5.0, threshold_unit="inches",
            comparison="above",
        )
        obs = {
            "precip_prob_mean": 0.70,
            "lead_hours": 12.0,
            "contract_span_days": 30.0,
            "current_month": 6,
            "qpf_total_inches": 3.0,
            "qpf_coverage_hours": 168.0,
        }
        result = engine.compute(spec, obs)
        # Miami June normal ~9.89". P(> 5") should be high.
        assert result.p_yes > 0.50

    def test_cumulative_no_threshold_returns_low_confidence(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="NYC", threshold=None,
            comparison="above",
        )
        obs = {
            "precip_prob_mean": 0.50,
            "lead_hours": 12.0,
            "contract_span_days": 28.0,
            "current_month": 2,
        }
        result = engine.compute(spec, obs)
        assert result.confidence == 0.1

    def test_cumulative_model_details_complete(self) -> None:
        """Model details include all decomposition fields."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="Chicago", threshold=3.0, threshold_unit="inches",
            comparison="below",
        )
        obs = {
            "precip_prob_mean": 0.40,
            "lead_hours": 6.0,
            "contract_span_days": 28.0,
            "current_month": 1,
            "qpf_total_inches": 0.5,
            "qpf_coverage_hours": 168.0,
        }
        result = engine.compute(spec, obs)
        d = result.model_details
        assert d["model"] == "cumulative"
        assert "qpf_total_inches" in d
        assert "forecast_total" in d
        assert "tail_mean" in d
        assert "total_mean" in d
        assert "coverage_ratio" in d
        assert "clim_monthly_normal" in d
        assert d["coverage_ratio"] == 168.0 / 24.0 / 28.0

    def test_cumulative_p_yes_clamped(self) -> None:
        """p_yes is always in [0.01, 0.99]."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="Los Angeles", threshold=0.001, threshold_unit="inches",
            comparison="below",
        )
        obs = {
            "precip_prob_mean": 0.05,
            "lead_hours": 6.0,
            "contract_span_days": 30.0,
            "current_month": 7,
            "qpf_total_inches": 0.0,
            "qpf_coverage_hours": 168.0,
        }
        result = engine.compute(spec, obs)
        assert 0.01 <= result.p_yes <= 0.99


class TestWeatherEnginePrecipitationQPF:
    """Point-in-time precipitation model with QPF enhancement."""

    def test_qpf_adjusts_gamma(self) -> None:
        """QPF data shifts the gamma distribution."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="NYC", threshold=0.5, threshold_unit="inches",
            comparison="above",
        )
        obs_no_qpf = {"precip_prob_mean": 0.80, "lead_hours": 12.0}
        obs_with_qpf = {
            "precip_prob_mean": 0.80,
            "lead_hours": 12.0,
            "qpf_total_inches": 2.0,
        }
        result_no_qpf = engine.compute(spec, obs_no_qpf)
        result_with_qpf = engine.compute(spec, obs_with_qpf)
        # With 2" QPF, probability of exceeding 0.5" should be higher.
        assert result_with_qpf.p_yes > result_no_qpf.p_yes

    def test_qpf_model_details_include_gamma(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="NYC", threshold=1.0, threshold_unit="inches",
            comparison="above",
        )
        obs = {"precip_prob_mean": 0.80, "lead_hours": 12.0, "qpf_total_inches": 1.5}
        result = engine.compute(spec, obs)
        assert result.model_details.get("qpf_total_inches") == 1.5
        assert result.model_details.get("gamma_shape") is not None
        assert result.model_details.get("gamma_scale") is not None


class TestWeatherEngineSnowfall:
    """Snowfall amount metric."""

    def test_cold_with_threshold(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="snowfall",
            location="Chicago", threshold=6.0, threshold_unit="inches",
            comparison="above",
        )
        obs = {"forecast_min": 20.0, "precip_prob_max": 0.80, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert 0.01 <= result.p_yes <= 0.99


class TestWeatherEngineDispatch:
    """Dispatch + edge cases."""

    def test_unknown_metric(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="wind_speed",
            location="Dallas",
        )
        result = engine.compute(spec, {})
        assert result.confidence == 0.1
        assert "unknown_metric" in result.model_details.get("error", "")

    def test_invalid_spec_type(self) -> None:
        engine = WeatherEngine()
        result = engine.compute("not_a_spec", {})
        assert result.confidence == 0.0

    def test_accepts_fetch_result(self) -> None:
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=100.0, threshold_unit="F",
            comparison="above",
        )
        fr = FetchResult(
            source_name="nws", source_key="test",
            ts_source=datetime.now(timezone.utc),
            raw_json={},
            normalized_json={"forecast_max": 105.0, "lead_hours": 6.0},
        )
        result = engine.compute(spec, fr)
        assert result.p_yes > 0.70

    def test_version(self) -> None:
        engine = WeatherEngine()
        assert engine.version == "weather_v2"
        assert engine.name == "weather"

    def test_precipitation_dispatches_cumulative_when_span_gt_3(self) -> None:
        """contract_span_days > 3 routes to cumulative model."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="New York", threshold=2.0, threshold_unit="inches",
            comparison="below",
        )
        obs = {
            "precip_prob_mean": 0.60,
            "lead_hours": 12.0,
            "contract_span_days": 28.0,
            "current_month": 2,
            "qpf_total_inches": 1.2,
            "qpf_coverage_hours": 168.0,
        }
        result = engine.compute(spec, obs)
        assert result.model_details.get("model") == "cumulative"

    def test_precipitation_dispatches_point_in_time_when_span_le_3(self) -> None:
        """contract_span_days <= 3 stays on point-in-time model."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="New York", threshold=0.5, threshold_unit="inches",
            comparison="above",
        )
        obs = {
            "precip_prob_mean": 0.60,
            "lead_hours": 12.0,
            "contract_span_days": 1.0,
        }
        result = engine.compute(spec, obs)
        assert result.model_details.get("model") == "point_in_time"

    def test_precipitation_no_span_uses_point_in_time(self) -> None:
        """No contract_span_days → point-in-time (backward compat)."""
        engine = WeatherEngine()
        spec = WeatherContractSpec(
            category="weather", metric="precipitation",
            location="NYC", threshold=0.5, threshold_unit="inches",
            comparison="above",
        )
        obs = {"precip_prob_mean": 0.80, "lead_hours": 12.0}
        result = engine.compute(spec, obs)
        assert result.model_details.get("model") == "point_in_time"


# ===========================================================================
# sources/nws — NWSAdapter
# ===========================================================================


class TestNWSAdapterQuality:
    """Quality score decay."""

    def test_quality_near_zero_lead(self) -> None:
        score = NWSAdapter._quality_score(0.0)
        assert abs(score - 1.0) < 0.001

    def test_quality_decays_with_lead(self) -> None:
        score_short = NWSAdapter._quality_score(12.0)
        score_long = NWSAdapter._quality_score(120.0)
        assert score_short > score_long

    def test_quality_at_horizon(self) -> None:
        # At 168h lead with 168h horizon: exp(-168/336) ≈ 0.607
        score = NWSAdapter._quality_score(168.0, 168.0)
        assert abs(score - 0.607) < 0.01


class TestNWSAdapterExtract:
    """Forecast value extraction."""

    def test_extract_temperature(self) -> None:
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", threshold=100.0, threshold_unit="F",
            comparison="above",
        )
        raw = {
            "properties": {
                "periods": [
                    {
                        "startTime": "2026-06-15T12:00:00-05:00",
                        "temperature": 102,
                        "temperatureUnit": "F",
                        "probabilityOfPrecipitation": {"value": 10},
                    },
                    {
                        "startTime": "2026-06-15T13:00:00-05:00",
                        "temperature": 104,
                        "temperatureUnit": "F",
                        "probabilityOfPrecipitation": {"value": 5},
                    },
                ]
            }
        }
        result = NWSAdapter._extract_forecast_values(raw, spec, None, None)
        assert result["forecast_max"] == 104
        assert result["forecast_min"] == 102

    def test_extract_no_periods(self) -> None:
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas",
        )
        result = NWSAdapter._extract_forecast_values(
            {"properties": {"periods": []}}, spec, None, None
        )
        assert result.get("error") == "no_periods"


class TestNWSAdapterFetch:
    """NWSAdapter.fetch() with mocked HTTP."""

    async def test_fetch_invalid_spec_type(self) -> None:
        adapter = NWSAdapter()
        result = await adapter.fetch("not_a_spec")
        assert not result.ok
        assert result.error == "invalid_spec_type"

    async def test_fetch_geocode_failure(self) -> None:
        adapter = NWSAdapter()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Nonexistent XYZ",
        )
        with patch("src.sources.nws.geocode", new_callable=AsyncMock) as mock:
            mock.return_value = None
            result = await adapter.fetch(spec)
        assert not result.ok
        assert "geocode_failed" in result.error


# ===========================================================================
# sources/nws — QPF extraction
# ===========================================================================


class TestNWSQPFExtraction:
    """_extract_qpf() static method tests."""

    def test_basic_qpf_extraction(self) -> None:
        raw = {
            "properties": {
                "quantitativePrecipitation": {
                    "values": [
                        {"validTime": "2026-02-24T00:00:00+00:00/PT6H", "value": 2.54},
                        {"validTime": "2026-02-24T06:00:00+00:00/PT6H", "value": 5.08},
                    ]
                }
            }
        }
        result = NWSAdapter._extract_qpf(raw, None, None)
        assert result["qpf_periods"] == 2
        assert abs(result["qpf_total_inches"] - (2.54 + 5.08) / 25.4) < 0.001
        assert result["qpf_coverage_hours"] == 12.0

    def test_qpf_date_filtering(self) -> None:
        """Only periods within target range are included."""
        raw = {
            "properties": {
                "quantitativePrecipitation": {
                    "values": [
                        {"validTime": "2026-02-20T00:00:00+00:00/PT6H", "value": 10.0},
                        {"validTime": "2026-02-25T00:00:00+00:00/PT6H", "value": 5.0},
                        {"validTime": "2026-03-05T00:00:00+00:00/PT6H", "value": 20.0},
                    ]
                }
            }
        }
        target_start = datetime(2026, 2, 24, tzinfo=timezone.utc)
        target_end = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)
        result = NWSAdapter._extract_qpf(raw, target_start, target_end)
        assert result["qpf_periods"] == 1
        assert abs(result["qpf_total_inches"] - 5.0 / 25.4) < 0.001

    def test_qpf_empty_values(self) -> None:
        raw = {"properties": {"quantitativePrecipitation": {"values": []}}}
        result = NWSAdapter._extract_qpf(raw, None, None)
        assert result == {}

    def test_qpf_missing_property(self) -> None:
        raw = {"properties": {}}
        result = NWSAdapter._extract_qpf(raw, None, None)
        assert result == {}

    def test_qpf_null_values_skipped(self) -> None:
        raw = {
            "properties": {
                "quantitativePrecipitation": {
                    "values": [
                        {"validTime": "2026-02-24T00:00:00+00:00/PT6H", "value": None},
                        {"validTime": "2026-02-24T06:00:00+00:00/PT6H", "value": 2.54},
                    ]
                }
            }
        }
        result = NWSAdapter._extract_qpf(raw, None, None)
        assert result["qpf_periods"] == 1
        assert abs(result["qpf_total_inches"] - 0.1) < 0.001  # 2.54mm = 0.1"


class TestNWSISODuration:
    """_parse_iso_duration() tests."""

    def test_pt6h(self) -> None:
        assert NWSAdapter._parse_iso_duration("PT6H") == 6.0

    def test_pt1h(self) -> None:
        assert NWSAdapter._parse_iso_duration("PT1H") == 1.0

    def test_pt30m(self) -> None:
        assert NWSAdapter._parse_iso_duration("PT30M") == 0.5

    def test_pt2h30m(self) -> None:
        assert NWSAdapter._parse_iso_duration("PT2H30M") == 2.5

    def test_invalid(self) -> None:
        assert NWSAdapter._parse_iso_duration("garbage") == 0.0


# ===========================================================================
# sources/awc — AWCAdapter
# ===========================================================================


class TestAWCAdapterFetch:
    """AWCAdapter.fetch() tests."""

    async def test_no_station_ids(self) -> None:
        adapter = AWCAdapter()
        spec = WeatherContractSpec(
            category="weather", metric="temperature_high",
            location="Dallas", nws_station_ids=[],
        )
        result = await adapter.fetch(spec)
        assert not result.ok
        assert result.error == "no_station_ids"

    async def test_invalid_spec_type(self) -> None:
        adapter = AWCAdapter()
        result = await adapter.fetch("not_a_spec")
        assert not result.ok

    def test_name(self) -> None:
        assert AWCAdapter().name == "awc"


# ===========================================================================
# trading/weather_pipeline
# ===========================================================================


class TestReconstructSpec:
    """_reconstruct_spec helper."""

    def test_valid_spec(self) -> None:
        spec_json = {
            "category": "weather",
            "metric": "temperature_high",
            "location": "Dallas",
            "threshold": 100.0,
            "threshold_unit": "F",
            "comparison": "above",
            "date_description": "June 15, 2026",
        }
        spec = _reconstruct_spec(spec_json)
        assert spec is not None
        assert spec.metric == "temperature_high"
        assert spec.location == "Dallas"
        assert spec.threshold == 100.0

    def test_empty_dict(self) -> None:
        spec = _reconstruct_spec({})
        assert spec is not None
        assert spec.category == "weather"

    def test_none_returns_none(self) -> None:
        # Simulating bad data.
        with patch("src.trading.weather_pipeline.WeatherContractSpec", side_effect=Exception):
            spec = _reconstruct_spec({"metric": "temp"})
            assert spec is None


class TestWeatherPipelineRunCycle:
    """WeatherPipeline.run_cycle() with mocked dependencies."""

    def _make_pipeline(self) -> tuple:
        repo = AsyncMock()
        # Entry guard: no existing position by default.
        repo.get_position.return_value = None
        # Horizon filter: resolution within policy limits by default.
        market_mock = MagicMock()
        market_mock.resolution_time_utc = datetime.now(timezone.utc) + timedelta(hours=100)
        repo.get_market.return_value = market_mock
        nws = AsyncMock(spec=NWSAdapter)
        awc = AsyncMock(spec=AWCAdapter)
        engine = MagicMock(spec=WeatherEngine)
        engine.version = "weather_v1"
        executor = MagicMock()
        risk_manager = MagicMock()
        policy = Policy()
        pipeline = WeatherPipeline(
            repo=repo, nws=nws, awc=awc, engine=engine,
            executor=executor, risk_manager=risk_manager, policy=policy,
        )
        return pipeline, repo, nws, awc, engine, executor, risk_manager

    async def test_no_markets(self) -> None:
        pipeline, repo, *_ = self._make_pipeline()
        repo.get_markets_by_category.return_value = []
        summary = await pipeline.run_cycle()
        assert summary["markets_found"] == 0

    async def test_skip_no_spec(self) -> None:
        pipeline, repo, *_ = self._make_pipeline()
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = None
        repo.get_markets_by_category.return_value = [assignment]
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1

    async def test_skip_past_market(self) -> None:
        pipeline, repo, nws, *_ = self._make_pipeline()
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = {
            "category": "weather",
            "metric": "temperature_high",
            "location": "Dallas",
            "threshold": 100.0,
            "threshold_unit": "F",
            "comparison": "above",
            "date_description": "January 1, 2020",
        }
        repo.get_markets_by_category.return_value = [assignment]
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1

    async def test_trade_executed(self) -> None:
        pipeline, repo, nws, awc, engine, executor, risk_mgr = self._make_pipeline()

        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = {
            "category": "weather",
            "metric": "temperature_high",
            "location": "Dallas",
            "threshold": 100.0,
            "threshold_unit": "F",
            "comparison": "above",
            "date_description": "June 15, 2027",
        }
        repo.get_markets_by_category.return_value = [assignment]

        # NWS returns good forecast.
        nws.fetch.return_value = FetchResult(
            source_name="nws", source_key="dallas",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={"forecast_max": 105.0, "lead_hours": 12.0},
        )

        # Engine produces high p_yes with edge.
        engine.compute.return_value = PriceEstimate(
            p_yes=0.85, confidence=0.80,
        )

        # Snapshot with market price = 0.50 (big edge).
        snapshot = MagicMock()
        snapshot.mid = 0.50
        snapshot.best_bid = 0.49
        snapshot.best_ask = 0.51
        repo.get_latest_snapshot.return_value = snapshot

        # Risk check passes.
        risk_mgr.check_new_trade_category.return_value = RiskCheck(allowed=True)

        # Open positions empty.
        repo.get_open_positions.return_value = []

        # Executor fill.
        executor.execute.return_value = PaperFill(
            side="BUY_YES", price=0.52, size_eur=10.0,
            fee_eur=0.0, slippage_applied=0.01,
        )

        # Repo returns IDs.
        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1
        repo.add_fill.return_value = 1

        summary = await pipeline.run_cycle()
        assert summary["trades_executed"] == 1
        assert summary["markets_priced"] == 1

    async def test_risk_blocked(self) -> None:
        pipeline, repo, nws, awc, engine, executor, risk_mgr = self._make_pipeline()

        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = {
            "category": "weather",
            "metric": "temperature_high",
            "location": "Dallas",
            "threshold": 100.0,
            "threshold_unit": "F",
            "comparison": "above",
            "date_description": "June 15, 2027",
        }
        repo.get_markets_by_category.return_value = [assignment]

        nws.fetch.return_value = FetchResult(
            source_name="nws", source_key="dallas",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={"forecast_max": 105.0, "lead_hours": 12.0},
        )

        engine.compute.return_value = PriceEstimate(p_yes=0.85, confidence=0.80)

        snapshot = MagicMock()
        snapshot.mid = 0.50
        snapshot.best_bid = 0.49
        snapshot.best_ask = 0.51
        repo.get_latest_snapshot.return_value = snapshot
        repo.get_open_positions.return_value = []

        risk_mgr.check_new_trade_category.return_value = RiskCheck(
            allowed=False, violations=["max_open_positions"]
        )

        summary = await pipeline.run_cycle()
        assert summary["trades_executed"] == 0
        assert summary["trades_attempted"] == 1

    async def test_exception_per_market(self) -> None:
        """Exception in one market doesn't block others."""
        pipeline, repo, nws, *_ = self._make_pipeline()

        assignment1 = MagicMock()
        assignment1.market_id = "mkt-bad"
        assignment1.contract_spec_json = {
            "category": "weather",
            "metric": "temperature_high",
            "location": "Dallas",
            "threshold": 100.0,
            "threshold_unit": "F",
            "comparison": "above",
            "date_description": "June 15, 2027",
        }
        assignment2 = MagicMock()
        assignment2.market_id = "mkt-good"
        assignment2.contract_spec_json = None  # Will be skipped cleanly.

        repo.get_markets_by_category.return_value = [assignment1, assignment2]
        nws.fetch.side_effect = RuntimeError("boom")

        summary = await pipeline.run_cycle()
        assert len(summary["errors"]) == 1
        assert summary["markets_skipped"] == 1


# ===========================================================================
# scheduler — weather wiring
# ===========================================================================


class TestSchedulerWeatherCycle:
    """StructuredTradingEngine.run_weather_cycle()."""

    async def test_no_pipeline_returns_empty(self) -> None:
        from src.app.scheduler import StructuredTradingEngine
        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(), policy=Policy(),
        )
        result = await engine.run_weather_cycle()
        assert result == {}

    async def test_delegates_to_pipeline(self) -> None:
        from src.app.scheduler import StructuredTradingEngine
        pipeline = AsyncMock()
        pipeline.run_cycle.return_value = {"trades_executed": 2}
        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(), policy=Policy(),
            weather_pipeline=pipeline,
        )
        result = await engine.run_weather_cycle()
        assert result["trades_executed"] == 2
        pipeline.run_cycle.assert_called_once()

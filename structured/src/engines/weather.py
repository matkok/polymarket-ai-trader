"""Weather pricing engine — CDF-based probability estimates per metric.

Dispatches by metric type:
- **Temperature**: Gaussian CDF
- **Precipitation**: Zero-inflated model (point-in-time ≤3d) or cumulative (>3d)
- **Snow occurrence**: Joint P(temp < 33) * P(precip > 0)
- **Hurricane**: Seasonal base rate (real model deferred to S4)
"""

from __future__ import annotations

import math
from typing import Any

import structlog
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from src.contracts.weather import WeatherContractSpec
from src.engines.base import PriceEstimate, PricingEngine
from src.sources.base import FetchResult

logger = structlog.get_logger(__name__)

_ENGINE_VERSION = "weather_v2"

# ---------------------------------------------------------------------------
# Climatology — NOAA 1991-2020 monthly precipitation normals (inches).
# ---------------------------------------------------------------------------

_PRECIP_NORMALS: dict[str, dict[int, float]] = {
    "new york": {1: 3.64, 2: 3.19, 3: 4.29, 4: 4.09, 5: 3.96, 6: 4.54,
                 7: 4.60, 8: 4.44, 9: 4.31, 10: 4.38, 11: 3.58, 12: 4.38},
    "chicago": {1: 2.05, 2: 1.95, 3: 2.67, 4: 3.37, 5: 4.07, 6: 4.14,
                7: 3.96, 8: 3.55, 9: 3.27, 10: 3.17, 11: 3.01, 12: 2.42},
    "dallas": {1: 2.56, 2: 2.81, 3: 3.56, 4: 3.66, 5: 4.87, 6: 3.99,
               7: 2.13, 8: 2.08, 9: 3.22, 10: 4.55, 11: 3.07, 12: 2.97},
    "miami": {1: 1.96, 2: 2.27, 3: 2.73, 4: 3.36, 5: 5.52, 6: 9.89,
              7: 6.50, 8: 8.90, 9: 9.84, 10: 6.34, 11: 3.26, 12: 2.14},
    "seattle": {1: 5.57, 2: 3.50, 3: 3.72, 4: 2.59, 5: 2.06, 6: 1.49,
                7: 0.60, 8: 0.83, 9: 1.63, 10: 3.48, 11: 5.90, 12: 5.62},
    "los angeles": {1: 3.12, 2: 3.80, 3: 2.43, 4: 0.91, 5: 0.26, 6: 0.06,
                    7: 0.01, 8: 0.04, 9: 0.24, 10: 0.59, 11: 1.05, 12: 2.34},
    "denver": {1: 0.51, 2: 0.49, 3: 1.28, 4: 1.82, 5: 2.16, 6: 1.76,
               7: 2.16, 8: 1.82, 9: 1.14, 10: 0.98, 11: 0.74, 12: 0.63},
    "phoenix": {1: 0.83, 2: 0.77, 3: 0.87, 4: 0.25, 5: 0.12, 6: 0.02,
                7: 0.94, 8: 0.94, 9: 0.63, 10: 0.52, 11: 0.57, 12: 0.78},
    "houston": {1: 3.68, 2: 3.00, 3: 3.36, 4: 3.60, 5: 5.15, 6: 5.86,
                7: 3.63, 8: 4.81, 9: 5.36, 10: 4.70, 11: 3.83, 12: 3.66},
    "atlanta": {1: 4.20, 2: 4.32, 3: 4.72, 4: 3.41, 5: 3.44, 6: 3.95,
                7: 5.04, 8: 3.90, 9: 3.72, 10: 3.11, 11: 3.54, 12: 3.82},
}

_DEFAULT_MONTHLY_NORMAL = 3.0  # inches, approximate US average


class WeatherEngine(PricingEngine):
    """Deterministic weather pricing engine."""

    def __init__(self, forecast_horizon_hours: float = 168.0) -> None:
        self._forecast_horizon_hours = forecast_horizon_hours

    @property
    def name(self) -> str:
        return "weather"

    @property
    def version(self) -> str:
        return _ENGINE_VERSION

    # ------------------------------------------------------------------
    # Confidence decay
    # ------------------------------------------------------------------

    def _confidence_decay(self, lead_hours: float) -> float:
        """Confidence decays exponentially with lead time.

        ``exp(-lead_hours / (2 * forecast_horizon_hours))``
        """
        return math.exp(-lead_hours / (2.0 * self._forecast_horizon_hours))

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    def _compute_temperature(
        self, spec: WeatherContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Gaussian CDF for temperature threshold markets.

        spread = 2°F base + lead-time decay factor.
        """
        lead_hours = obs.get("lead_hours", 0.0)

        if spec.metric == "temperature_high":
            forecast_val = obs.get("forecast_max")
        elif spec.metric == "temperature_low":
            forecast_val = obs.get("forecast_min")
        else:
            forecast_val = obs.get("forecast_max")

        if forecast_val is None:
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.1,
                model_details={"error": "no_forecast_value", "metric": spec.metric},
            )

        threshold = spec.threshold
        if threshold is None:
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.1,
                model_details={"error": "no_threshold"},
            )

        # Convert threshold if units differ.
        if spec.threshold_unit.upper() == "C":
            threshold = threshold * 9.0 / 5.0 + 32.0

        # Spread widens with lead time: base 2°F + 0.02°F per hour.
        spread = 2.0 + 0.02 * lead_hours

        # P(actual > threshold) for "above" comparisons.
        if spec.comparison in ("above", "at_least"):
            p_yes = 1.0 - norm.cdf(threshold, loc=forecast_val, scale=spread)
        elif spec.comparison in ("below", "at_most"):
            p_yes = norm.cdf(threshold, loc=forecast_val, scale=spread)
        else:
            # Default to "above".
            p_yes = 1.0 - norm.cdf(threshold, loc=forecast_val, scale=spread)

        confidence = self._confidence_decay(lead_hours)

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=float(confidence),
            source_confidence=obs.get("quality_score", 1.0),
            model_details={
                "metric": spec.metric,
                "forecast_val": forecast_val,
                "threshold": threshold,
                "spread": spread,
                "lead_hours": lead_hours,
                "comparison": spec.comparison,
            },
        )

    # ------------------------------------------------------------------
    # Precipitation
    # ------------------------------------------------------------------

    def _compute_precipitation(
        self, spec: WeatherContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Zero-inflated model for point-in-time precipitation markets (≤3 days).

        Uses QPF amounts when available, falls back to generic gamma.
        Confidence capped at 0.70.
        """
        lead_hours = obs.get("lead_hours", 0.0)
        precip_prob = obs.get("precip_prob_mean", 0.0)
        qpf_total = obs.get("qpf_total_inches")

        threshold = spec.threshold
        if threshold is None or threshold <= 0:
            # No threshold → just use probability of any precipitation.
            p_yes = precip_prob
            confidence = min(0.70, self._confidence_decay(lead_hours))
            return PriceEstimate(
                p_yes=float(max(0.01, min(0.99, p_yes))),
                confidence=float(confidence),
                model_details={
                    "model": "point_in_time",
                    "metric": spec.metric,
                    "precip_prob": precip_prob,
                    "lead_hours": lead_hours,
                },
            )

        # Use QPF-informed gamma when QPF data is available.
        if qpf_total is not None and qpf_total > 0:
            shape = max(1.0, qpf_total / 0.5)
            scale = qpf_total / shape
        else:
            # Fallback: generic gamma(2.0, 0.5).
            shape = 2.0
            scale = 0.5

        p_exceed = 1.0 - gamma_dist.cdf(threshold, a=shape, scale=scale)

        if spec.comparison in ("below", "at_most"):
            p_yes = 1.0 - (precip_prob * p_exceed)
        else:
            p_yes = precip_prob * p_exceed

        confidence = min(0.70, self._confidence_decay(lead_hours))

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=float(confidence),
            model_details={
                "model": "point_in_time",
                "metric": spec.metric,
                "precip_prob": precip_prob,
                "threshold": threshold,
                "p_exceed": float(p_exceed),
                "qpf_total_inches": qpf_total,
                "gamma_shape": shape,
                "gamma_scale": scale,
                "lead_hours": lead_hours,
            },
        )

    # ------------------------------------------------------------------
    # Climatology lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _get_monthly_normal(location: str, month: int) -> float:
        """Look up the NOAA monthly precipitation normal (inches).

        Falls back to ``_DEFAULT_MONTHLY_NORMAL`` for unknown cities.
        """
        key = location.strip().lower()
        # Try exact match, then substring match (e.g. "New York City" → "new york").
        normals = _PRECIP_NORMALS.get(key)
        if normals is None:
            for city, vals in _PRECIP_NORMALS.items():
                if city in key or key in city:
                    normals = vals
                    break
        if normals is None:
            return _DEFAULT_MONTHLY_NORMAL
        return normals.get(month, _DEFAULT_MONTHLY_NORMAL)

    # ------------------------------------------------------------------
    # Cumulative precipitation model (span > 3 days)
    # ------------------------------------------------------------------

    def _compute_precipitation_cumulative(
        self, spec: WeatherContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Cumulative precipitation model for multi-day contracts.

        Combines QPF forecast amounts with climatological tail estimates
        for the portion of the contract beyond NWS forecast coverage.
        """
        contract_span_days = obs.get("contract_span_days", 7.0)
        current_month = obs.get("current_month", 1)
        lead_hours = obs.get("lead_hours", 0.0)
        threshold = spec.threshold

        if threshold is None or threshold <= 0:
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.1,
                model_details={"error": "no_threshold", "model": "cumulative"},
            )

        # --- Forecast portion (QPF data) ---
        qpf_total = obs.get("qpf_total_inches", 0.0)
        qpf_coverage_hours = obs.get("qpf_coverage_hours", 0.0)
        forecast_coverage_days = qpf_coverage_hours / 24.0

        # Forecast variance: QPF has ~30% relative uncertainty.
        forecast_variance = (qpf_total * 0.3) ** 2 if qpf_total > 0 else 0.01

        # --- Climatological tail (uncovered days) ---
        remaining_days = max(0.0, contract_span_days - forecast_coverage_days)
        monthly_normal = self._get_monthly_normal(spec.location, current_month)
        import calendar
        days_in_month = calendar.monthrange(2026, current_month)[1]
        daily_clim_mean = monthly_normal / days_in_month

        tail_mean = remaining_days * daily_clim_mean
        # Precipitation has high day-to-day variability: variance ≈ (daily_mean * 0.8)².
        daily_clim_variance = (daily_clim_mean * 0.8) ** 2
        tail_variance = remaining_days * daily_clim_variance

        # --- Total distribution ---
        total_mean = qpf_total + tail_mean
        total_variance = forecast_variance + tail_variance

        # Fit a gamma distribution: shape = mean² / var, scale = var / mean.
        if total_mean <= 0 or total_variance <= 0:
            p_below = 1.0 if threshold > 0 else 0.0
        else:
            shape = total_mean ** 2 / total_variance
            scale = total_variance / total_mean
            p_below = float(gamma_dist.cdf(threshold, a=shape, scale=scale))

        if spec.comparison in ("below", "at_most"):
            p_yes = p_below
        else:
            p_yes = 1.0 - p_below

        # --- Confidence: scales with forecast coverage ---
        coverage_ratio = min(1.0, forecast_coverage_days / contract_span_days) if contract_span_days > 0 else 0.0
        decay = self._confidence_decay(lead_hours)
        confidence = min(0.85, coverage_ratio * decay)
        # Floor at 0.10 when we have at least some QPF data.
        if qpf_total > 0 and confidence < 0.10:
            confidence = 0.10

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=float(confidence),
            model_details={
                "model": "cumulative",
                "metric": spec.metric,
                "threshold": threshold,
                "comparison": spec.comparison,
                "contract_span_days": contract_span_days,
                "qpf_total_inches": qpf_total,
                "qpf_coverage_hours": qpf_coverage_hours,
                "forecast_total": qpf_total,
                "tail_mean": tail_mean,
                "total_mean": total_mean,
                "total_variance": total_variance,
                "clim_monthly_normal": monthly_normal,
                "coverage_ratio": coverage_ratio,
                "lead_hours": lead_hours,
            },
        )

    # ------------------------------------------------------------------
    # Snow occurrence
    # ------------------------------------------------------------------

    def _compute_snow_occurrence(
        self, spec: WeatherContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Joint P(temp < 33°F) * P(precip > 0) for snow occurrence.

        Confidence capped at 0.75.
        """
        lead_hours = obs.get("lead_hours", 0.0)
        forecast_min = obs.get("forecast_min")
        precip_prob = obs.get("precip_prob_max", 0.0)

        if forecast_min is None:
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.1,
                model_details={"error": "no_forecast_min"},
            )

        # P(temp < 33°F) using Gaussian.
        spread = 2.0 + 0.02 * lead_hours
        p_cold = norm.cdf(33.0, loc=forecast_min, scale=spread)

        p_yes = float(p_cold) * precip_prob
        confidence = min(0.75, self._confidence_decay(lead_hours))

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=float(confidence),
            model_details={
                "metric": "snow_occurrence",
                "forecast_min": forecast_min,
                "precip_prob": precip_prob,
                "p_cold": float(p_cold),
                "lead_hours": lead_hours,
            },
        )

    # ------------------------------------------------------------------
    # Hurricane
    # ------------------------------------------------------------------

    def _compute_hurricane(
        self, spec: WeatherContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Seasonal base rate fallback for hurricane markets.

        Real model deferred to S4.  Historical base rates:
        - Any named storm: ~0.95 per season
        - Cat 3+: ~0.45 per season
        - Cat 5: ~0.05 per season
        """
        threshold = spec.threshold  # Category level.
        if threshold is not None and threshold >= 5:
            p_yes = 0.05
        elif threshold is not None and threshold >= 3:
            p_yes = 0.45
        else:
            p_yes = 0.70

        return PriceEstimate(
            p_yes=p_yes,
            confidence=0.30,
            model_details={
                "metric": "hurricane",
                "method": "seasonal_base_rate",
                "threshold_category": threshold,
            },
        )

    # ------------------------------------------------------------------
    # Snowfall amount
    # ------------------------------------------------------------------

    def _compute_snowfall(
        self, spec: WeatherContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Model for snowfall amount thresholds.

        Uses snow occurrence probability scaled by a gamma distribution
        for snowfall amounts.
        """
        lead_hours = obs.get("lead_hours", 0.0)
        forecast_min = obs.get("forecast_min")
        precip_prob = obs.get("precip_prob_max", 0.0)

        if forecast_min is None:
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.1,
                model_details={"error": "no_forecast_min"},
            )

        # P(snow given precip): temp-based.
        spread = 2.0 + 0.02 * lead_hours
        p_cold = float(norm.cdf(33.0, loc=forecast_min, scale=spread))
        p_snow = p_cold * precip_prob

        threshold = spec.threshold
        if threshold is not None and threshold > 0:
            # Gamma for snowfall amount given it snows.
            shape = 1.5
            scale = 2.0  # inches.
            p_exceed = 1.0 - gamma_dist.cdf(threshold, a=shape, scale=scale)
            if spec.comparison in ("below", "at_most"):
                p_yes = 1.0 - (p_snow * p_exceed)
            else:
                p_yes = p_snow * float(p_exceed)
        else:
            p_yes = p_snow

        confidence = min(0.70, self._confidence_decay(lead_hours))

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=float(confidence),
            model_details={
                "metric": "snowfall",
                "p_cold": p_cold,
                "p_snow": p_snow,
                "threshold": threshold,
                "lead_hours": lead_hours,
            },
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def compute(self, spec: Any, observation: Any) -> PriceEstimate:
        """Compute a price estimate by dispatching on metric type.

        Parameters
        ----------
        spec:
            A :class:`WeatherContractSpec`.
        observation:
            A :class:`FetchResult` or dict with normalized forecast data.
        """
        if not isinstance(spec, WeatherContractSpec):
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.0,
                model_details={"error": "invalid_spec_type"},
            )

        # Accept FetchResult or raw dict.
        if isinstance(observation, FetchResult):
            obs = observation.normalized_json
        elif isinstance(observation, dict):
            obs = observation
        else:
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.0,
                model_details={"error": "invalid_observation_type"},
            )

        metric = spec.metric

        if metric in ("temperature_high", "temperature_low", "temperature"):
            return self._compute_temperature(spec, obs)
        elif metric == "precipitation":
            contract_span_days = obs.get("contract_span_days")
            if contract_span_days is not None and contract_span_days > 3:
                return self._compute_precipitation_cumulative(spec, obs)
            return self._compute_precipitation(spec, obs)
        elif metric == "snow_occurrence":
            return self._compute_snow_occurrence(spec, obs)
        elif metric == "hurricane":
            return self._compute_hurricane(spec, obs)
        elif metric == "snowfall":
            return self._compute_snowfall(spec, obs)
        else:
            logger.warning("weather_engine_unknown_metric", metric=metric)
            return PriceEstimate(
                p_yes=0.5,
                confidence=0.1,
                model_details={"error": "unknown_metric", "metric": metric},
            )

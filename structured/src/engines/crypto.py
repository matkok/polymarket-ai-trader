"""Crypto pricing engine — volatility-based CDF models for price thresholds.

Uses distance-to-threshold with estimated volatility and time remaining.
Near resolution: high-confidence override when price is clearly above/below.
"""

from __future__ import annotations

import math
from typing import Any

import structlog
from scipy.stats import norm

from src.contracts.crypto import CryptoContractSpec
from src.engines.base import PriceEstimate, PricingEngine
from src.sources.base import FetchResult

logger = structlog.get_logger(__name__)

_ENGINE_VERSION = "crypto_v1"

# Default daily volatility (as fraction, not %). These are approximate.
_DEFAULT_VOLATILITY: dict[str, float] = {
    "BTC": 0.03,    # ~3% daily
    "ETH": 0.04,    # ~4% daily
    "SOL": 0.05,    # ~5% daily
    "DOGE": 0.06,   # ~6% daily
    "ADA": 0.05,
    "XRP": 0.05,
}


class CryptoEngine(PricingEngine):
    """Deterministic crypto price threshold pricing engine."""

    def __init__(
        self,
        default_volatility_daily_pct: float = 3.0,
        near_resolution_hours: float = 1.0,
    ) -> None:
        self._default_vol = default_volatility_daily_pct / 100.0
        self._near_res_hours = near_resolution_hours

    @property
    def name(self) -> str:
        return "crypto"

    @property
    def version(self) -> str:
        return _ENGINE_VERSION

    def compute(self, spec: Any, observation: Any) -> PriceEstimate:
        """Compute a price estimate for a crypto threshold market."""
        if not isinstance(spec, CryptoContractSpec):
            return PriceEstimate(
                p_yes=0.5, confidence=0.0,
                model_details={"error": "invalid_spec_type"},
            )

        # Accept FetchResult or raw dict.
        if isinstance(observation, FetchResult):
            obs = observation.normalized_json
        elif isinstance(observation, dict):
            obs = observation
        else:
            return PriceEstimate(
                p_yes=0.5, confidence=0.0,
                model_details={"error": "invalid_observation_type"},
            )

        current_price = obs.get("price")
        if current_price is None or current_price <= 0:
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": "no_price"},
            )

        threshold = spec.threshold
        if threshold is None or threshold <= 0:
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": "no_threshold"},
            )

        # Get time remaining (hours, from observation or default).
        hours_remaining = obs.get("hours_remaining", 24 * 30)  # default 30 days
        days_remaining = max(hours_remaining / 24.0, 0.01)

        # Get volatility.
        asset = spec.asset.upper()
        daily_vol = _DEFAULT_VOLATILITY.get(asset, self._default_vol)

        # Near resolution override.
        if hours_remaining <= self._near_res_hours:
            return self._near_resolution(spec, current_price, threshold)

        # Price distance model using log-normal approximation.
        # sigma = daily_vol * sqrt(days)
        sigma = daily_vol * math.sqrt(days_remaining)

        # Use log-price for better fit.
        log_current = math.log(current_price)
        log_threshold = math.log(threshold)

        if spec.comparison in ("above", "at_least"):
            # P(price > threshold at resolution)
            p_yes = float(1.0 - norm.cdf(log_threshold, loc=log_current, scale=sigma))
        elif spec.comparison in ("below", "at_most"):
            p_yes = float(norm.cdf(log_threshold, loc=log_current, scale=sigma))
        else:
            p_yes = float(1.0 - norm.cdf(log_threshold, loc=log_current, scale=sigma))

        # Confidence: high near resolution, low far out.
        if days_remaining <= 1:
            confidence = 0.85
        elif days_remaining <= 7:
            confidence = 0.70
        elif days_remaining <= 30:
            confidence = 0.55
        else:
            confidence = 0.50

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=confidence,
            model_details={
                "asset": asset,
                "method": "log_normal_cdf",
                "current_price": current_price,
                "threshold": threshold,
                "daily_vol": daily_vol,
                "sigma": sigma,
                "days_remaining": days_remaining,
                "comparison": spec.comparison,
            },
        )

    def _near_resolution(
        self,
        spec: CryptoContractSpec,
        current_price: float,
        threshold: float,
    ) -> PriceEstimate:
        """High-confidence override when very close to resolution."""
        distance_pct = abs(current_price - threshold) / threshold

        if spec.comparison in ("above", "at_least"):
            if current_price > threshold:
                p_yes = 0.95
            elif distance_pct < 0.02:
                p_yes = 0.50  # Too close to call.
            else:
                p_yes = 0.05
        elif spec.comparison in ("below", "at_most"):
            if current_price < threshold:
                p_yes = 0.95
            elif distance_pct < 0.02:
                p_yes = 0.50
            else:
                p_yes = 0.05
        else:
            p_yes = 0.50

        return PriceEstimate(
            p_yes=p_yes,
            confidence=0.90,
            model_details={
                "asset": spec.asset,
                "method": "near_resolution",
                "current_price": current_price,
                "threshold": threshold,
                "distance_pct": distance_pct,
            },
        )

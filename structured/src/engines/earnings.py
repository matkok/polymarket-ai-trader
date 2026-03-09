"""Earnings pricing engine — SEC filing and financial metric models.

Post-filing: near-certainty resolution from XBRL data.
Pre-filing: base-rate priors with low confidence.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.contracts.earnings import EarningsContractSpec
from src.engines.base import PriceEstimate, PricingEngine
from src.sources.base import FetchResult

logger = structlog.get_logger(__name__)

_ENGINE_VERSION = "earnings_v1"


class EarningsEngine(PricingEngine):
    """Deterministic earnings/filing pricing engine."""

    def __init__(self, pre_filing_confidence: float = 0.30) -> None:
        self._pre_filing_conf = pre_filing_confidence

    @property
    def name(self) -> str:
        return "earnings"

    @property
    def version(self) -> str:
        return _ENGINE_VERSION

    def compute(self, spec: Any, observation: Any) -> PriceEstimate:
        """Compute a price estimate for an earnings/filing market."""
        if not isinstance(spec, EarningsContractSpec):
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

        metric = spec.metric
        if metric in ("eps", "revenue"):
            return self._compute_financial_metric(spec, obs)
        elif metric.startswith("filing_"):
            return self._compute_filing_existence(spec, obs)
        else:
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": f"unsupported_metric: {metric}"},
            )

    def _compute_financial_metric(
        self, spec: EarningsContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Compute probability for EPS/revenue threshold markets."""
        has_filed = obs.get("has_filed", False)
        latest_value = obs.get("latest_value")

        if not has_filed or latest_value is None:
            # Pre-filing: no actual data, use uniform prior.
            return PriceEstimate(
                p_yes=0.50,
                confidence=self._pre_filing_conf,
                model_details={
                    "method": "pre_filing_prior",
                    "metric": spec.metric,
                    "has_filed": False,
                },
            )

        # Post-filing: compare actual value to threshold.
        threshold = spec.threshold
        if threshold is None:
            return PriceEstimate(
                p_yes=0.50,
                confidence=self._pre_filing_conf,
                model_details={
                    "method": "no_threshold",
                    "metric": spec.metric,
                    "latest_value": latest_value,
                },
            )

        # Scale threshold by unit multiplier.
        effective_threshold = _scale_threshold(threshold, spec.threshold_unit)

        # Compare actual vs threshold.
        if spec.comparison in ("above", "at_least"):
            if latest_value > effective_threshold:
                p_yes = 0.99
            elif latest_value == effective_threshold:
                p_yes = 0.50
            else:
                p_yes = 0.01
        elif spec.comparison in ("below",):
            if latest_value < effective_threshold:
                p_yes = 0.99
            elif latest_value == effective_threshold:
                p_yes = 0.50
            else:
                p_yes = 0.01
        else:
            # Default: treat as "above".
            if latest_value > effective_threshold:
                p_yes = 0.99
            else:
                p_yes = 0.01

        return PriceEstimate(
            p_yes=p_yes,
            confidence=0.95,
            model_details={
                "method": "post_filing_comparison",
                "metric": spec.metric,
                "latest_value": latest_value,
                "threshold": threshold,
                "effective_threshold": effective_threshold,
                "comparison": spec.comparison,
                "has_filed": True,
            },
        )

    def _compute_filing_existence(
        self, spec: EarningsContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Compute probability for filing existence markets."""
        filing_found = obs.get("filing_found", False)

        if filing_found:
            # Filing already exists.
            return PriceEstimate(
                p_yes=0.99,
                confidence=0.95,
                model_details={
                    "method": "filing_confirmed",
                    "filing_type": obs.get("filing_type", ""),
                    "filing_date": obs.get("filing_date", ""),
                    "company": obs.get("company", ""),
                },
            )

        # Filing not yet found: base rate (most companies file on time).
        return PriceEstimate(
            p_yes=0.85,
            confidence=0.40,
            model_details={
                "method": "filing_base_rate",
                "filing_type": obs.get("filing_type", ""),
                "company": obs.get("company", ""),
                "filing_found": False,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scale_threshold(threshold: float, unit: str) -> float:
    """Scale threshold by unit multiplier."""
    unit = unit.upper()
    if unit == "B":
        return threshold * 1_000_000_000
    elif unit == "M":
        return threshold * 1_000_000
    elif unit == "K":
        return threshold * 1_000
    return threshold

"""Macro pricing engine — distribution-based CDF models per indicator.

Dispatches by indicator type:
- **CPI/PPI/PCE/Core variants:** Normal distribution on MoM changes
- **Unemployment:** Normal distribution on rate levels
- **Nonfarm payrolls:** Normal distribution on actual vs trend
- **GDP:** Quarterly growth distribution
- **Fed rate:** Binary pre/post-meeting model
"""

from __future__ import annotations

import math
from typing import Any

import structlog
from scipy.stats import norm

from src.contracts.macro import MacroContractSpec
from src.engines.base import PriceEstimate, PricingEngine
from src.sources.base import FetchResult

logger = structlog.get_logger(__name__)

_ENGINE_VERSION = "macro_v1"

# Default historical standard deviations when history is insufficient.
_DEFAULT_STD: dict[str, float] = {
    "cpi": 0.15,            # ~0.15% MoM std
    "core_cpi": 0.10,
    "pce": 0.12,
    "core_pce": 0.08,
    "ppi": 0.30,
    "unemployment": 0.10,   # ~0.1pp MoM change std
    "nonfarm_payrolls": 80.0,  # ~80K jobs std
    "gdp": 1.0,             # ~1.0% quarterly growth std
    "fed_rate": 0.25,
}

# Minimum historical data points for distribution fitting.
_MIN_HISTORY = 6


class MacroEngine(PricingEngine):
    """Deterministic macro-economic pricing engine."""

    def __init__(self, lookback_months: int = 24) -> None:
        self._lookback_months = lookback_months

    @property
    def name(self) -> str:
        return "macro"

    @property
    def version(self) -> str:
        return _ENGINE_VERSION

    # ------------------------------------------------------------------
    # CPI / PPI / PCE / Core variants (MoM change distribution)
    # ------------------------------------------------------------------

    def _compute_price_index(
        self, spec: MacroContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Normal distribution on month-over-month percentage changes."""
        history = obs.get("history", [])
        is_preliminary = obs.get("is_preliminary", False)

        # Compute MoM changes from history.
        changes = _compute_mom_changes(history)

        if len(changes) >= _MIN_HISTORY:
            mean_change = sum(changes[-3:]) / len(changes[-3:])  # Trailing 3
            std_change = _std(changes)
        else:
            mean_change = obs.get("month_over_month_pct", 0.0)
            std_change = _DEFAULT_STD.get(spec.indicator, 0.15)

        # If we already have the actual value, compare directly.
        latest_value = obs.get("latest_value")
        if latest_value is not None and spec.threshold is not None:
            return self._direct_comparison(
                spec, latest_value, is_preliminary,
            )

        threshold = spec.threshold
        if threshold is None:
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": "no_threshold"},
            )

        # CDF comparison.
        p_yes = _cdf_compare(threshold, mean_change, max(std_change, 0.01), spec.comparison)
        confidence = 0.95 if not is_preliminary else 0.75

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=confidence,
            model_details={
                "indicator": spec.indicator,
                "method": "mom_normal_cdf",
                "mean_change": mean_change,
                "std_change": std_change,
                "threshold": threshold,
                "comparison": spec.comparison,
            },
        )

    # ------------------------------------------------------------------
    # Unemployment
    # ------------------------------------------------------------------

    def _compute_unemployment(
        self, spec: MacroContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Normal distribution on rate levels with MoM change variance."""
        history = obs.get("history", [])
        is_preliminary = obs.get("is_preliminary", False)
        latest_value = obs.get("latest_value")

        if latest_value is not None and spec.threshold is not None:
            return self._direct_comparison(spec, latest_value, is_preliminary)

        threshold = spec.threshold
        if threshold is None:
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": "no_threshold"},
            )

        # Use recent value as center, std from MoM changes.
        center = latest_value if latest_value is not None else threshold
        changes = _compute_level_changes(history)
        std = _std(changes) if len(changes) >= _MIN_HISTORY else _DEFAULT_STD["unemployment"]

        p_yes = _cdf_compare(threshold, center, max(std, 0.01), spec.comparison)
        confidence = 0.70

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=confidence,
            model_details={
                "indicator": "unemployment",
                "method": "level_normal_cdf",
                "center": center,
                "std": std,
                "threshold": threshold,
            },
        )

    # ------------------------------------------------------------------
    # Nonfarm payrolls
    # ------------------------------------------------------------------

    def _compute_payrolls(
        self, spec: MacroContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Normal distribution on actual vs recent trend."""
        history = obs.get("history", [])
        is_preliminary = obs.get("is_preliminary", False)
        latest_value = obs.get("latest_value")

        if latest_value is not None and spec.threshold is not None:
            return self._direct_comparison(spec, latest_value, is_preliminary)

        threshold = spec.threshold
        if threshold is None:
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": "no_threshold"},
            )

        # Payroll thresholds are in K (thousands).
        threshold_actual = threshold
        if spec.threshold_unit == "K":
            threshold_actual = threshold  # Already in K.
        elif spec.threshold_unit == "M":
            threshold_actual = threshold * 1000

        changes = _compute_level_changes(history)
        if len(changes) >= _MIN_HISTORY:
            mean_change = sum(changes[-3:]) / len(changes[-3:])
            std_change = _std(changes)
        else:
            mean_change = 0.0
            std_change = _DEFAULT_STD["nonfarm_payrolls"]

        p_yes = _cdf_compare(threshold_actual, mean_change, max(std_change, 1.0), spec.comparison)
        confidence = 0.65  # Payrolls are noisy.

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=confidence,
            model_details={
                "indicator": "nonfarm_payrolls",
                "method": "change_normal_cdf",
                "mean_change": mean_change,
                "std_change": std_change,
                "threshold": threshold_actual,
            },
        )

    # ------------------------------------------------------------------
    # GDP
    # ------------------------------------------------------------------

    def _compute_gdp(
        self, spec: MacroContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Quarterly growth distribution from recent history."""
        history = obs.get("history", [])
        latest_value = obs.get("latest_value")

        if latest_value is not None and spec.threshold is not None:
            return self._direct_comparison(spec, latest_value, False)

        threshold = spec.threshold
        if threshold is None:
            # "Will GDP contract?" → threshold = 0.
            if spec.comparison in ("below",):
                threshold = 0.0
            else:
                return PriceEstimate(
                    p_yes=0.5, confidence=0.1,
                    model_details={"error": "no_threshold"},
                )

        changes = _compute_pct_changes(history)
        if len(changes) >= 4:
            mean_growth = sum(changes[-4:]) / len(changes[-4:])
            std_growth = _std(changes)
        else:
            mean_growth = 2.0  # Historical US average ~2%.
            std_growth = _DEFAULT_STD["gdp"]

        p_yes = _cdf_compare(threshold, mean_growth, max(std_growth, 0.1), spec.comparison)
        confidence = 0.60  # GDP has revisions.

        return PriceEstimate(
            p_yes=float(max(0.01, min(0.99, p_yes))),
            confidence=confidence,
            model_details={
                "indicator": "gdp",
                "method": "growth_normal_cdf",
                "mean_growth": mean_growth,
                "std_growth": std_growth,
                "threshold": threshold,
            },
        )

    # ------------------------------------------------------------------
    # Fed rate
    # ------------------------------------------------------------------

    def _compute_fed_rate(
        self, spec: MacroContractSpec, obs: dict[str, Any]
    ) -> PriceEstimate:
        """Binary model for Fed rate decisions."""
        latest_value = obs.get("latest_value")
        previous_value = obs.get("previous_value")

        # If we can see the actual rate decision (post-meeting):
        if latest_value is not None and previous_value is not None:
            actual_change = latest_value - previous_value
            if spec.comparison == "raise":
                p_yes = 0.99 if actual_change > 0 else 0.01
            elif spec.comparison == "cut":
                p_yes = 0.99 if actual_change < 0 else 0.01
            elif spec.comparison == "hold":
                p_yes = 0.99 if actual_change == 0 else 0.01
            else:
                p_yes = 0.5
            return PriceEstimate(
                p_yes=p_yes,
                confidence=0.95,
                model_details={
                    "indicator": "fed_rate",
                    "method": "post_meeting",
                    "actual_change": actual_change,
                    "comparison": spec.comparison,
                },
            )

        # Pre-meeting: use simple historical pattern.
        if spec.comparison == "hold":
            p_yes = 0.60  # Most meetings are holds.
        elif spec.comparison == "raise":
            p_yes = 0.20
        elif spec.comparison == "cut":
            p_yes = 0.20
        else:
            p_yes = 0.50

        return PriceEstimate(
            p_yes=p_yes,
            confidence=0.50,
            model_details={
                "indicator": "fed_rate",
                "method": "pre_meeting_base_rate",
                "comparison": spec.comparison,
            },
        )

    # ------------------------------------------------------------------
    # Direct comparison (post-release)
    # ------------------------------------------------------------------

    def _direct_comparison(
        self,
        spec: MacroContractSpec,
        actual_value: float,
        is_preliminary: bool,
    ) -> PriceEstimate:
        """Compare actual released value to threshold."""
        threshold = spec.threshold
        if threshold is None:
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": "no_threshold"},
            )

        if spec.comparison in ("above", "at_least"):
            p_yes = 0.99 if actual_value >= threshold else 0.01
        elif spec.comparison in ("below", "at_most"):
            p_yes = 0.99 if actual_value <= threshold else 0.01
        else:
            p_yes = 0.99 if actual_value >= threshold else 0.01

        confidence = 0.90 if is_preliminary else 0.95

        return PriceEstimate(
            p_yes=p_yes,
            confidence=confidence,
            model_details={
                "indicator": spec.indicator,
                "method": "direct_comparison",
                "actual_value": actual_value,
                "threshold": threshold,
                "is_preliminary": is_preliminary,
            },
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def compute(self, spec: Any, observation: Any) -> PriceEstimate:
        """Compute a price estimate by dispatching on indicator type."""
        if not isinstance(spec, MacroContractSpec):
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

        indicator = spec.indicator

        if indicator in ("cpi", "core_cpi", "pce", "core_pce", "ppi", "retail_sales", "housing_starts"):
            return self._compute_price_index(spec, obs)
        elif indicator == "unemployment":
            return self._compute_unemployment(spec, obs)
        elif indicator == "nonfarm_payrolls":
            return self._compute_payrolls(spec, obs)
        elif indicator == "gdp":
            return self._compute_gdp(spec, obs)
        elif indicator == "fed_rate":
            return self._compute_fed_rate(spec, obs)
        else:
            logger.warning("macro_engine_unknown_indicator", indicator=indicator)
            return PriceEstimate(
                p_yes=0.5, confidence=0.1,
                model_details={"error": "unknown_indicator", "indicator": indicator},
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cdf_compare(threshold: float, mean: float, std: float, comparison: str) -> float:
    """Compute P(X > threshold) or P(X < threshold) using normal CDF."""
    if comparison in ("above", "at_least"):
        return float(1.0 - norm.cdf(threshold, loc=mean, scale=std))
    elif comparison in ("below", "at_most"):
        return float(norm.cdf(threshold, loc=mean, scale=std))
    else:
        # Default to P(X > threshold).
        return float(1.0 - norm.cdf(threshold, loc=mean, scale=std))


def _compute_mom_changes(history: list[dict[str, Any]]) -> list[float]:
    """Compute month-over-month percentage changes from history."""
    if len(history) < 2:
        return []
    # History is typically newest-first; reverse to oldest-first.
    values = [h["value"] for h in reversed(history) if "value" in h]
    changes = []
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            changes.append((values[i] - values[i - 1]) / values[i - 1] * 100.0)
    return changes


def _compute_level_changes(history: list[dict[str, Any]]) -> list[float]:
    """Compute simple level changes (difference) from history."""
    if len(history) < 2:
        return []
    values = [h["value"] for h in reversed(history) if "value" in h]
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def _compute_pct_changes(history: list[dict[str, Any]]) -> list[float]:
    """Compute percentage changes for GDP-style quarterly data."""
    if len(history) < 2:
        return []
    values = [h["value"] for h in reversed(history) if "value" in h]
    changes = []
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            changes.append((values[i] - values[i - 1]) / values[i - 1] * 100.0)
    return changes


def _std(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)

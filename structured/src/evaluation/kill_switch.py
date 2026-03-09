"""Category kill switch — auto-disable poorly calibrated categories.

Monitors Brier score and drawdown per category. Disables trading when
performance falls below configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KillSwitchConfig:
    """Configuration for the category kill switch."""

    max_brier_score: float = 0.35
    max_daily_drawdown_eur: float = 500.0
    min_predictions_before_check: int = 10
    cooldown_hours: float = 24.0


@dataclass
class CategoryHealth:
    """Health status for a single category."""

    category: str
    is_active: bool = True
    brier_score: float | None = None
    daily_drawdown_eur: float = 0.0
    n_predictions: int = 0
    kill_reasons: list[str] = field(default_factory=list)


class KillSwitch:
    """Monitors category health and auto-disables on poor performance."""

    def __init__(self, config: KillSwitchConfig | None = None) -> None:
        self._config = config or KillSwitchConfig()
        self._disabled_categories: set[str] = set()

    @property
    def disabled_categories(self) -> set[str]:
        """Return the set of currently disabled categories."""
        return set(self._disabled_categories)

    def is_enabled(self, category: str) -> bool:
        """Check if a category is currently enabled for trading."""
        return category not in self._disabled_categories

    def check_category(
        self,
        category: str,
        brier_score: float | None = None,
        daily_drawdown_eur: float = 0.0,
        n_predictions: int = 0,
    ) -> CategoryHealth:
        """Evaluate category health and disable if necessary."""
        health = CategoryHealth(
            category=category,
            brier_score=brier_score,
            daily_drawdown_eur=daily_drawdown_eur,
            n_predictions=n_predictions,
        )

        # Check Brier score (only if enough predictions).
        if (
            brier_score is not None
            and n_predictions >= self._config.min_predictions_before_check
            and brier_score > self._config.max_brier_score
        ):
            health.kill_reasons.append(
                f"brier_score={brier_score:.3f} > max={self._config.max_brier_score:.3f}"
            )

        # Check daily drawdown.
        if daily_drawdown_eur > self._config.max_daily_drawdown_eur:
            health.kill_reasons.append(
                f"daily_drawdown={daily_drawdown_eur:.2f} > max={self._config.max_daily_drawdown_eur:.2f}"
            )

        if health.kill_reasons:
            health.is_active = False
            self._disabled_categories.add(category)
            logger.warning(
                "kill_switch_triggered",
                category=category,
                reasons=health.kill_reasons,
            )
        else:
            health.is_active = True

        return health

    def reset_category(self, category: str) -> None:
        """Re-enable a previously disabled category."""
        self._disabled_categories.discard(category)
        logger.info("kill_switch_reset", category=category)

    def reset_all(self) -> None:
        """Re-enable all disabled categories (e.g., daily reset)."""
        self._disabled_categories.clear()
        logger.info("kill_switch_reset_all")

    def status(self) -> dict[str, Any]:
        """Return current kill switch status."""
        return {
            "disabled_categories": sorted(self._disabled_categories),
            "config": {
                "max_brier_score": self._config.max_brier_score,
                "max_daily_drawdown_eur": self._config.max_daily_drawdown_eur,
                "min_predictions_before_check": self._config.min_predictions_before_check,
            },
        }

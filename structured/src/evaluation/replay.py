"""Historical replay engine for backtesting pricing engines.

Replays resolved markets against stored source observations to evaluate
engine accuracy without live data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.engines.base import PriceEstimate, PricingEngine
from src.evaluation.calibration import PredictionOutcome, evaluate_category

logger = structlog.get_logger(__name__)


@dataclass
class ReplayMarket:
    """A historical market for replay."""

    market_id: str
    category: str
    spec_json: dict[str, Any]
    observation_json: dict[str, Any]
    outcome: float  # 1.0 = yes, 0.0 = no
    resolved_ts: datetime | None = None


@dataclass
class ReplayResult:
    """Results from a replay run."""

    category: str
    engine_version: str
    n_markets: int
    n_errors: int
    predictions: list[dict[str, Any]] = field(default_factory=list)
    brier_score: float | None = None
    log_loss: float | None = None

    def to_backtest_dict(
        self,
        ts_start: datetime,
        ts_end: datetime,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert to dict for BacktestRun DB row."""
        return {
            "category": self.category,
            "engine_version": self.engine_version,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "config_json": config or {},
            "results_json": {
                "n_markets": self.n_markets,
                "n_errors": self.n_errors,
                "brier_score": self.brier_score,
                "log_loss": self.log_loss,
            },
        }


class ReplayEngine:
    """Replay historical markets through a pricing engine."""

    def __init__(self, engine: PricingEngine, spec_factory: Any = None) -> None:
        self._engine = engine
        self._spec_factory = spec_factory

    def replay(self, markets: list[ReplayMarket]) -> ReplayResult:
        """Replay a list of historical markets.

        Returns aggregated calibration metrics.
        """
        if not markets:
            return ReplayResult(
                category="",
                engine_version=self._engine.version,
                n_markets=0,
                n_errors=0,
            )

        category = markets[0].category
        pairs: list[PredictionOutcome] = []
        predictions: list[dict[str, Any]] = []
        n_errors = 0

        for market in markets:
            spec = self._build_spec(market)
            if spec is None:
                n_errors += 1
                continue

            estimate = self._engine.compute(spec, market.observation_json)

            pairs.append(PredictionOutcome(
                market_id=market.market_id,
                category=market.category,
                engine_version=self._engine.version,
                p_yes=estimate.p_yes,
                outcome=market.outcome,
            ))

            predictions.append({
                "market_id": market.market_id,
                "p_yes": estimate.p_yes,
                "confidence": estimate.confidence,
                "outcome": market.outcome,
                "error": (estimate.p_yes - market.outcome) ** 2,
            })

        cal = evaluate_category(pairs, category, self._engine.version)

        return ReplayResult(
            category=category,
            engine_version=self._engine.version,
            n_markets=len(markets),
            n_errors=n_errors,
            predictions=predictions,
            brier_score=cal.brier_score,
            log_loss=cal.log_loss,
        )

    def _build_spec(self, market: ReplayMarket) -> Any:
        """Build a contract spec from stored JSON."""
        if self._spec_factory is not None:
            try:
                return self._spec_factory(market.spec_json)
            except Exception:
                logger.exception(
                    "replay_spec_build_error",
                    market_id=market.market_id,
                )
                return None

        # Return raw dict as fallback (engine must handle it).
        return market.spec_json

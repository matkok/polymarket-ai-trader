"""Weather trading pipeline — full cycle orchestrator.

Drives the loop: market fetch → NWS forecast → engine pricing → sizing →
risk check → paper execution → persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from src.config.policy import Policy, policy_version_hash
from src.contracts.weather import WeatherContractSpec
from src.db.repository import Repository
from src.engines.base import PriceEstimate
from src.engines.weather import WeatherEngine
from src.execution.paper_executor import PaperExecutor
from src.portfolio.risk_manager import RiskManager
from src.portfolio.sizing import SizingInput, compute_size
from src.sources.awc import AWCAdapter
from src.sources.nws import NWSAdapter
from src.trading.date_resolver import resolve_date_range

logger = structlog.get_logger(__name__)

_NEAR_RESOLUTION_HOURS = 24


class WeatherPipeline:
    """Full weather trading cycle orchestrator."""

    def __init__(
        self,
        repo: Repository,
        nws: NWSAdapter,
        awc: AWCAdapter,
        engine: WeatherEngine,
        executor: PaperExecutor,
        risk_manager: RiskManager,
        policy: Policy,
    ) -> None:
        self.repo = repo
        self.nws = nws
        self.awc = awc
        self.engine = engine
        self.executor = executor
        self.risk_manager = risk_manager
        self.policy = policy
        self._weather_policy = policy.for_category("weather")
        self._policy_hash = policy_version_hash(policy)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> dict[str, Any]:
        """Run one full weather trading cycle.

        Returns a summary dict with counts and actions taken.
        """
        logger.info("weather_cycle_start")
        now = datetime.now(timezone.utc)

        assignments = await self.repo.get_markets_by_category("weather")
        summary: dict[str, Any] = {
            "markets_found": len(assignments),
            "markets_skipped": 0,
            "markets_priced": 0,
            "trades_attempted": 0,
            "trades_executed": 0,
            "errors": [],
        }

        for assignment in assignments:
            try:
                result = await self._process_market(assignment, now)
                if result == "skipped":
                    summary["markets_skipped"] += 1
                elif result == "priced":
                    summary["markets_priced"] += 1
                elif result == "traded":
                    summary["markets_priced"] += 1
                    summary["trades_attempted"] += 1
                    summary["trades_executed"] += 1
                elif result == "no_trade":
                    summary["markets_priced"] += 1
                    summary["trades_attempted"] += 1
            except Exception as exc:
                logger.exception(
                    "weather_market_error",
                    market_id=assignment.market_id,
                )
                summary["errors"].append(
                    {"market_id": assignment.market_id, "error": str(exc)}
                )

        logger.info("weather_cycle_done", **summary)
        return summary

    # ------------------------------------------------------------------
    # Per-market processing
    # ------------------------------------------------------------------

    async def _process_market(
        self, assignment: Any, now: datetime
    ) -> str:
        """Process a single weather market.

        Returns one of: ``"skipped"``, ``"priced"``, ``"traded"``, ``"no_trade"``.
        """
        market_id = assignment.market_id
        spec_json = assignment.contract_spec_json
        if not spec_json:
            logger.debug("weather_skip_no_spec", market_id=market_id)
            return "skipped"

        # Reconstruct WeatherContractSpec from stored JSON.
        spec = _reconstruct_spec(spec_json)
        if spec is None:
            logger.debug("weather_skip_bad_spec", market_id=market_id)
            return "skipped"

        # Entry guard: skip if we already hold a position or recently closed.
        existing = await self.repo.get_position(market_id)
        if existing:
            if existing.status == "open":
                return "skipped"
            if existing.status == "closed":
                hours_since_close = (
                    (now - existing.last_update_ts_utc).total_seconds() / 3600
                )
                if hours_since_close < self._weather_policy.reentry_cooldown_hours:
                    return "skipped"

        # Horizon filter: skip markets too far from resolution.
        market = await self.repo.get_market(market_id)
        if market and market.resolution_time_utc:
            hours_to_res = (market.resolution_time_utc - now).total_seconds() / 3600
            if hours_to_res > self._weather_policy.max_hours_to_resolution:
                return "skipped"
            if hours_to_res < self._weather_policy.min_hours_to_resolution:
                return "skipped"

        # Resolve date range.
        date_range = resolve_date_range(spec.date_description)
        contract_span_days: float | None = None
        if date_range is not None:
            start, end = date_range
            # Skip if the event is already past.
            if end < now:
                logger.debug("weather_skip_past", market_id=market_id)
                return "skipped"

            # Populate spec date fields for NWS filtering.
            spec.date_start = start.isoformat()
            spec.date_end = end.isoformat()

            # Compute contract span for cumulative model dispatch.
            contract_span_days = (end - start).total_seconds() / 86400

            # Check if near resolution — use observations if within 24h.
            hours_to_end = (end - now).total_seconds() / 3600
        else:
            hours_to_end = None

        # Fetch forecast from NWS.
        forecast = await self.nws.fetch(spec)
        if not forecast.ok:
            logger.warning(
                "weather_forecast_error",
                market_id=market_id, error=forecast.error,
            )
            return "skipped"

        # Persist observation.
        await self.repo.add_source_observation(
            forecast.to_observation_dict("weather")
        )

        # Near resolution: also fetch METAR observations.
        observation_override = None
        if hours_to_end is not None and hours_to_end <= _NEAR_RESOLUTION_HOURS:
            observation_override = await self._check_observations(spec, market_id)

        # Inject contract span and current month into normalized_json for engine.
        if contract_span_days is not None:
            forecast.normalized_json["contract_span_days"] = contract_span_days
            forecast.normalized_json["current_month"] = now.month

        # Compute price estimate.
        if observation_override is not None:
            estimate = observation_override
        else:
            estimate = self.engine.compute(spec, forecast)

        # Persist engine price.
        snapshot = await self.repo.get_latest_snapshot(market_id)
        p_market = snapshot.mid if snapshot else None

        await self.repo.add_engine_price(
            estimate.to_engine_price_dict(
                market_id=market_id,
                category="weather",
                engine_version=self.engine.version,
                ts_utc=now,
                p_market=p_market,
            )
        )

        # Need market price to size.
        if p_market is None:
            logger.debug("weather_skip_no_market_price", market_id=market_id)
            return "priced"

        # Size the trade.
        sizing_input = SizingInput(
            p_consensus=estimate.p_yes,
            p_market=p_market,
            confidence=estimate.confidence,
            disagreement=0.0,  # Single engine, no disagreement.
            best_bid=snapshot.best_bid if snapshot and snapshot.best_bid else 0.0,
            best_ask=snapshot.best_ask if snapshot and snapshot.best_ask else 0.0,
        )
        sizing_result = compute_size(sizing_input, self._weather_policy)

        if sizing_result.skip_reason:
            logger.info(
                "weather_sizing_skip",
                market_id=market_id,
                reason=sizing_result.skip_reason,
            )
            return "priced"

        # Risk check.
        all_positions = await self.repo.get_open_positions()
        category_assignments = await self.repo.get_markets_by_category("weather")
        category_market_ids = {a.market_id for a in category_assignments}

        risk_check = self.risk_manager.check_new_trade_category(
            size_eur=sizing_result.clamped_size_eur,
            market_id=market_id,
            category="weather",
            all_positions=all_positions,
            category_market_ids=category_market_ids,
            daily_realized_pnl=0.0,  # TODO: track per-category PnL.
        )

        if not risk_check.allowed:
            logger.info(
                "weather_risk_blocked",
                market_id=market_id,
                violations=risk_check.violations,
            )
            return "no_trade"

        # Execute paper trade.
        best_bid = snapshot.best_bid if snapshot and snapshot.best_bid else p_market - 0.01
        best_ask = snapshot.best_ask if snapshot and snapshot.best_ask else p_market + 0.01

        fill = self.executor.execute(
            side=sizing_result.side,
            size_eur=sizing_result.clamped_size_eur,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        # Persist decision.
        decision_id = await self.repo.add_decision({
            "market_id": market_id,
            "ts_utc": now,
            "action": sizing_result.side,
            "size_eur": sizing_result.clamped_size_eur,
            "reason_json": {
                "engine": self.engine.version,
                "p_yes": estimate.p_yes,
                "p_market": p_market,
                "edge": sizing_result.edge,
                "confidence": estimate.confidence,
                "metric": spec.metric,
                "location": spec.location,
                "threshold": spec.threshold,
                "comparison": spec.comparison,
                "contract_span_days": contract_span_days,
                "model_details": estimate.model_details,
            },
            "policy_version": self._policy_hash,
        })

        # Persist order.
        order_id = await self.repo.add_order({
            "decision_id": decision_id,
            "market_id": market_id,
            "side": fill.side,
            "size_eur": fill.size_eur,
            "limit_price_ref": fill.price,
            "status": "filled",
        })

        # Persist fill.
        await self.repo.add_fill({
            "order_id": order_id,
            "ts_utc": now,
            "price": fill.price,
            "size_eur": fill.size_eur,
            "fee_eur": fill.fee_eur,
        })

        # Upsert position.
        await self.repo.upsert_position({
            "market_id": market_id,
            "side": fill.side,
            "size_eur": fill.size_eur,
            "avg_entry_price": fill.price,
            "status": "open",
            "opened_ts_utc": now,
            "last_update_ts_utc": now,
        })

        logger.info(
            "weather_trade_executed",
            market_id=market_id,
            side=fill.side,
            size_eur=fill.size_eur,
            price=fill.price,
            p_yes=estimate.p_yes,
            p_market=p_market,
            edge=sizing_result.edge,
        )

        return "traded"

    # ------------------------------------------------------------------
    # Observation check near resolution
    # ------------------------------------------------------------------

    async def _check_observations(
        self, spec: WeatherContractSpec, market_id: str
    ) -> PriceEstimate | None:
        """Fetch METAR observations and override forecast if conclusive.

        Returns a high-confidence PriceEstimate if observations confirm
        the outcome, otherwise ``None`` (falls through to forecast).
        """
        obs_result = await self.awc.fetch(spec)
        if not obs_result.ok:
            return None

        norm = obs_result.normalized_json
        metric = spec.metric

        if metric in ("temperature_high", "temperature_low", "temperature"):
            observed_temp = norm.get("temperature_f")
            if observed_temp is None or spec.threshold is None:
                return None

            threshold = spec.threshold
            if spec.threshold_unit.upper() == "C":
                threshold = threshold * 9.0 / 5.0 + 32.0

            if spec.comparison in ("above", "at_least"):
                if observed_temp > threshold:
                    p_yes = 0.95
                else:
                    p_yes = 0.05
            elif spec.comparison in ("below", "at_most"):
                if observed_temp < threshold:
                    p_yes = 0.95
                else:
                    p_yes = 0.05
            else:
                return None

            return PriceEstimate(
                p_yes=p_yes,
                confidence=0.95,
                source_confidence=1.0,
                model_details={
                    "method": "observation_override",
                    "observed_temp_f": observed_temp,
                    "threshold": threshold,
                },
            )

        if metric == "snow_occurrence":
            has_snow = norm.get("has_snow", False)
            p_yes = 0.95 if has_snow else 0.05
            return PriceEstimate(
                p_yes=p_yes,
                confidence=0.95,
                source_confidence=1.0,
                model_details={"method": "observation_override", "has_snow": has_snow},
            )

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconstruct_spec(spec_json: dict[str, Any]) -> WeatherContractSpec | None:
    """Reconstruct a WeatherContractSpec from stored JSON."""
    try:
        return WeatherContractSpec(
            category=spec_json.get("category", "weather"),
            metric=spec_json.get("metric", ""),
            location=spec_json.get("location", ""),
            threshold=spec_json.get("threshold"),
            threshold_unit=spec_json.get("threshold_unit", ""),
            comparison=spec_json.get("comparison", ""),
            date_start=spec_json.get("date_start", ""),
            date_end=spec_json.get("date_end", ""),
            date_description=spec_json.get("date_description", ""),
            nws_station_ids=spec_json.get("nws_station_ids", []),
        )
    except Exception:
        logger.exception("spec_reconstruct_error", spec_json=spec_json)
        return None

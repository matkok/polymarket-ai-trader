"""Macro trading pipeline — full cycle orchestrator.

Drives the loop: market fetch → source data → engine pricing → sizing →
risk check → paper execution → persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from src.config.policy import Policy, policy_version_hash
from src.contracts.macro import MacroContractSpec
from src.db.repository import Repository
from src.engines.base import PriceEstimate
from src.engines.macro import MacroEngine
from src.execution.paper_executor import PaperExecutor
from src.portfolio.risk_manager import RiskManager
from src.portfolio.sizing import SizingInput, compute_size
from src.sources.base import FetchResult
from src.sources.bls import BLSAdapter
from src.sources.fred import FREDAdapter
from src.sources.release_calendar import is_near_release, resolve_release_date

logger = structlog.get_logger(__name__)

# Indicators that BLS covers natively.
_BLS_INDICATORS = {"cpi", "core_cpi", "unemployment", "nonfarm_payrolls", "ppi"}


class MacroPipeline:
    """Full macro trading cycle orchestrator."""

    def __init__(
        self,
        repo: Repository,
        bls: BLSAdapter,
        fred: FREDAdapter,
        engine: MacroEngine,
        executor: PaperExecutor,
        risk_manager: RiskManager,
        policy: Policy,
    ) -> None:
        self.repo = repo
        self.bls = bls
        self.fred = fred
        self.engine = engine
        self.executor = executor
        self.risk_manager = risk_manager
        self.policy = policy
        self._macro_policy = policy.for_category("macro")
        self._policy_hash = policy_version_hash(policy)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> dict[str, Any]:
        """Run one full macro trading cycle."""
        logger.info("macro_cycle_start")
        now = datetime.now(timezone.utc)

        assignments = await self.repo.get_markets_by_category("macro")
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
            except Exception:
                logger.exception(
                    "macro_market_error",
                    market_id=assignment.market_id,
                )
                summary["errors"].append(
                    {"market_id": assignment.market_id, "error": "exception"}
                )

        logger.info("macro_cycle_done", **summary)
        return summary

    # ------------------------------------------------------------------
    # Per-market processing
    # ------------------------------------------------------------------

    async def _process_market(self, assignment: Any, now: datetime) -> str:
        """Process a single macro market.

        Returns one of: ``"skipped"``, ``"priced"``, ``"traded"``, ``"no_trade"``.
        """
        market_id = assignment.market_id
        spec_json = assignment.contract_spec_json
        if not spec_json:
            return "skipped"

        spec = _reconstruct_spec(spec_json)
        if spec is None:
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
                if hours_since_close < self._macro_policy.reentry_cooldown_hours:
                    return "skipped"

        # Horizon filter: skip markets too far from resolution.
        market = await self.repo.get_market(market_id)
        if market and market.resolution_time_utc:
            hours_to_res = (market.resolution_time_utc - now).total_seconds() / 3600
            if hours_to_res > self._macro_policy.max_hours_to_resolution:
                return "skipped"
            if hours_to_res < self._macro_policy.min_hours_to_resolution:
                return "skipped"

        # Choose source adapter.
        if spec.indicator in _BLS_INDICATORS:
            source = self.bls
        else:
            source = self.fred

        # Fetch data.
        fetch_result = await source.fetch(spec)
        if not fetch_result.ok:
            logger.warning(
                "macro_source_error",
                market_id=market_id,
                source=source.name,
                error=fetch_result.error,
            )
            return "skipped"

        # Persist observation.
        await self.repo.add_source_observation(
            fetch_result.to_observation_dict("macro")
        )

        # Compute price estimate.
        estimate = self.engine.compute(spec, fetch_result)

        # Near-release confidence boost.
        near_release_hours = (
            self.policy.categories.get("macro", None)
        )
        boost_hours = 24.0
        if near_release_hours and near_release_hours.engine_params:
            boost_hours = float(
                near_release_hours.engine_params.get("near_release_hours", 24.0)
            )

        if is_near_release(spec.indicator, spec.release_period, hours_threshold=boost_hours, now=now):
            estimate = PriceEstimate(
                p_yes=estimate.p_yes,
                confidence=max(estimate.confidence, 0.85),
                source_confidence=estimate.source_confidence,
                model_details={**estimate.model_details, "near_release_boost": True},
            )

        # Persist engine price.
        snapshot = await self.repo.get_latest_snapshot(market_id)
        p_market = snapshot.mid if snapshot else None

        await self.repo.add_engine_price(
            estimate.to_engine_price_dict(
                market_id=market_id,
                category="macro",
                engine_version=self.engine.version,
                ts_utc=now,
                p_market=p_market,
            )
        )

        if p_market is None:
            return "priced"

        # Size the trade.
        sizing_input = SizingInput(
            p_consensus=estimate.p_yes,
            p_market=p_market,
            confidence=estimate.confidence,
            disagreement=0.0,
            best_bid=snapshot.best_bid if snapshot and snapshot.best_bid else 0.0,
            best_ask=snapshot.best_ask if snapshot and snapshot.best_ask else 0.0,
        )
        sizing_result = compute_size(sizing_input, self._macro_policy)

        if sizing_result.skip_reason:
            logger.info(
                "macro_sizing_skip",
                market_id=market_id,
                reason=sizing_result.skip_reason,
            )
            return "priced"

        # Risk check.
        all_positions = await self.repo.get_open_positions()
        category_assignments = await self.repo.get_markets_by_category("macro")
        category_market_ids = {a.market_id for a in category_assignments}

        risk_check = self.risk_manager.check_new_trade_category(
            size_eur=sizing_result.clamped_size_eur,
            market_id=market_id,
            category="macro",
            all_positions=all_positions,
            category_market_ids=category_market_ids,
            daily_realized_pnl=0.0,
        )

        if not risk_check.allowed:
            logger.info(
                "macro_risk_blocked",
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
            "macro_trade_executed",
            market_id=market_id,
            side=fill.side,
            size_eur=fill.size_eur,
            price=fill.price,
        )

        return "traded"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconstruct_spec(spec_json: dict[str, Any]) -> MacroContractSpec | None:
    """Reconstruct a MacroContractSpec from stored JSON."""
    try:
        return MacroContractSpec(
            category=spec_json.get("category", "macro"),
            indicator=spec_json.get("indicator", ""),
            threshold=spec_json.get("threshold"),
            threshold_unit=spec_json.get("threshold_unit", ""),
            comparison=spec_json.get("comparison", ""),
            release_period=spec_json.get("release_period", ""),
            release_date=spec_json.get("release_date", ""),
            bls_series_id=spec_json.get("bls_series_id", ""),
            fred_series_id=spec_json.get("fred_series_id", ""),
        )
    except Exception:
        logger.exception("macro_spec_reconstruct_error", spec_json=spec_json)
        return None

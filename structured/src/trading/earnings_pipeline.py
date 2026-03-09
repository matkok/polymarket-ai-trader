"""Earnings trading pipeline — full cycle orchestrator.

Drives the loop: market fetch → EDGAR data → engine pricing → sizing →
risk check → paper execution → persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from src.config.policy import Policy, policy_version_hash
from src.contracts.earnings import EarningsContractSpec
from src.db.repository import Repository
from src.engines.earnings import EarningsEngine
from src.execution.paper_executor import PaperExecutor
from src.portfolio.risk_manager import RiskManager
from src.portfolio.sizing import SizingInput, compute_size
from src.sources.edgar import EDGARAdapter

logger = structlog.get_logger(__name__)


class EarningsPipeline:
    """Full earnings trading cycle orchestrator."""

    def __init__(
        self,
        repo: Repository,
        edgar: EDGARAdapter,
        engine: EarningsEngine,
        executor: PaperExecutor,
        risk_manager: RiskManager,
        policy: Policy,
    ) -> None:
        self.repo = repo
        self.edgar = edgar
        self.engine = engine
        self.executor = executor
        self.risk_manager = risk_manager
        self.policy = policy
        self._earnings_policy = policy.for_category("earnings")
        self._policy_hash = policy_version_hash(policy)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> dict[str, Any]:
        """Run one full earnings trading cycle."""
        logger.info("earnings_cycle_start")
        now = datetime.now(timezone.utc)

        assignments = await self.repo.get_markets_by_category("earnings")
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
                    "earnings_market_error",
                    market_id=assignment.market_id,
                )
                summary["errors"].append(
                    {"market_id": assignment.market_id, "error": "exception"}
                )

        logger.info("earnings_cycle_done", **summary)
        return summary

    # ------------------------------------------------------------------
    # Per-market processing
    # ------------------------------------------------------------------

    async def _process_market(self, assignment: Any, now: datetime) -> str:
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
                if hours_since_close < self._earnings_policy.reentry_cooldown_hours:
                    return "skipped"

        # Horizon filter: skip markets too far from resolution.
        market = await self.repo.get_market(market_id)
        if market and market.resolution_time_utc:
            hours_to_res = (market.resolution_time_utc - now).total_seconds() / 3600
            if hours_to_res > self._earnings_policy.max_hours_to_resolution:
                return "skipped"
            if hours_to_res < self._earnings_policy.min_hours_to_resolution:
                return "skipped"

        # Fetch data from EDGAR.
        fetch_result = await self.edgar.fetch(spec)
        if not fetch_result.ok:
            logger.warning(
                "earnings_source_error",
                market_id=market_id,
                error=fetch_result.error,
            )
            return "skipped"

        # Persist observation.
        await self.repo.add_source_observation(
            fetch_result.to_observation_dict("earnings")
        )

        # Compute price estimate.
        estimate = self.engine.compute(spec, fetch_result)

        # Persist engine price.
        snapshot = await self.repo.get_latest_snapshot(market_id)
        p_market = snapshot.mid if snapshot else None

        await self.repo.add_engine_price(
            estimate.to_engine_price_dict(
                market_id=market_id,
                category="earnings",
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
        sizing_result = compute_size(sizing_input, self._earnings_policy)

        if sizing_result.skip_reason:
            logger.info(
                "earnings_sizing_skip",
                market_id=market_id,
                reason=sizing_result.skip_reason,
            )
            return "priced"

        # Risk check.
        all_positions = await self.repo.get_open_positions()
        category_assignments = await self.repo.get_markets_by_category("earnings")
        category_market_ids = {a.market_id for a in category_assignments}

        risk_check = self.risk_manager.check_new_trade_category(
            size_eur=sizing_result.clamped_size_eur,
            market_id=market_id,
            category="earnings",
            all_positions=all_positions,
            category_market_ids=category_market_ids,
            daily_realized_pnl=0.0,
        )

        if not risk_check.allowed:
            logger.info(
                "earnings_risk_blocked",
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

        order_id = await self.repo.add_order({
            "decision_id": decision_id,
            "market_id": market_id,
            "side": fill.side,
            "size_eur": fill.size_eur,
            "limit_price_ref": fill.price,
            "status": "filled",
        })

        await self.repo.add_fill({
            "order_id": order_id,
            "ts_utc": now,
            "price": fill.price,
            "size_eur": fill.size_eur,
            "fee_eur": fill.fee_eur,
        })

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
            "earnings_trade_executed",
            market_id=market_id,
            side=fill.side,
            size_eur=fill.size_eur,
            price=fill.price,
        )

        return "traded"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconstruct_spec(spec_json: dict[str, Any]) -> EarningsContractSpec | None:
    try:
        return EarningsContractSpec(
            category=spec_json.get("category", "earnings"),
            company=spec_json.get("company", ""),
            ticker=spec_json.get("ticker", ""),
            metric=spec_json.get("metric", ""),
            threshold=spec_json.get("threshold"),
            threshold_unit=spec_json.get("threshold_unit", "USD"),
            comparison=spec_json.get("comparison", ""),
            filing_type=spec_json.get("filing_type", ""),
            fiscal_period=spec_json.get("fiscal_period", ""),
            cik=spec_json.get("cik", ""),
        )
    except Exception:
        logger.exception("earnings_spec_reconstruct_error", spec_json=spec_json)
        return None

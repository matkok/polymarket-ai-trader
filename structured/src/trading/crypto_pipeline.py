"""Crypto trading pipeline — full cycle orchestrator.

Drives the loop: market fetch → exchange data → engine pricing → sizing →
risk check → paper execution → persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from src.config.policy import Policy, policy_version_hash
from src.contracts.crypto import CryptoContractSpec
from src.db.repository import Repository
from src.engines.crypto import CryptoEngine
from src.execution.paper_executor import PaperExecutor
from src.portfolio.risk_manager import RiskManager
from src.portfolio.sizing import SizingInput, compute_size
from src.sources.exchange_router import ExchangeRouter

logger = structlog.get_logger(__name__)


class CryptoPipeline:
    """Full crypto trading cycle orchestrator."""

    def __init__(
        self,
        repo: Repository,
        exchange_router: ExchangeRouter,
        engine: CryptoEngine,
        executor: PaperExecutor,
        risk_manager: RiskManager,
        policy: Policy,
    ) -> None:
        self.repo = repo
        self.exchange_router = exchange_router
        self.engine = engine
        self.executor = executor
        self.risk_manager = risk_manager
        self.policy = policy
        self._crypto_policy = policy.for_category("crypto")
        self._policy_hash = policy_version_hash(policy)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> dict[str, Any]:
        """Run one full crypto trading cycle."""
        logger.info("crypto_cycle_start")
        now = datetime.now(timezone.utc)

        # Build asset direction map for coherence guard.
        direction_map = await self._build_direction_map()

        assignments = await self.repo.get_markets_by_category("crypto")
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
                result = await self._process_market(assignment, now, direction_map)
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
                    "crypto_market_error",
                    market_id=assignment.market_id,
                )
                summary["errors"].append(
                    {"market_id": assignment.market_id, "error": "exception"}
                )

        logger.info("crypto_cycle_done", **summary)
        return summary

    async def _build_direction_map(self) -> dict[tuple[str, str], str]:
        """Build (asset, expiry_date) → side map from open crypto positions."""
        direction_map: dict[tuple[str, str], str] = {}
        positions = await self.repo.get_open_positions()
        for pos in positions:
            assignment = await self.repo.get_assignment(pos.market_id)
            if not assignment or assignment.category != "crypto":
                continue
            spec_json = assignment.contract_spec_json
            if not spec_json:
                continue
            asset = spec_json.get("asset", "")
            if not asset:
                continue
            market = await self.repo.get_market(pos.market_id)
            expiry_date = _extract_expiry_date(market)
            if expiry_date:
                direction_map[(asset, expiry_date)] = pos.side
        return direction_map

    # ------------------------------------------------------------------
    # Per-market processing
    # ------------------------------------------------------------------

    async def _process_market(
        self,
        assignment: Any,
        now: datetime,
        direction_map: dict[tuple[str, str], str] | None = None,
    ) -> str:
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
                if hours_since_close < self._crypto_policy.reentry_cooldown_hours:
                    return "skipped"

        # Horizon filter: skip markets too far from resolution.
        market = await self.repo.get_market(market_id)
        if market and market.resolution_time_utc:
            hours_to_res = (market.resolution_time_utc - now).total_seconds() / 3600
            if hours_to_res > self._crypto_policy.max_hours_to_resolution:
                return "skipped"
            if hours_to_res < self._crypto_policy.min_hours_to_resolution:
                return "skipped"

        # Fetch current price from exchange.
        fetch_result = await self.exchange_router.fetch(spec)
        if not fetch_result.ok:
            logger.warning(
                "crypto_source_error",
                market_id=market_id,
                error=fetch_result.error,
            )
            return "skipped"

        # Persist observation.
        await self.repo.add_source_observation(
            fetch_result.to_observation_dict("crypto")
        )

        # Compute price estimate.
        estimate = self.engine.compute(spec, fetch_result)

        # Persist engine price.
        snapshot = await self.repo.get_latest_snapshot(market_id)
        p_market = snapshot.mid if snapshot else None

        await self.repo.add_engine_price(
            estimate.to_engine_price_dict(
                market_id=market_id,
                category="crypto",
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
        sizing_result = compute_size(sizing_input, self._crypto_policy)

        if sizing_result.skip_reason:
            logger.info(
                "crypto_sizing_skip",
                market_id=market_id,
                reason=sizing_result.skip_reason,
            )
            return "priced"

        # Asset coherence guard: block opposite-direction trades on same asset+expiry.
        if direction_map and spec.asset:
            expiry_date = _extract_expiry_date(market)
            if expiry_date:
                existing_side = direction_map.get((spec.asset, expiry_date))
                if existing_side and existing_side != sizing_result.side:
                    logger.info(
                        "crypto_coherence_blocked",
                        market_id=market_id,
                        asset=spec.asset,
                        expiry=expiry_date,
                        existing_side=existing_side,
                        new_side=sizing_result.side,
                    )
                    return "priced"

        # Risk check.
        all_positions = await self.repo.get_open_positions()
        category_assignments = await self.repo.get_markets_by_category("crypto")
        category_market_ids = {a.market_id for a in category_assignments}

        risk_check = self.risk_manager.check_new_trade_category(
            size_eur=sizing_result.clamped_size_eur,
            market_id=market_id,
            category="crypto",
            all_positions=all_positions,
            category_market_ids=category_market_ids,
            daily_realized_pnl=0.0,
        )

        if not risk_check.allowed:
            logger.info(
                "crypto_risk_blocked",
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
            "crypto_trade_executed",
            market_id=market_id,
            side=fill.side,
            size_eur=fill.size_eur,
            price=fill.price,
        )

        return "traded"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_expiry_date(market: Any) -> str | None:
    """Extract expiry date string (YYYY-MM-DD) from a market's resolution time."""
    if market and market.resolution_time_utc:
        return market.resolution_time_utc.strftime("%Y-%m-%d")
    return None


def _reconstruct_spec(spec_json: dict[str, Any]) -> CryptoContractSpec | None:
    try:
        return CryptoContractSpec(
            category=spec_json.get("category", "crypto"),
            asset=spec_json.get("asset", ""),
            threshold=spec_json.get("threshold"),
            threshold_unit=spec_json.get("threshold_unit", "USD"),
            comparison=spec_json.get("comparison", ""),
            exchange=spec_json.get("exchange", ""),
            reference_price=spec_json.get("reference_price", "last_trade"),
            resolution_timestamp=spec_json.get("resolution_timestamp", ""),
            date_description=spec_json.get("date_description", ""),
        )
    except Exception:
        logger.exception("crypto_spec_reconstruct_error", spec_json=spec_json)
        return None

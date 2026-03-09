"""Core structured trading engine that orchestrates all operations.

The :class:`StructuredTradingEngine` drives market ingestion, classification,
weather trading cycles, and daily resets.  It is intended to be invoked by
APScheduler jobs configured in ``main.py``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog

from src.config.policy import Policy, policy_version_hash
from src.contracts.registry import ParserRegistry, classify_markets_batch
from src.db.repository import Repository
from src.evaluation.calibration import PredictionOutcome, evaluate_category
from src.evaluation.kill_switch import KillSwitch
from src.execution.fills import calculate_realized_pnl, calculate_unrealized_pnl
from src.execution.paper_executor import PaperExecutor
from src.polymarket.gamma_client import GammaClient
from src.portfolio.lifecycle import LifecycleAction, PositionLifecycle

if TYPE_CHECKING:
    from src.trading.crypto_pipeline import CryptoPipeline
    from src.trading.earnings_pipeline import EarningsPipeline
    from src.trading.macro_pipeline import MacroPipeline
    from src.trading.weather_pipeline import WeatherPipeline

logger = structlog.get_logger(__name__)


class StructuredTradingEngine:
    """Core structured trading engine."""

    def __init__(
        self,
        repo: Repository,
        gamma_client: GammaClient,
        policy: Policy,
        parser_registry: ParserRegistry | None = None,
        weather_pipeline: WeatherPipeline | None = None,
        macro_pipeline: MacroPipeline | None = None,
        crypto_pipeline: CryptoPipeline | None = None,
        earnings_pipeline: EarningsPipeline | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.repo = repo
        self.gamma = gamma_client
        self.policy = policy
        self.policy_hash = policy_version_hash(policy)
        self.daily_realized_pnl = 0.0
        self._registry = parser_registry or ParserRegistry()
        self._weather_pipeline = weather_pipeline
        self._macro_pipeline = macro_pipeline
        self._crypto_pipeline = crypto_pipeline
        self._earnings_pipeline = earnings_pipeline
        self._kill_switch = kill_switch

    # ------------------------------------------------------------------
    # Market ingestion
    # ------------------------------------------------------------------

    async def ingest_markets(self) -> int:
        """Fetch markets from Gamma API and upsert to DB.

        Returns the count of markets ingested.
        """
        logger.info("market_ingestion_start")
        gamma_markets = await self.gamma.get_all_active_markets()

        market_dicts: list[dict] = []
        snapshot_dicts: list[dict] = []
        for gm in gamma_markets:
            if not gm.condition_id:
                continue

            # Parse resolution time from end_date_iso.
            resolution_time = None
            if gm.end_date_iso:
                try:
                    resolution_time = datetime.fromisoformat(
                        gm.end_date_iso.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            now = datetime.now(timezone.utc)
            market_dicts.append(
                {
                    "market_id": gm.condition_id,
                    "question": gm.question,
                    "rules_text": gm.description or None,
                    "category": gm.category or None,
                    "resolution_time_utc": resolution_time,
                    "status": "active" if gm.active and not gm.closed else "closed",
                    "updated_ts_utc": now,
                }
            )

            # Prepare snapshot (insert after markets are upserted).
            bid, ask = gm.best_bid_ask()
            mid = (bid + ask) / 2 if bid is not None and ask is not None else None
            snapshot_dicts.append(
                {
                    "market_id": gm.condition_id,
                    "ts_utc": now,
                    "best_bid": bid,
                    "best_ask": ask,
                    "mid": mid,
                    "liquidity": gm.liquidity,
                    "volume": gm.volume,
                }
            )

        # Deduplicate by market_id (Gamma API can return the same market
        # on multiple pages).  Keep the last occurrence for each id.
        seen: dict[str, int] = {}
        for idx, md in enumerate(market_dicts):
            seen[md["market_id"]] = idx
        unique_idxs = set(seen.values())
        market_dicts = [market_dicts[i] for i in sorted(unique_idxs)]
        snapshot_dicts = [snapshot_dicts[i] for i in sorted(unique_idxs)]

        # Upsert markets first (FK parent), then insert snapshots.
        if market_dicts:
            await self.repo.bulk_upsert_markets(market_dicts)

        if snapshot_dicts:
            await self.repo.bulk_add_snapshots(snapshot_dicts)

        logger.info(
            "market_ingestion_done",
            markets=len(market_dicts),
            snapshots=len(snapshot_dicts),
        )
        return len(market_dicts)

    # ------------------------------------------------------------------
    # Market classification
    # ------------------------------------------------------------------

    async def classify_markets(self) -> int:
        """Classify unparsed markets via the contract parser registry.

        Returns the count of newly classified (matched) markets.
        """
        return await classify_markets_batch(self.repo, self._registry)

    # ------------------------------------------------------------------
    # Weather cycle
    # ------------------------------------------------------------------

    async def run_weather_cycle(self) -> dict:
        """Run the weather trading pipeline if configured."""
        if self._kill_switch and not self._kill_switch.is_enabled("weather"):
            logger.warning("weather_cycle_killed")
            return {"killed": True}
        if self._weather_pipeline is None:
            logger.debug("weather_pipeline_not_configured")
            return {}
        return await self._weather_pipeline.run_cycle()

    # ------------------------------------------------------------------
    # Macro cycle
    # ------------------------------------------------------------------

    async def run_macro_cycle(self) -> dict:
        """Run the macro trading pipeline if configured."""
        if self._kill_switch and not self._kill_switch.is_enabled("macro"):
            logger.warning("macro_cycle_killed")
            return {"killed": True}
        if self._macro_pipeline is None:
            logger.debug("macro_pipeline_not_configured")
            return {}
        return await self._macro_pipeline.run_cycle()

    # ------------------------------------------------------------------
    # Crypto cycle
    # ------------------------------------------------------------------

    async def run_crypto_cycle(self) -> dict:
        """Run the crypto trading pipeline if configured."""
        if self._kill_switch and not self._kill_switch.is_enabled("crypto"):
            logger.warning("crypto_cycle_killed")
            return {"killed": True}
        if self._crypto_pipeline is None:
            logger.debug("crypto_pipeline_not_configured")
            return {}
        return await self._crypto_pipeline.run_cycle()

    # ------------------------------------------------------------------
    # Earnings cycle
    # ------------------------------------------------------------------

    async def run_earnings_cycle(self) -> dict:
        """Run the earnings trading pipeline if configured."""
        if self._kill_switch and not self._kill_switch.is_enabled("earnings"):
            logger.warning("earnings_cycle_killed")
            return {"killed": True}
        if self._earnings_pipeline is None:
            logger.debug("earnings_pipeline_not_configured")
            return {}
        return await self._earnings_pipeline.run_cycle()

    # ------------------------------------------------------------------
    # Position review
    # ------------------------------------------------------------------

    async def review_open_positions(self) -> dict:
        """Review all open positions and apply lifecycle decisions."""
        logger.info("position_review_start")
        now = datetime.now(timezone.utc)

        positions = await self.repo.get_open_positions()
        summary: dict[str, int] = {
            "reviewed": 0,
            "held": 0,
            "closed": 0,
            "reduced": 0,
            "errors": 0,
        }

        for position in positions:
            try:
                result = await self._review_position(position, now)
                summary["reviewed"] += 1
                if result == "held":
                    summary["held"] += 1
                elif result == "closed":
                    summary["closed"] += 1
                elif result == "reduced":
                    summary["reduced"] += 1
            except Exception:
                logger.exception(
                    "position_review_error",
                    market_id=position.market_id,
                )
                summary["errors"] += 1

        logger.info("position_review_done", **summary)
        return summary

    async def _review_position(self, position: Any, now: datetime) -> str:
        """Review a single position. Returns ``held``, ``closed``, or ``reduced``."""
        market_id = position.market_id

        market = await self.repo.get_market(market_id)
        snapshot = await self.repo.get_latest_snapshot(market_id)
        engine_price = await self.repo.get_latest_engine_price(market_id)

        # Compute hours to resolution.
        hours_to_resolution: float | None = None
        if market and market.resolution_time_utc:
            hours_to_resolution = (
                (market.resolution_time_utc - now).total_seconds() / 3600
            )

        # Determine category for this position.
        assignment = await self.repo.get_assignment(market_id)
        category = assignment.category if assignment else None
        cat_policy = (
            self.policy.for_category(category)
            if category
            else self.policy
        )

        lifecycle = PositionLifecycle(cat_policy)

        # Engine staleness check.
        engine_fresh = False
        if engine_price and engine_price.ts_utc:
            engine_age_hours = (now - engine_price.ts_utc).total_seconds() / 3600
            engine_fresh = engine_age_hours < cat_policy.engine_stale_hours

        if engine_fresh and engine_price:
            decision = lifecycle.evaluate_with_engine(
                position=position,
                current_snapshot=snapshot,
                entry_snapshot_mid=position.avg_entry_price,
                hours_to_resolution=hours_to_resolution,
                engine_p_yes=engine_price.p_yes,
                engine_confidence=engine_price.confidence,
            )
        else:
            decision = lifecycle.evaluate(
                position=position,
                current_snapshot=snapshot,
                entry_snapshot_mid=position.avg_entry_price,
                hours_to_resolution=hours_to_resolution,
            )

        # Execute decision.
        if decision.action == LifecycleAction.CLOSE:
            return await self._execute_close(
                position, snapshot, decision, cat_policy, now,
            )

        if decision.action == LifecycleAction.REDUCE:
            # Reduce cooldown: at most one reduce per 60 minutes.
            if position.last_update_ts_utc:
                minutes_since_update = (
                    (now - position.last_update_ts_utc).total_seconds() / 60
                )
                if minutes_since_update < 60:
                    logger.info(
                        "reduce_cooldown_skip",
                        market_id=market_id,
                        minutes_since_update=round(minutes_since_update, 1),
                    )
                    return "held"
            return await self._execute_reduce(
                position, snapshot, decision, cat_policy, now,
            )

        return "held"

    async def _execute_close(
        self,
        position: Any,
        snapshot: Any,
        decision: Any,
        cat_policy: Policy,
        now: datetime,
    ) -> str:
        """Execute a CLOSE lifecycle decision."""
        market_id = position.market_id

        # Require real market data for closing.
        if not snapshot or snapshot.best_bid is None or snapshot.best_ask is None:
            logger.warning(
                "lifecycle_close_skip_no_book", market_id=market_id,
            )
            return "held"

        executor = PaperExecutor(policy=cat_policy)
        fill = executor.execute(
            side="SELL",
            size_eur=position.size_eur,
            best_bid=snapshot.best_bid,
            best_ask=snapshot.best_ask,
        )

        realized = calculate_realized_pnl(
            side=position.side,
            size_eur=position.size_eur,
            avg_entry_price=position.avg_entry_price,
            exit_price=fill.price,
            fee_eur=fill.fee_eur,
        )

        # PnL loss bounds invariant: loss cannot exceed size + fees.
        fee_buffer = fill.fee_eur * 2
        if realized < -(position.size_eur + fee_buffer):
            logger.critical(
                "pnl_loss_bounds_violation",
                market_id=market_id,
                realized=realized,
                size_eur=position.size_eur,
                fee_buffer=fee_buffer,
            )

        decision_id = await self.repo.add_decision({
            "market_id": market_id,
            "ts_utc": now,
            "action": "CLOSE",
            "size_eur": position.size_eur,
            "reason_json": {
                "reasons": decision.reasons,
                "p_market": snapshot.mid,
            },
            "policy_version": self.policy_hash,
        })

        order_id = await self.repo.add_order({
            "decision_id": decision_id,
            "market_id": market_id,
            "side": "SELL",
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
            "side": position.side,
            "size_eur": position.size_eur,
            "avg_entry_price": position.avg_entry_price,
            "status": "closed",
            "realized_pnl": (position.realized_pnl or 0) + realized,
            "last_update_ts_utc": now,
            "opened_ts_utc": position.opened_ts_utc,
        })

        self.daily_realized_pnl += realized

        logger.info(
            "lifecycle_close",
            market_id=market_id,
            realized_pnl=realized,
            exit_price=fill.price,
            reasons=decision.reasons,
        )
        return "closed"

    async def _execute_reduce(
        self,
        position: Any,
        snapshot: Any,
        decision: Any,
        cat_policy: Policy,
        now: datetime,
    ) -> str:
        """Execute a REDUCE lifecycle decision."""
        market_id = position.market_id

        if not snapshot or snapshot.best_bid is None or snapshot.best_ask is None:
            logger.warning(
                "lifecycle_reduce_skip_no_book", market_id=market_id,
            )
            return "held"

        # Zeno guard: close entirely if position is dust.
        if position.size_eur < cat_policy.dust_position_eur:
            return await self._execute_close(
                position, snapshot, decision, cat_policy, now,
            )

        reduce_size = position.size_eur * cat_policy.reduce_fraction
        reduce_size = min(reduce_size, position.size_eur)  # defensive clamp
        remaining_size = position.size_eur - reduce_size
        if remaining_size < 0:
            logger.critical(
                "negative_remaining_size",
                market_id=market_id,
                remaining=remaining_size,
            )
            return "held"

        executor = PaperExecutor(policy=cat_policy)
        fill = executor.execute(
            side="SELL",
            size_eur=reduce_size,
            best_bid=snapshot.best_bid,
            best_ask=snapshot.best_ask,
        )

        realized = calculate_realized_pnl(
            side=position.side,
            size_eur=reduce_size,
            avg_entry_price=position.avg_entry_price,
            exit_price=fill.price,
            fee_eur=fill.fee_eur,
        )

        # PnL loss bounds invariant: loss cannot exceed size + fees.
        fee_buffer = fill.fee_eur * 2
        if realized < -(reduce_size + fee_buffer):
            logger.critical(
                "pnl_loss_bounds_violation",
                market_id=market_id,
                realized=realized,
                size_eur=reduce_size,
                fee_buffer=fee_buffer,
            )

        decision_id = await self.repo.add_decision({
            "market_id": market_id,
            "ts_utc": now,
            "action": "REDUCE",
            "size_eur": reduce_size,
            "reason_json": {
                "reasons": decision.reasons,
                "p_market": snapshot.mid,
            },
            "policy_version": self.policy_hash,
        })

        order_id = await self.repo.add_order({
            "decision_id": decision_id,
            "market_id": market_id,
            "side": "SELL",
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
            "side": position.side,
            "size_eur": remaining_size,
            "avg_entry_price": position.avg_entry_price,
            "status": "open",
            "realized_pnl": (position.realized_pnl or 0) + realized,
            "last_update_ts_utc": now,
            "opened_ts_utc": position.opened_ts_utc,
        })

        self.daily_realized_pnl += realized

        logger.info(
            "lifecycle_reduce",
            market_id=market_id,
            reduced_size=reduce_size,
            remaining_size=remaining_size,
            realized_pnl=realized,
            reasons=decision.reasons,
        )
        return "reduced"

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Calibration cycle (S6)
    # ------------------------------------------------------------------

    _ACTIVE_CATEGORIES = ("weather", "macro", "crypto", "earnings")

    async def run_calibration_cycle(self) -> dict:
        """Compute calibration metrics for recently resolved markets.

        Returns a summary dict with per-category Brier scores.
        """
        logger.info("calibration_cycle_start")
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        try:
            resolutions = await self.repo.get_resolutions_since(yesterday)
        except Exception:
            logger.exception("calibration_fetch_resolutions_error")
            return {}

        if not resolutions:
            logger.info("calibration_no_resolutions")
            return {"resolutions": 0}

        # Build market_id -> (category, outcome) mapping.
        market_ids = [r.market_id for r in resolutions]
        resolution_map: dict[str, tuple[str, float]] = {}
        for r in resolutions:
            try:
                assignment = await self.repo.get_assignment(r.market_id)
                if assignment is None:
                    continue
                outcome = 1.0 if r.outcome == "yes" else 0.0
                resolution_map[r.market_id] = (assignment.category, outcome)
            except Exception:
                logger.exception(
                    "calibration_assignment_error", market_id=r.market_id
                )

        if not resolution_map:
            logger.info("calibration_no_assigned_resolutions")
            return {"resolutions": len(resolutions), "assigned": 0}

        # Fetch latest engine prices for resolved markets.
        try:
            engine_prices = await self.repo.get_engine_prices_for_markets(
                list(resolution_map.keys())
            )
        except Exception:
            logger.exception("calibration_fetch_engine_prices_error")
            return {"resolutions": len(resolutions), "assigned": len(resolution_map)}

        # Group into PredictionOutcome lists by (category, engine_version).
        groups: dict[tuple[str, str], list[PredictionOutcome]] = defaultdict(list)
        for ep in engine_prices:
            cat_outcome = resolution_map.get(ep.market_id)
            if cat_outcome is None:
                continue
            category, outcome = cat_outcome
            groups[(category, ep.engine_version)].append(
                PredictionOutcome(
                    market_id=ep.market_id,
                    category=category,
                    engine_version=ep.engine_version,
                    p_yes=ep.p_yes,
                    outcome=outcome,
                )
            )

        # Evaluate and persist per group.
        today = date.today()
        summary: dict[str, float | None] = {}
        for (category, engine_version), pairs in groups.items():
            cal_result = evaluate_category(pairs, category, engine_version)
            try:
                await self.repo.add_calibration_stat(
                    cal_result.to_stat_dict(today)
                )
            except Exception:
                logger.exception(
                    "calibration_persist_error",
                    category=category,
                    engine_version=engine_version,
                )

            summary[category] = cal_result.brier_score

            # Feed into kill switch.
            if self._kill_switch and cal_result.brier_score is not None:
                self._kill_switch.check_category(
                    category=category,
                    brier_score=cal_result.brier_score,
                    n_predictions=cal_result.n_predictions,
                )

        logger.info("calibration_cycle_done", summary=summary)
        return {
            "resolutions": len(resolutions),
            "assigned": len(resolution_map),
            "calibrated": len(groups),
            "brier_scores": summary,
        }

    # ------------------------------------------------------------------
    # Daily PnL aggregation (S6)
    # ------------------------------------------------------------------

    async def aggregate_daily_pnl(self) -> dict:
        """Aggregate realized + unrealized PnL per category for today.

        Returns a summary dict with per-category PnL.
        """
        logger.info("daily_pnl_aggregation_start")
        today = date.today()
        summary: dict[str, dict] = {}

        for category in self._ACTIVE_CATEGORIES:
            realized = 0.0
            unrealized = 0.0
            trades_opened = 0
            trades_closed = 0

            try:
                assignments = await self.repo.get_markets_by_category(category)
            except Exception:
                logger.exception(
                    "daily_pnl_assignments_error", category=category
                )
                continue

            for assignment in assignments:
                try:
                    position = await self.repo.get_position(
                        assignment.market_id
                    )
                except Exception:
                    logger.exception(
                        "daily_pnl_position_error",
                        market_id=assignment.market_id,
                    )
                    continue

                if position is None:
                    continue

                if position.status == "closed":
                    realized += position.realized_pnl or 0.0
                    trades_closed += 1
                elif position.status == "open":
                    trades_opened += 1
                    snap = await self.repo.get_latest_snapshot(
                        assignment.market_id
                    )
                    if snap and snap.mid is not None:
                        unrealized += calculate_unrealized_pnl(
                            side=position.side,
                            size_eur=position.size_eur,
                            avg_entry_price=position.avg_entry_price,
                            current_price=snap.mid,
                        )

            try:
                await self.repo.add_category_pnl_daily(
                    {
                        "category": category,
                        "pnl_date": today,
                        "realized_pnl_eur": realized,
                        "unrealized_pnl_eur": unrealized,
                        "trades_opened": trades_opened,
                        "trades_closed": trades_closed,
                    }
                )
            except Exception:
                logger.exception(
                    "daily_pnl_persist_error", category=category
                )

            summary[category] = {
                "realized": realized,
                "unrealized": unrealized,
                "trades_opened": trades_opened,
                "trades_closed": trades_closed,
            }

        logger.info("daily_pnl_aggregation_done", summary=summary)
        return summary

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each day)."""
        self.daily_realized_pnl = 0.0
        if self._kill_switch:
            self._kill_switch.reset_all()
        logger.info("daily_reset")

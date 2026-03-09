"""Position lifecycle evaluation.

Determines whether an open position should be held, reduced, or closed
based on current market conditions and time-to-resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import structlog

from src.aggregation.aggregator import AggregationResult
from src.config.policy import Policy
from src.db.models import MarketSnapshot, Position


class LifecycleAction(str, Enum):
    """Actions that can be taken on an open position."""

    HOLD = "HOLD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    ADD = "ADD"


@dataclass
class LifecycleDecision:
    """Result of a lifecycle evaluation for a single position."""

    action: LifecycleAction
    reasons: list[str] = field(default_factory=list)


class PositionLifecycle:
    """Evaluate open positions and recommend lifecycle actions."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.logger = structlog.get_logger(__name__)

    def evaluate(
        self,
        position: Position,
        current_snapshot: MarketSnapshot | None,
        entry_snapshot_mid: float,
        hours_to_resolution: float | None,
    ) -> LifecycleDecision:
        """Evaluate what to do with an open position.

        Parameters
        ----------
        position:
            The open position to evaluate.
        current_snapshot:
            Latest market snapshot, or ``None`` if unavailable.
        entry_snapshot_mid:
            The mid price at time of entry.
        hours_to_resolution:
            Hours remaining until market resolves, or ``None`` if unknown.
        """
        reasons: list[str] = []

        # CLOSE if no current snapshot (market data unavailable).
        if current_snapshot is None:
            reasons.append("no current market snapshot available")
            self.logger.info(
                "lifecycle_close",
                market_id=position.market_id,
                reasons=reasons,
            )
            return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=reasons)

        current_mid = current_snapshot.mid
        if current_mid is None:
            reasons.append("current snapshot has no mid price")
            self.logger.info(
                "lifecycle_close",
                market_id=position.market_id,
                reasons=reasons,
            )
            return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=reasons)

        # CLOSE if approaching resolution.
        if (
            hours_to_resolution is not None
            and hours_to_resolution < self.policy.min_hours_to_resolution
        ):
            reasons.append(
                f"hours_to_resolution {hours_to_resolution:.1f} "
                f"< minimum {self.policy.min_hours_to_resolution}"
            )
            self.logger.info(
                "lifecycle_close",
                market_id=position.market_id,
                reasons=reasons,
            )
            return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=reasons)

        # Check mid movement against position.
        stop_loss_threshold = 2.0 * self.policy.edge_threshold
        if position.side == "BUY_YES":
            mid_move_against = entry_snapshot_mid - current_mid
        else:
            mid_move_against = current_mid - entry_snapshot_mid

        # CLOSE if mid moved against us by more than 2x edge_threshold.
        if mid_move_against > stop_loss_threshold:
            reasons.append(
                f"mid moved against position by {mid_move_against:.4f} "
                f"> stop loss {stop_loss_threshold:.4f}"
            )
            self.logger.info(
                "lifecycle_close",
                market_id=position.market_id,
                reasons=reasons,
            )
            return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=reasons)

        # REDUCE if unrealized PnL < -3% of position size.
        partial_stop_frac = 0.03
        if entry_snapshot_mid > 0:
            if position.side == "BUY_YES":
                unrealized_pnl = (
                    position.size_eur
                    * (current_mid - entry_snapshot_mid)
                    / entry_snapshot_mid
                )
            else:
                unrealized_pnl = (
                    position.size_eur
                    * (entry_snapshot_mid - current_mid)
                    / entry_snapshot_mid
                )

            loss_threshold = -partial_stop_frac * position.size_eur
            if unrealized_pnl < loss_threshold:
                reasons.append(
                    f"unrealized PnL {unrealized_pnl:.2f} "
                    f"< partial stop {loss_threshold:.2f}"
                )
                self.logger.info(
                    "lifecycle_reduce",
                    market_id=position.market_id,
                    reasons=reasons,
                )
                return LifecycleDecision(
                    action=LifecycleAction.REDUCE, reasons=reasons
                )

        return LifecycleDecision(action=LifecycleAction.HOLD, reasons=[])

    def evaluate_with_aggregation(
        self,
        position: Position,
        current_snapshot: MarketSnapshot | None,
        entry_snapshot_mid: float,
        hours_to_resolution: float | None,
        current_agg: AggregationResult,
        prior_agg: AggregationResult | None,
    ) -> LifecycleDecision:
        """Evaluate an open position using fresh aggregation results.

        Runs deterministic checks first, then applies aggregation-based
        logic (edge flip, ambiguity veto, take-profit, disagreement).
        """
        # 1. Run deterministic checks (no snapshot, approaching resolution, stop loss).
        det = self.evaluate(position, current_snapshot, entry_snapshot_mid, hours_to_resolution)
        if det.action != LifecycleAction.HOLD:
            return det

        # 1b. Minimum hold time — skip aggregation-based exits for young positions.
        if position.opened_ts_utc:
            hold_minutes = (datetime.now(timezone.utc) - position.opened_ts_utc).total_seconds() / 60
            if hold_minutes < self.policy.min_hold_minutes:
                self.logger.info(
                    "lifecycle_hold_minimum",
                    market_id=position.market_id,
                    hold_minutes=round(hold_minutes),
                    min_required=self.policy.min_hold_minutes,
                )
                return LifecycleDecision(action=LifecycleAction.HOLD, reasons=["min_hold_not_met"])

        # 2. Edge flip: consensus implies opposite side vs entry.
        if position.side == "BUY_YES":
            entry_implies_yes = True
        else:
            entry_implies_yes = False

        consensus_implies_yes = current_agg.p_consensus > 0.50

        if entry_implies_yes != consensus_implies_yes:
            reason = (
                f"edge flip: consensus {current_agg.p_consensus:.4f} "
                f"implies {'YES' if consensus_implies_yes else 'NO'} "
                f"but position is {position.side}"
            )
            self.logger.info("lifecycle_close", market_id=position.market_id, reasons=[reason])
            return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=[reason])

        # 3. Ambiguity veto — require consecutive vetoes to close.
        if current_agg.veto or not current_agg.trade_allowed:
            if prior_agg is not None and (prior_agg.veto or not prior_agg.trade_allowed):
                reason = "consecutive aggregation vetoes"
                self.logger.info("lifecycle_close", market_id=position.market_id, reasons=[reason])
                return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=[reason])
            else:
                self.logger.info(
                    "lifecycle_veto_first_strike",
                    market_id=position.market_id,
                    veto_reasons=current_agg.veto_reasons,
                )
                return LifecycleDecision(action=LifecycleAction.HOLD, reasons=["first_veto_hold"])

        # 4. Take-profit: edge nearly gone and confidence dropped.
        if current_snapshot and current_snapshot.mid is not None:
            edge_remaining = abs(current_snapshot.mid - current_agg.p_consensus)
            if edge_remaining < self.policy.take_profit_band and prior_agg is not None:
                confidence_drop = prior_agg.confidence - current_agg.confidence
                if confidence_drop >= self.policy.confidence_drop_threshold:
                    reason = (
                        f"take-profit: edge remaining {edge_remaining:.4f} "
                        f"< band {self.policy.take_profit_band}, "
                        f"confidence dropped {confidence_drop:.4f}"
                    )
                    self.logger.info("lifecycle_close", market_id=position.market_id, reasons=[reason])
                    return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=[reason])

        # 5. Disagreement increase → block.
        if current_agg.disagreement >= self.policy.disagreement_block_threshold:
            reason = (
                f"disagreement {current_agg.disagreement:.4f} "
                f">= block threshold {self.policy.disagreement_block_threshold}"
            )
            self.logger.info("lifecycle_close", market_id=position.market_id, reasons=[reason])
            return LifecycleDecision(action=LifecycleAction.CLOSE, reasons=[reason])

        # 6. Disagreement increase → partial reduce.
        if (
            prior_agg is not None
            and current_agg.disagreement > prior_agg.disagreement
            and current_agg.disagreement > self.policy.disagreement_size_penalty_start
        ):
            reason = (
                f"disagreement increased {prior_agg.disagreement:.4f} -> "
                f"{current_agg.disagreement:.4f}, above penalty start "
                f"{self.policy.disagreement_size_penalty_start}"
            )
            self.logger.info("lifecycle_reduce", market_id=position.market_id, reasons=[reason])
            return LifecycleDecision(action=LifecycleAction.REDUCE, reasons=[reason])

        # 7. Otherwise hold.
        return LifecycleDecision(action=LifecycleAction.HOLD, reasons=[])

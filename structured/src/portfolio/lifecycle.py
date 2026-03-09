"""Position lifecycle evaluation.

Determines whether an open position should be held, reduced, or closed
based on current market conditions and time-to-resolution.

Two evaluation methods:
- ``evaluate()`` — deterministic safety rules (no engine data needed).
- ``evaluate_with_engine()`` — adds engine-driven edge/profit/confidence rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import structlog

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

    def evaluate_with_engine(
        self,
        position: Position,
        current_snapshot: MarketSnapshot | None,
        entry_snapshot_mid: float,
        hours_to_resolution: float | None,
        engine_p_yes: float,
        engine_confidence: float,
    ) -> LifecycleDecision:
        """Evaluate with engine-driven rules on top of deterministic safety.

        Parameters
        ----------
        position:
            The open position to evaluate.
        current_snapshot:
            Latest market snapshot, or ``None`` if unavailable.
        entry_snapshot_mid:
            The mid price at time of entry.
        hours_to_resolution:
            Hours remaining until market resolves, or ``None``.
        engine_p_yes:
            Latest engine probability estimate.
        engine_confidence:
            Latest engine confidence in the estimate.
        """
        # Rules 1-4: run existing deterministic checks first.
        base_decision = self.evaluate(
            position, current_snapshot, entry_snapshot_mid, hours_to_resolution,
        )
        if base_decision.action != LifecycleAction.HOLD:
            return base_decision

        # Rule 5: min hold time gate — skip engine exits on fresh positions.
        min_hold_minutes = self.policy.min_hold_minutes
        if position.opened_ts_utc is not None:
            now = datetime.now(timezone.utc)
            age_minutes = (now - position.opened_ts_utc).total_seconds() / 60
            if age_minutes < min_hold_minutes:
                return LifecycleDecision(action=LifecycleAction.HOLD, reasons=[])

        # We know current_snapshot is valid (passed base checks).
        p_market = current_snapshot.mid  # type: ignore[union-attr]
        spread = (
            (current_snapshot.best_ask or p_market)  # type: ignore[union-attr]
            - (current_snapshot.best_bid or p_market)  # type: ignore[union-attr]
        )
        slippage = self.policy.slippage_bps / 10_000
        cost_buffer = spread / 2 + slippage + self.policy.fee_rate

        # Compute direction-aware signed edge.
        if position.side == "BUY_YES":
            signed_edge = engine_p_yes - p_market
        else:  # BUY_NO
            signed_edge = p_market - engine_p_yes

        # Rule 6: edge flip — engine's view has flipped against our position.
        flip_threshold = self.policy.exit_flip_threshold
        if signed_edge < -(flip_threshold + cost_buffer):
            reason = (
                f"edge_flip: signed_edge {signed_edge:.4f} "
                f"< -{flip_threshold + cost_buffer:.4f}"
            )
            self.logger.info(
                "lifecycle_close",
                market_id=position.market_id,
                reasons=[reason],
            )
            return LifecycleDecision(
                action=LifecycleAction.CLOSE, reasons=[reason],
            )

        # Rule 7: take profit — remaining edge won't cover exit costs.
        take_profit_band = self.policy.take_profit_band
        if abs(signed_edge) < (take_profit_band + cost_buffer):
            reason = (
                f"take_profit: edge_remaining {abs(signed_edge):.4f} "
                f"< {take_profit_band + cost_buffer:.4f}"
            )
            self.logger.info(
                "lifecycle_close",
                market_id=position.market_id,
                reasons=[reason],
            )
            return LifecycleDecision(
                action=LifecycleAction.CLOSE, reasons=[reason],
            )

        # Rule 8: confidence collapse.
        if engine_confidence < self.policy.min_confidence_hard:
            reason = (
                f"confidence_collapse: {engine_confidence:.4f} "
                f"< {self.policy.min_confidence_hard:.4f}"
            )
            self.logger.info(
                "lifecycle_close",
                market_id=position.market_id,
                reasons=[reason],
            )
            return LifecycleDecision(
                action=LifecycleAction.CLOSE, reasons=[reason],
            )

        return LifecycleDecision(action=LifecycleAction.HOLD, reasons=[])

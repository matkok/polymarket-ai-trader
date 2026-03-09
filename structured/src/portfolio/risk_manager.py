"""Portfolio-level risk checks.

Enforces all policy constraints before a new trade is allowed.  Each
constraint violation is collected so that callers can log or display
the exact reasons a trade was blocked.

Includes a category-scoped wrapper for isolated portfolio risk checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from src.config.policy import Policy
from src.db.models import Position


@dataclass
class RiskCheck:
    """Result of a pre-trade risk check."""

    allowed: bool
    violations: list[str] = field(default_factory=list)


class RiskManager:
    """Enforce policy constraints on new trades and report portfolio state."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.logger = structlog.get_logger(__name__)

    def check_new_trade(
        self,
        size_eur: float,
        market_id: str,
        current_positions: list[Position],
        daily_realized_pnl: float,
        total_unrealized_pnl: float = 0.0,
    ) -> RiskCheck:
        """Check if a new trade is allowed under policy constraints.

        All constraints are evaluated independently so the returned
        :class:`RiskCheck` contains every violation, not just the first.
        """
        violations: list[str] = []
        p = self.policy

        # 1. Max open positions.
        if len(current_positions) >= p.max_open_positions:
            violations.append(
                f"max_open_positions: {len(current_positions)} "
                f">= limit {p.max_open_positions}"
            )

        # Current totals.
        total_exposure = sum(pos.size_eur for pos in current_positions)
        market_exposure = sum(
            pos.size_eur
            for pos in current_positions
            if pos.market_id == market_id
        )

        # 2. Total exposure limit.
        max_total = p.bankroll_eur * p.max_total_exposure_frac
        if total_exposure + size_eur > max_total:
            violations.append(
                f"max_total_exposure: {total_exposure + size_eur:.2f} "
                f"> limit {max_total:.2f}"
            )

        # 3. Per-market exposure limit.
        max_market = p.bankroll_eur * p.max_exposure_per_market_frac
        if market_exposure + size_eur > max_market:
            violations.append(
                f"max_exposure_per_market: {market_exposure + size_eur:.2f} "
                f"> limit {max_market:.2f}"
            )

        # 4. Daily loss limit.
        max_daily_loss = p.bankroll_eur * p.max_daily_loss_frac
        if daily_realized_pnl < -max_daily_loss:
            violations.append(
                f"max_daily_loss: realised PnL {daily_realized_pnl:.2f} "
                f"< limit -{max_daily_loss:.2f}"
            )

        # 5. Cash reserve.
        cash_limit = p.bankroll_eur * (1.0 - p.cash_reserve_target_frac)
        if total_exposure + size_eur > cash_limit:
            violations.append(
                f"cash_reserve: exposure {total_exposure + size_eur:.2f} "
                f"> limit {cash_limit:.2f}"
            )

        # 6. Daily drawdown (realised + unrealised).
        max_drawdown = p.bankroll_eur * p.max_daily_drawdown_frac
        total_daily_pnl = daily_realized_pnl + total_unrealized_pnl
        if total_daily_pnl < -max_drawdown:
            violations.append(
                f"max_daily_drawdown: total PnL {total_daily_pnl:.2f} "
                f"< limit -{max_drawdown:.2f}"
            )

        allowed = len(violations) == 0
        if not allowed:
            self.logger.info(
                "risk_check_blocked",
                market_id=market_id,
                size_eur=size_eur,
                violations=violations,
            )
        return RiskCheck(allowed=allowed, violations=violations)

    def check_new_trade_category(
        self,
        size_eur: float,
        market_id: str,
        category: str,
        all_positions: list[Position],
        category_market_ids: set[str],
        daily_realized_pnl: float,
        total_unrealized_pnl: float = 0.0,
    ) -> RiskCheck:
        """Category-scoped risk check.

        Filters *all_positions* to only those belonging to the given
        category (via *category_market_ids*) before delegating to
        :meth:`check_new_trade`.
        """
        category_positions = [
            p for p in all_positions if p.market_id in category_market_ids
        ]
        return self.check_new_trade(
            size_eur=size_eur,
            market_id=market_id,
            current_positions=category_positions,
            daily_realized_pnl=daily_realized_pnl,
            total_unrealized_pnl=total_unrealized_pnl,
        )

    def get_portfolio_state(self, positions: list[Position]) -> dict:
        """Return current portfolio state summary."""
        total_exposure = sum(p.size_eur for p in positions)
        bankroll = self.policy.bankroll_eur
        return {
            "total_exposure_eur": total_exposure,
            "available_cash_eur": bankroll - total_exposure,
            "num_positions": len(positions),
            "exposure_frac": total_exposure / bankroll if bankroll > 0 else 0.0,
        }

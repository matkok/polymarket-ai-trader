"""Daily reporting module.

Uses the existing :class:`~src.db.repository.Repository` to generate
portfolio snapshot reports with PnL and exposure breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

from src.config.policy import Policy
from src.db.repository import Repository
from src.portfolio.risk_manager import RiskManager

logger = structlog.get_logger(__name__)


@dataclass
class PositionSummary:
    """Summary of a single position."""

    market_id: str
    side: str
    size_eur: float
    avg_entry_price: float
    last_price: float | None
    unrealized_pnl: float
    realized_pnl: float
    status: str


@dataclass
class DailyReport:
    """Daily PnL and exposure report."""

    report_ts_utc: datetime
    bankroll_eur: float
    total_exposure_eur: float
    available_cash_eur: float
    exposure_frac: float
    num_open_positions: int
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    positions: list[PositionSummary] = field(default_factory=list)


class ReportGenerator:
    """Generate daily trading reports."""

    def __init__(self, repo: Repository, policy: Policy) -> None:
        self.repo = repo
        self.policy = policy
        self.risk_manager = RiskManager(policy)

    async def generate_daily_report(self) -> DailyReport:
        """Generate a snapshot report of current portfolio state."""
        positions = await self.repo.get_open_positions()
        state = self.risk_manager.get_portfolio_state(positions)

        position_summaries: list[PositionSummary] = []
        total_unrealized = 0.0
        total_realized = 0.0

        for pos in positions:
            summary = PositionSummary(
                market_id=pos.market_id,
                side=pos.side,
                size_eur=pos.size_eur,
                avg_entry_price=pos.avg_entry_price,
                last_price=pos.last_price,
                unrealized_pnl=pos.unrealized_pnl,
                realized_pnl=pos.realized_pnl,
                status=pos.status,
            )
            position_summaries.append(summary)
            total_unrealized += pos.unrealized_pnl
            total_realized += pos.realized_pnl

        report = DailyReport(
            report_ts_utc=datetime.now(timezone.utc),
            bankroll_eur=self.policy.bankroll_eur,
            total_exposure_eur=state["total_exposure_eur"],
            available_cash_eur=state["available_cash_eur"],
            exposure_frac=state["exposure_frac"],
            num_open_positions=state["num_positions"],
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            total_pnl=total_unrealized + total_realized,
            positions=position_summaries,
        )

        logger.info(
            "daily_report",
            exposure_eur=report.total_exposure_eur,
            unrealized_pnl=report.total_unrealized_pnl,
            realized_pnl=report.total_realized_pnl,
            total_pnl=report.total_pnl,
            open_positions=report.num_open_positions,
        )

        return report

    def format_report(self, report: DailyReport) -> str:
        """Format a :class:`DailyReport` as a human-readable string."""
        lines = [
            f"=== Daily Report ({report.report_ts_utc.strftime('%Y-%m-%d %H:%M UTC')}) ===",
            f"Bankroll:          {report.bankroll_eur:>10.2f} EUR",
            f"Total Exposure:    {report.total_exposure_eur:>10.2f} EUR ({report.exposure_frac:.1%})",
            f"Available Cash:    {report.available_cash_eur:>10.2f} EUR",
            f"Open Positions:    {report.num_open_positions:>10d}",
            f"Unrealized PnL:    {report.total_unrealized_pnl:>+10.2f} EUR",
            f"Realized PnL:      {report.total_realized_pnl:>+10.2f} EUR",
            f"Total PnL:         {report.total_pnl:>+10.2f} EUR",
            "",
        ]

        if report.positions:
            lines.append("--- Positions ---")
            for p in report.positions:
                last = f"{p.last_price:.4f}" if p.last_price is not None else "N/A"
                lines.append(
                    f"  {p.market_id[:20]:<20s} {p.side:<8s} "
                    f"size={p.size_eur:>8.2f} entry={p.avg_entry_price:.4f} "
                    f"last={last} pnl={p.unrealized_pnl:>+8.2f}"
                )
        else:
            lines.append("No open positions.")

        return "\n".join(lines)

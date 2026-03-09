"""Tests for src.evaluation.reports — daily reporting module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.policy import Policy
from src.evaluation.reports import DailyReport, PositionSummary, ReportGenerator


# ---- Helpers ----------------------------------------------------------------


def _make_position(
    market_id: str = "mkt-1",
    side: str = "BUY_YES",
    size_eur: float = 100.0,
    avg_entry_price: float = 0.50,
    last_price: float | None = 0.55,
    unrealized_pnl: float = 10.0,
    realized_pnl: float = 0.0,
    status: str = "open",
) -> MagicMock:
    """Create a mock Position with the given attributes."""
    pos = MagicMock()
    pos.market_id = market_id
    pos.side = side
    pos.size_eur = size_eur
    pos.avg_entry_price = avg_entry_price
    pos.last_price = last_price
    pos.unrealized_pnl = unrealized_pnl
    pos.realized_pnl = realized_pnl
    pos.status = status
    return pos


def _make_repo(positions: list[MagicMock] | None = None) -> MagicMock:
    """Create a mock Repository whose get_open_positions returns *positions*."""
    repo = MagicMock()
    repo.get_open_positions = AsyncMock(return_value=positions or [])
    return repo


# ---- PositionSummary -------------------------------------------------------


class TestPositionSummary:
    """PositionSummary dataclass construction."""

    def test_fields(self) -> None:
        ps = PositionSummary(
            market_id="mkt-1",
            side="BUY_YES",
            size_eur=100.0,
            avg_entry_price=0.50,
            last_price=0.55,
            unrealized_pnl=10.0,
            realized_pnl=0.0,
            status="open",
        )
        assert ps.market_id == "mkt-1"
        assert ps.side == "BUY_YES"
        assert ps.size_eur == 100.0
        assert ps.avg_entry_price == 0.50
        assert ps.last_price == 0.55
        assert ps.unrealized_pnl == 10.0
        assert ps.realized_pnl == 0.0
        assert ps.status == "open"

    def test_last_price_none(self) -> None:
        """last_price accepts None."""
        ps = PositionSummary(
            market_id="mkt-1",
            side="BUY_NO",
            size_eur=50.0,
            avg_entry_price=0.40,
            last_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            status="open",
        )
        assert ps.last_price is None


# ---- DailyReport -----------------------------------------------------------


class TestDailyReport:
    """DailyReport dataclass construction."""

    def test_fields(self) -> None:
        ts = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
        report = DailyReport(
            report_ts_utc=ts,
            bankroll_eur=10_000.0,
            total_exposure_eur=500.0,
            available_cash_eur=9_500.0,
            exposure_frac=0.05,
            num_open_positions=2,
            total_unrealized_pnl=15.0,
            total_realized_pnl=-5.0,
            total_pnl=10.0,
        )
        assert report.report_ts_utc == ts
        assert report.bankroll_eur == 10_000.0
        assert report.num_open_positions == 2
        assert report.total_pnl == 10.0
        assert report.positions == []

    def test_positions_default_empty(self) -> None:
        """Positions list defaults to empty."""
        ts = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
        report = DailyReport(
            report_ts_utc=ts,
            bankroll_eur=10_000.0,
            total_exposure_eur=0.0,
            available_cash_eur=10_000.0,
            exposure_frac=0.0,
            num_open_positions=0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
        )
        assert report.positions == []


# ---- ReportGenerator -------------------------------------------------------


class TestReportGenerator:
    """ReportGenerator report generation and formatting tests."""

    async def test_generate_empty_portfolio(self) -> None:
        """Report for an empty portfolio has zero values."""
        repo = _make_repo(positions=[])
        policy = Policy(bankroll_eur=10_000.0)
        gen = ReportGenerator(repo, policy)

        report = await gen.generate_daily_report()

        assert report.bankroll_eur == 10_000.0
        assert report.total_exposure_eur == 0.0
        assert report.available_cash_eur == 10_000.0
        assert report.exposure_frac == 0.0
        assert report.num_open_positions == 0
        assert report.total_unrealized_pnl == 0.0
        assert report.total_realized_pnl == 0.0
        assert report.total_pnl == 0.0
        assert report.positions == []

    async def test_generate_with_positions(self) -> None:
        """Report aggregates PnL from multiple positions."""
        positions = [
            _make_position(
                market_id="mkt-1", size_eur=200.0,
                unrealized_pnl=15.0, realized_pnl=5.0,
            ),
            _make_position(
                market_id="mkt-2", size_eur=300.0,
                unrealized_pnl=-10.0, realized_pnl=3.0,
            ),
        ]
        repo = _make_repo(positions=positions)
        policy = Policy(bankroll_eur=10_000.0)
        gen = ReportGenerator(repo, policy)

        report = await gen.generate_daily_report()

        assert report.total_exposure_eur == pytest.approx(500.0)
        assert report.available_cash_eur == pytest.approx(9_500.0)
        assert report.num_open_positions == 2
        assert report.total_unrealized_pnl == pytest.approx(5.0)
        assert report.total_realized_pnl == pytest.approx(8.0)
        assert report.total_pnl == pytest.approx(13.0)
        assert len(report.positions) == 2

    async def test_generate_position_summaries_match(self) -> None:
        """Each PositionSummary reflects the source position."""
        pos = _make_position(
            market_id="mkt-abc",
            side="BUY_NO",
            size_eur=150.0,
            avg_entry_price=0.40,
            last_price=0.35,
            unrealized_pnl=12.5,
            realized_pnl=2.0,
            status="open",
        )
        repo = _make_repo(positions=[pos])
        policy = Policy()
        gen = ReportGenerator(repo, policy)

        report = await gen.generate_daily_report()

        assert len(report.positions) == 1
        ps = report.positions[0]
        assert ps.market_id == "mkt-abc"
        assert ps.side == "BUY_NO"
        assert ps.size_eur == 150.0
        assert ps.avg_entry_price == 0.40
        assert ps.last_price == 0.35
        assert ps.unrealized_pnl == 12.5
        assert ps.realized_pnl == 2.0
        assert ps.status == "open"

    async def test_generate_report_timestamp_is_utc(self) -> None:
        """Report timestamp is timezone-aware UTC."""
        repo = _make_repo()
        gen = ReportGenerator(repo, Policy())

        report = await gen.generate_daily_report()

        assert report.report_ts_utc.tzinfo is not None
        assert report.report_ts_utc.tzinfo == timezone.utc

    async def test_generate_exposure_frac(self) -> None:
        """Exposure fraction is correctly computed."""
        positions = [_make_position(size_eur=500.0)]
        repo = _make_repo(positions=positions)
        policy = Policy(bankroll_eur=10_000.0)
        gen = ReportGenerator(repo, policy)

        report = await gen.generate_daily_report()

        assert report.exposure_frac == pytest.approx(0.05)

    async def test_generate_zero_bankroll(self) -> None:
        """Zero bankroll does not cause division by zero."""
        repo = _make_repo(positions=[])
        policy = Policy(bankroll_eur=0.0)
        gen = ReportGenerator(repo, policy)

        report = await gen.generate_daily_report()

        assert report.exposure_frac == 0.0

    # ---- format_report ------------------------------------------------------

    def test_format_report_no_positions(self) -> None:
        """Formatted report shows 'No open positions.' for empty portfolio."""
        ts = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
        report = DailyReport(
            report_ts_utc=ts,
            bankroll_eur=10_000.0,
            total_exposure_eur=0.0,
            available_cash_eur=10_000.0,
            exposure_frac=0.0,
            num_open_positions=0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
        )
        gen = ReportGenerator(_make_repo(), Policy())
        text = gen.format_report(report)

        assert "2025-01-15 12:00 UTC" in text
        assert "10000.00" in text
        assert "No open positions." in text

    def test_format_report_with_positions(self) -> None:
        """Formatted report lists each position."""
        ts = datetime(2025, 6, 1, 8, 30, tzinfo=timezone.utc)
        ps = PositionSummary(
            market_id="will-btc-hit-100k",
            side="BUY_YES",
            size_eur=200.0,
            avg_entry_price=0.6500,
            last_price=0.7000,
            unrealized_pnl=15.38,
            realized_pnl=0.0,
            status="open",
        )
        report = DailyReport(
            report_ts_utc=ts,
            bankroll_eur=10_000.0,
            total_exposure_eur=200.0,
            available_cash_eur=9_800.0,
            exposure_frac=0.02,
            num_open_positions=1,
            total_unrealized_pnl=15.38,
            total_realized_pnl=0.0,
            total_pnl=15.38,
            positions=[ps],
        )
        gen = ReportGenerator(_make_repo(), Policy())
        text = gen.format_report(report)

        assert "2025-06-01 08:30 UTC" in text
        assert "--- Positions ---" in text
        assert "will-btc-hit-100k" in text
        assert "BUY_YES" in text
        assert "0.6500" in text
        assert "0.7000" in text

    def test_format_report_last_price_none(self) -> None:
        """Position with None last_price shows 'N/A'."""
        ts = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        ps = PositionSummary(
            market_id="mkt-1",
            side="BUY_NO",
            size_eur=100.0,
            avg_entry_price=0.50,
            last_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            status="open",
        )
        report = DailyReport(
            report_ts_utc=ts,
            bankroll_eur=10_000.0,
            total_exposure_eur=100.0,
            available_cash_eur=9_900.0,
            exposure_frac=0.01,
            num_open_positions=1,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            positions=[ps],
        )
        gen = ReportGenerator(_make_repo(), Policy())
        text = gen.format_report(report)

        assert "last=N/A" in text

    def test_format_report_long_market_id_truncated(self) -> None:
        """Market IDs longer than 20 chars are truncated in formatted output."""
        ts = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        long_id = "a-very-long-market-id-that-exceeds-twenty-characters"
        ps = PositionSummary(
            market_id=long_id,
            side="BUY_YES",
            size_eur=50.0,
            avg_entry_price=0.30,
            last_price=0.35,
            unrealized_pnl=8.33,
            realized_pnl=0.0,
            status="open",
        )
        report = DailyReport(
            report_ts_utc=ts,
            bankroll_eur=10_000.0,
            total_exposure_eur=50.0,
            available_cash_eur=9_950.0,
            exposure_frac=0.005,
            num_open_positions=1,
            total_unrealized_pnl=8.33,
            total_realized_pnl=0.0,
            total_pnl=8.33,
            positions=[ps],
        )
        gen = ReportGenerator(_make_repo(), Policy())
        text = gen.format_report(report)

        # The first 20 characters should appear, but the full ID should not.
        assert long_id[:20] in text
        assert long_id not in text

    def test_format_report_header_structure(self) -> None:
        """Formatted report contains expected header lines."""
        ts = datetime(2025, 3, 10, 14, 45, tzinfo=timezone.utc)
        report = DailyReport(
            report_ts_utc=ts,
            bankroll_eur=5_000.0,
            total_exposure_eur=1_000.0,
            available_cash_eur=4_000.0,
            exposure_frac=0.20,
            num_open_positions=3,
            total_unrealized_pnl=25.0,
            total_realized_pnl=-10.0,
            total_pnl=15.0,
        )
        gen = ReportGenerator(_make_repo(), Policy())
        text = gen.format_report(report)

        assert "=== Daily Report" in text
        assert "Bankroll:" in text
        assert "Total Exposure:" in text
        assert "Available Cash:" in text
        assert "Open Positions:" in text
        assert "Unrealized PnL:" in text
        assert "Realized PnL:" in text
        assert "Total PnL:" in text

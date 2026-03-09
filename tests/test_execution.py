"""Tests for src.execution — paper_executor and fills modules."""

from __future__ import annotations

import pytest

from src.config.policy import Policy
from src.execution.fills import PnLSnapshot, calculate_realized_pnl, calculate_unrealized_pnl
from src.execution.paper_executor import PaperExecutor, PaperFill, walk_order_book


# ---- Paper Executor ---------------------------------------------------------


class TestPaperFill:
    """PaperFill dataclass construction."""

    def test_fields(self) -> None:
        fill = PaperFill(
            side="BUY_YES", price=0.55, size_eur=100.0,
            fee_eur=2.0, slippage_applied=0.005,
        )
        assert fill.side == "BUY_YES"
        assert fill.price == 0.55
        assert fill.size_eur == 100.0
        assert fill.fee_eur == 2.0
        assert fill.slippage_applied == 0.005


class TestPaperExecutor:
    """Paper executor simulation tests."""

    def test_buy_yes_fill_at_ask_plus_slippage(self) -> None:
        """BUY_YES fills at best_ask + slippage."""
        policy = Policy(slippage_bps=50, fee_rate=0.02)
        ex = PaperExecutor(policy)
        fill = ex.execute(side="BUY_YES", size_eur=100.0, best_bid=0.48, best_ask=0.52)

        expected_slippage = 50 / 10_000  # 0.005
        expected_price = 0.52 + expected_slippage  # 0.525
        assert fill.price == pytest.approx(expected_price)
        assert fill.fee_eur == pytest.approx(2.0)
        assert fill.slippage_applied == pytest.approx(expected_slippage)
        assert fill.side == "BUY_YES"
        assert fill.size_eur == 100.0

    def test_buy_no_fill_at_ask_plus_slippage(self) -> None:
        """BUY_NO also fills at best_ask + slippage."""
        policy = Policy(slippage_bps=100, fee_rate=0.01)
        ex = PaperExecutor(policy)
        fill = ex.execute(side="BUY_NO", size_eur=200.0, best_bid=0.45, best_ask=0.55)

        expected_slippage = 100 / 10_000  # 0.01
        expected_price = 0.55 + expected_slippage  # 0.56
        assert fill.price == pytest.approx(expected_price)
        assert fill.fee_eur == pytest.approx(2.0)

    def test_sell_fill_at_bid_minus_slippage(self) -> None:
        """Sell orders fill at best_bid - slippage."""
        policy = Policy(slippage_bps=50, fee_rate=0.02)
        ex = PaperExecutor(policy)
        fill = ex.execute(side="SELL", size_eur=100.0, best_bid=0.48, best_ask=0.52)

        expected_slippage = 50 / 10_000  # 0.005
        expected_price = 0.48 - expected_slippage  # 0.475
        assert fill.price == pytest.approx(expected_price)

    def test_price_clamped_high(self) -> None:
        """Fill price is clamped to 0.99 maximum."""
        policy = Policy(slippage_bps=200)
        ex = PaperExecutor(policy)
        fill = ex.execute(side="BUY_YES", size_eur=100.0, best_bid=0.96, best_ask=0.98)

        assert fill.price == 0.99  # 0.98 + 0.02 = 1.00, clamped to 0.99

    def test_price_clamped_low(self) -> None:
        """Fill price is clamped to 0.01 minimum."""
        policy = Policy(slippage_bps=200)
        ex = PaperExecutor(policy)
        fill = ex.execute(side="SELL", size_eur=100.0, best_bid=0.02, best_ask=0.04)

        assert fill.price == 0.01  # 0.02 - 0.02 = 0.00, clamped to 0.01

    def test_zero_slippage(self) -> None:
        """Zero slippage fills at exact bid/ask."""
        policy = Policy(slippage_bps=0, fee_rate=0.0)
        ex = PaperExecutor(policy)
        fill = ex.execute(side="BUY_YES", size_eur=50.0, best_bid=0.40, best_ask=0.60)

        assert fill.price == pytest.approx(0.60)
        assert fill.fee_eur == pytest.approx(0.0)
        assert fill.slippage_applied == pytest.approx(0.0)


# ---- Fills / PnL ------------------------------------------------------------


class TestPnLSnapshot:
    """PnLSnapshot dataclass construction."""

    def test_fields(self) -> None:
        snap = PnLSnapshot(unrealized_pnl=10.0, realized_pnl=-5.0, total_pnl=5.0)
        assert snap.unrealized_pnl == 10.0
        assert snap.realized_pnl == -5.0
        assert snap.total_pnl == 5.0


class TestCalculateUnrealizedPnl:
    """Unrealized PnL calculation tests."""

    def test_buy_yes_profit(self) -> None:
        """BUY_YES with rising price shows profit."""
        pnl = calculate_unrealized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50, current_price=0.60,
        )
        # 100 * (0.60 - 0.50) / 0.50 = 20.0
        assert pnl == pytest.approx(20.0)

    def test_buy_yes_loss(self) -> None:
        """BUY_YES with falling price shows loss."""
        pnl = calculate_unrealized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50, current_price=0.40,
        )
        # 100 * (0.40 - 0.50) / 0.50 = -20.0
        assert pnl == pytest.approx(-20.0)

    def test_buy_no_profit(self) -> None:
        """BUY_NO with falling price shows profit."""
        pnl = calculate_unrealized_pnl(
            side="BUY_NO", size_eur=100.0,
            avg_entry_price=0.50, current_price=0.40,
        )
        # 100 * (0.50 - 0.40) / 0.50 = 20.0
        assert pnl == pytest.approx(20.0)

    def test_buy_no_loss(self) -> None:
        """BUY_NO with rising price shows loss."""
        pnl = calculate_unrealized_pnl(
            side="BUY_NO", size_eur=100.0,
            avg_entry_price=0.50, current_price=0.60,
        )
        # 100 * (0.50 - 0.60) / 0.50 = -20.0
        assert pnl == pytest.approx(-20.0)

    def test_no_change(self) -> None:
        """PnL is zero when price has not moved."""
        pnl = calculate_unrealized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50, current_price=0.50,
        )
        assert pnl == pytest.approx(0.0)

    def test_zero_entry_price(self) -> None:
        """Zero entry price returns zero to avoid division by zero."""
        pnl = calculate_unrealized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.0, current_price=0.50,
        )
        assert pnl == 0.0

    def test_negative_entry_price(self) -> None:
        """Negative entry price returns zero."""
        pnl = calculate_unrealized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=-0.10, current_price=0.50,
        )
        assert pnl == 0.0


class TestCalculateRealizedPnl:
    """Realized PnL calculation tests."""

    def test_buy_yes_profit_minus_fees(self) -> None:
        """Realized PnL subtracts fees from gross profit."""
        pnl = calculate_realized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50, exit_price=0.60, fee_eur=3.0,
        )
        # gross = 100 * (0.60 - 0.50) / 0.50 = 20.0; net = 20.0 - 3.0 = 17.0
        assert pnl == pytest.approx(17.0)

    def test_buy_no_profit_minus_fees(self) -> None:
        """BUY_NO realized PnL with fees."""
        pnl = calculate_realized_pnl(
            side="BUY_NO", size_eur=100.0,
            avg_entry_price=0.50, exit_price=0.40, fee_eur=2.0,
        )
        # gross = 100 * (0.50 - 0.40) / 0.50 = 20.0; net = 20.0 - 2.0 = 18.0
        assert pnl == pytest.approx(18.0)

    def test_loss_with_fees(self) -> None:
        """Fees deepen a loss."""
        pnl = calculate_realized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50, exit_price=0.40, fee_eur=2.0,
        )
        # gross = 100 * (0.40 - 0.50) / 0.50 = -20.0; net = -20.0 - 2.0 = -22.0
        assert pnl == pytest.approx(-22.0)

    def test_zero_entry_price(self) -> None:
        """Zero entry price returns negative fees."""
        pnl = calculate_realized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.0, exit_price=0.50, fee_eur=5.0,
        )
        assert pnl == pytest.approx(-5.0)

    def test_zero_fees(self) -> None:
        """Realized PnL with zero fees equals gross PnL."""
        pnl = calculate_realized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50, exit_price=0.60, fee_eur=0.0,
        )
        unrealized = calculate_unrealized_pnl(
            side="BUY_YES", size_eur=100.0,
            avg_entry_price=0.50, current_price=0.60,
        )
        assert pnl == pytest.approx(unrealized)


# ---- Fee Rate Override -------------------------------------------------------


class TestFeeRateOverride:
    """Fee rate override in PaperExecutor."""

    def test_fee_rate_override_used(self) -> None:
        """When fee_rate_override is provided, it replaces the policy default."""
        policy = Policy(slippage_bps=50, fee_rate=0.0)
        ex = PaperExecutor(policy)
        fill = ex.execute(
            side="BUY_YES", size_eur=100.0,
            best_bid=0.48, best_ask=0.52,
            fee_rate_override=0.02,
        )
        assert fill.fee_eur == pytest.approx(2.0)

    def test_default_zero_fee(self) -> None:
        """Default fee_rate=0.0 results in zero fees."""
        policy = Policy(fee_rate=0.0, slippage_bps=50)
        ex = PaperExecutor(policy)
        fill = ex.execute(
            side="BUY_YES", size_eur=100.0,
            best_bid=0.48, best_ask=0.52,
        )
        assert fill.fee_eur == pytest.approx(0.0)

    def test_fee_rate_override_none_uses_policy(self) -> None:
        """When fee_rate_override is None, policy.fee_rate is used."""
        policy = Policy(slippage_bps=50, fee_rate=0.01)
        ex = PaperExecutor(policy)
        fill = ex.execute(
            side="BUY_YES", size_eur=100.0,
            best_bid=0.48, best_ask=0.52,
            fee_rate_override=None,
        )
        assert fill.fee_eur == pytest.approx(1.0)


# ---- Walk Order Book ---------------------------------------------------------


class TestWalkOrderBook:
    """Order book VWAP walk tests."""

    def test_single_level_sufficient(self) -> None:
        """Single level with enough depth fills at that price."""
        entries = [(0.55, 200.0)]
        vwap = walk_order_book(entries, 100.0)
        assert vwap == pytest.approx(0.55)

    def test_multi_level_vwap(self) -> None:
        """Multiple levels compute correct VWAP."""
        # Level 1: 50 EUR at 0.50, Level 2: 50 EUR at 0.55
        entries = [(0.50, 50.0), (0.55, 50.0)]
        vwap = walk_order_book(entries, 100.0)
        # VWAP = (50*0.50 + 50*0.55) / 100 = 52.5 / 100 = 0.525
        assert vwap == pytest.approx(0.525)

    def test_insufficient_depth_returns_none(self) -> None:
        """Returns None when order book doesn't have enough depth."""
        entries = [(0.50, 30.0)]
        vwap = walk_order_book(entries, 100.0)
        assert vwap is None

    def test_empty_book_returns_none(self) -> None:
        """Returns None for an empty order book."""
        vwap = walk_order_book([], 100.0)
        assert vwap is None

    def test_zero_target_returns_none(self) -> None:
        """Returns None for zero target size."""
        entries = [(0.55, 200.0)]
        vwap = walk_order_book(entries, 0.0)
        assert vwap is None


# ---- Executor with Order Book ------------------------------------------------


class TestExecutorWithOrderBook:
    """Integration of order book with executor."""

    def test_executor_uses_order_book_vwap(self) -> None:
        """When order book is provided and has depth, VWAP is used."""
        policy = Policy(slippage_bps=50, fee_rate=0.0)
        ex = PaperExecutor(policy)
        ob = [(0.53, 200.0)]  # Sufficient depth at 0.53
        fill = ex.execute(
            side="BUY_YES", size_eur=100.0,
            best_bid=0.48, best_ask=0.52,
            order_book=ob,
        )
        # VWAP is 0.53, clamped to [0.01, 0.99] → 0.53
        assert fill.price == pytest.approx(0.53)

    def test_executor_falls_back_on_insufficient_ob(self) -> None:
        """Insufficient order book depth falls back to constant slippage."""
        policy = Policy(slippage_bps=50, fee_rate=0.0)
        ex = PaperExecutor(policy)
        ob = [(0.53, 10.0)]  # Not enough depth
        fill = ex.execute(
            side="BUY_YES", size_eur=100.0,
            best_bid=0.48, best_ask=0.52,
            order_book=ob,
        )
        # Falls back to ask + slippage = 0.52 + 0.005 = 0.525
        assert fill.price == pytest.approx(0.525)

"""Tests for src.portfolio — sizing, risk_manager, and lifecycle modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.aggregation.aggregator import AggregationResult
from src.config.policy import Policy
from src.portfolio.lifecycle import LifecycleAction, LifecycleDecision, PositionLifecycle
from src.portfolio.risk_manager import RiskCheck, RiskManager
from src.portfolio.sizing import SizingInput, SizingResult, compute_size


# ---- Helpers ----------------------------------------------------------------


def _make_position(
    market_id: str = "mkt-1",
    side: str = "BUY_YES",
    size_eur: float = 100.0,
    avg_entry_price: float = 0.50,
    opened_ts_utc: datetime | None = None,
) -> MagicMock:
    """Create a mock Position with the given attributes."""
    pos = MagicMock()
    pos.market_id = market_id
    pos.side = side
    pos.size_eur = size_eur
    pos.avg_entry_price = avg_entry_price
    # Default to 2 hours ago (well past min_hold_minutes=60).
    pos.opened_ts_utc = opened_ts_utc or (datetime.now(timezone.utc) - timedelta(hours=2))
    return pos


def _make_snapshot(mid: float | None = 0.50, best_bid: float = 0.48, best_ask: float = 0.52) -> MagicMock:
    """Create a mock MarketSnapshot with the given attributes."""
    snap = MagicMock()
    snap.mid = mid
    snap.best_bid = best_bid
    snap.best_ask = best_ask
    return snap


# ---- Sizing -----------------------------------------------------------------


class TestSizingInput:
    """SizingInput dataclass construction."""

    def test_fields(self) -> None:
        inp = SizingInput(p_consensus=0.6, p_market=0.5, confidence=1.0, disagreement=0.0)
        assert inp.p_consensus == 0.6
        assert inp.p_market == 0.5
        assert inp.confidence == 1.0
        assert inp.disagreement == 0.0


class TestComputeSize:
    """Position sizing formula tests."""

    def test_buy_yes_with_edge(self) -> None:
        """Positive edge_yes leads to a BUY_YES trade."""
        policy = Policy()
        inp = SizingInput(p_consensus=0.65, p_market=0.50, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)

        assert result.side == "BUY_YES"
        assert result.gross_edge == pytest.approx(0.15)
        assert result.edge == pytest.approx(0.145)  # 0.15 - 0.005 slippage
        assert result.skip_reason is None
        assert result.clamped_size_eur > 0
        assert result.raw_size_eur > 0

    def test_buy_no_with_edge(self) -> None:
        """Negative edge_yes leads to a BUY_NO trade."""
        policy = Policy()
        inp = SizingInput(p_consensus=0.35, p_market=0.50, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)

        assert result.side == "BUY_NO"
        assert result.gross_edge == pytest.approx(0.15)
        assert result.edge == pytest.approx(0.145)  # 0.15 - 0.005 slippage
        assert result.skip_reason is None
        assert result.clamped_size_eur > 0

    def test_skip_below_edge_threshold(self) -> None:
        """Trade is skipped when edge is below threshold."""
        policy = Policy(edge_threshold=0.10)
        inp = SizingInput(p_consensus=0.55, p_market=0.50, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)

        assert result.skip_reason is not None
        assert "below threshold" in result.skip_reason
        assert result.clamped_size_eur == 0.0

    def test_skip_at_zero_edge(self) -> None:
        """No edge means no trade."""
        policy = Policy()
        inp = SizingInput(p_consensus=0.50, p_market=0.50, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)

        assert result.skip_reason is not None
        assert result.clamped_size_eur == 0.0

    def test_disagreement_blocks_trade(self) -> None:
        """Trade is blocked when disagreement exceeds block threshold."""
        policy = Policy()
        inp = SizingInput(
            p_consensus=0.70, p_market=0.50, confidence=1.0,
            disagreement=0.15,  # equals block threshold
        )
        result = compute_size(inp, policy)

        assert result.skip_reason is not None
        assert "block threshold" in result.skip_reason
        assert result.clamped_size_eur == 0.0

    def test_disagreement_penalty_reduces_size(self) -> None:
        """Disagreement above penalty start reduces position size."""
        policy = Policy()
        inp_no_disagree = SizingInput(
            p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.0,
        )
        inp_disagree = SizingInput(
            p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.10,
        )
        result_full = compute_size(inp_no_disagree, policy)
        result_penalised = compute_size(inp_disagree, policy)

        assert result_penalised.raw_size_eur < result_full.raw_size_eur

    def test_confidence_scales_size(self) -> None:
        """Lower confidence produces smaller size via effective_conf ramp."""
        policy = Policy()
        inp_full = SizingInput(p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.0)
        inp_mid = SizingInput(p_consensus=0.70, p_market=0.50, confidence=0.5, disagreement=0.0)

        result_full = compute_size(inp_full, policy)
        result_mid = compute_size(inp_mid, policy)

        # confidence=1.0 → effective_conf=1.0, confidence=0.5 → effective_conf=(0.5-0.25)/0.35
        expected_eff = (0.5 - policy.min_confidence_hard) / (
            policy.min_confidence_full - policy.min_confidence_hard
        )
        assert result_mid.raw_size_eur == pytest.approx(
            result_full.raw_size_eur * expected_eff, rel=0.01
        )
        assert result_mid.raw_size_eur < result_full.raw_size_eur

    def test_clamped_to_max_per_market(self) -> None:
        """Size is clamped to max_exposure_per_market_frac."""
        policy = Policy(
            bankroll_eur=10_000.0,
            max_exposure_per_market_frac=0.01,  # 100 EUR limit
            base_risk_frac=0.10,  # large base to trigger clamp
            edge_scale=0.05,
        )
        inp = SizingInput(p_consensus=0.80, p_market=0.50, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)

        assert result.clamped_size_eur == pytest.approx(100.0)
        assert result.raw_size_eur > 100.0

    def test_raw_size_formula(self) -> None:
        """Verify the exact raw size formula: base * (net_edge / edge_scale) * confidence."""
        policy = Policy(bankroll_eur=10_000.0, base_risk_frac=0.02, edge_scale=0.20)
        inp = SizingInput(p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)

        expected_base = 10_000.0 * 0.02  # 200
        net_edge = 0.20 - 0.005  # gross 0.20 minus slippage
        expected_raw = expected_base * (net_edge / 0.20) * 1.0  # 200 * 0.975 = 195
        assert result.raw_size_eur == pytest.approx(expected_raw)

    def test_skip_tiny_size(self) -> None:
        """Sizes below 1.0 EUR are skipped."""
        policy = Policy(
            bankroll_eur=100.0,
            base_risk_frac=0.001,
            edge_scale=0.50,
            edge_threshold=0.01,
        )
        inp = SizingInput(p_consensus=0.52, p_market=0.50, confidence=0.1, disagreement=0.0)
        result = compute_size(inp, policy)

        assert result.skip_reason is not None
        assert "below minimum" in result.skip_reason
        assert result.clamped_size_eur == 0.0


# ---- Risk Manager -----------------------------------------------------------


class TestRiskCheck:
    """RiskCheck dataclass construction."""

    def test_allowed(self) -> None:
        rc = RiskCheck(allowed=True, violations=[])
        assert rc.allowed is True
        assert rc.violations == []

    def test_blocked(self) -> None:
        rc = RiskCheck(allowed=False, violations=["reason"])
        assert rc.allowed is False
        assert len(rc.violations) == 1


class TestRiskManager:
    """RiskManager constraint enforcement."""

    def test_trade_allowed_empty_portfolio(self) -> None:
        """A trade is allowed when there are no existing positions."""
        rm = RiskManager(Policy())
        check = rm.check_new_trade(
            size_eur=100.0,
            market_id="mkt-1",
            current_positions=[],
            daily_realized_pnl=0.0,
        )
        assert check.allowed is True
        assert check.violations == []

    def test_max_open_positions_violated(self) -> None:
        """Trade is blocked when max_open_positions is reached."""
        policy = Policy(max_open_positions=2)
        rm = RiskManager(policy)
        positions = [_make_position(market_id=f"mkt-{i}") for i in range(2)]
        check = rm.check_new_trade(
            size_eur=100.0,
            market_id="mkt-new",
            current_positions=positions,
            daily_realized_pnl=0.0,
        )
        assert check.allowed is False
        assert any("max_open_positions" in v for v in check.violations)

    def test_total_exposure_violated(self) -> None:
        """Trade is blocked when total exposure exceeds limit."""
        policy = Policy(bankroll_eur=1_000.0, max_total_exposure_frac=0.50)
        rm = RiskManager(policy)
        positions = [_make_position(size_eur=400.0)]
        check = rm.check_new_trade(
            size_eur=200.0,
            market_id="mkt-new",
            current_positions=positions,
            daily_realized_pnl=0.0,
        )
        assert check.allowed is False
        assert any("max_total_exposure" in v for v in check.violations)

    def test_per_market_exposure_violated(self) -> None:
        """Trade is blocked when per-market exposure exceeds limit."""
        policy = Policy(bankroll_eur=10_000.0, max_exposure_per_market_frac=0.05)
        rm = RiskManager(policy)
        positions = [_make_position(market_id="mkt-1", size_eur=400.0)]
        check = rm.check_new_trade(
            size_eur=200.0,
            market_id="mkt-1",
            current_positions=positions,
            daily_realized_pnl=0.0,
        )
        assert check.allowed is False
        assert any("max_exposure_per_market" in v for v in check.violations)

    def test_daily_loss_violated(self) -> None:
        """Trade is blocked when daily loss limit has been exceeded."""
        policy = Policy(bankroll_eur=10_000.0, max_daily_loss_frac=0.05)
        rm = RiskManager(policy)
        check = rm.check_new_trade(
            size_eur=100.0,
            market_id="mkt-1",
            current_positions=[],
            daily_realized_pnl=-600.0,  # exceeds 500 limit
        )
        assert check.allowed is False
        assert any("max_daily_loss" in v for v in check.violations)

    def test_cash_reserve_violated(self) -> None:
        """Trade is blocked when cash reserve would be breached."""
        policy = Policy(bankroll_eur=1_000.0, cash_reserve_target_frac=0.20)
        rm = RiskManager(policy)
        positions = [_make_position(size_eur=750.0)]
        check = rm.check_new_trade(
            size_eur=100.0,
            market_id="mkt-new",
            current_positions=positions,
            daily_realized_pnl=0.0,
        )
        assert check.allowed is False
        assert any("cash_reserve" in v for v in check.violations)

    def test_daily_drawdown_violated(self) -> None:
        """Trade is blocked when realised + unrealised PnL exceeds drawdown limit."""
        policy = Policy(bankroll_eur=10_000.0, max_daily_drawdown_frac=0.08)
        rm = RiskManager(policy)
        check = rm.check_new_trade(
            size_eur=100.0,
            market_id="mkt-1",
            current_positions=[],
            daily_realized_pnl=-500.0,
            total_unrealized_pnl=-400.0,  # total = -900 > -800 limit
        )
        assert check.allowed is False
        assert any("max_daily_drawdown" in v for v in check.violations)

    def test_daily_drawdown_allowed(self) -> None:
        """Trade is allowed when drawdown is within limits."""
        policy = Policy(bankroll_eur=10_000.0, max_daily_drawdown_frac=0.08)
        rm = RiskManager(policy)
        check = rm.check_new_trade(
            size_eur=100.0,
            market_id="mkt-1",
            current_positions=[],
            daily_realized_pnl=-200.0,
            total_unrealized_pnl=-100.0,  # total = -300 < -800 limit
        )
        assert check.allowed is True

    def test_daily_drawdown_unrealized_only_trigger(self) -> None:
        """Unrealised losses alone can trigger the drawdown guard."""
        policy = Policy(bankroll_eur=10_000.0, max_daily_drawdown_frac=0.08)
        rm = RiskManager(policy)
        check = rm.check_new_trade(
            size_eur=100.0,
            market_id="mkt-1",
            current_positions=[],
            daily_realized_pnl=0.0,
            total_unrealized_pnl=-900.0,  # total = -900 > -800 limit
        )
        assert check.allowed is False
        assert any("max_daily_drawdown" in v for v in check.violations)

    def test_multiple_violations(self) -> None:
        """All violations are collected, not just the first."""
        policy = Policy(
            bankroll_eur=1_000.0,
            max_open_positions=1,
            max_total_exposure_frac=0.10,
        )
        rm = RiskManager(policy)
        positions = [_make_position(size_eur=500.0)]
        check = rm.check_new_trade(
            size_eur=200.0,
            market_id="mkt-new",
            current_positions=positions,
            daily_realized_pnl=0.0,
        )
        assert check.allowed is False
        assert len(check.violations) >= 2

    def test_get_portfolio_state_empty(self) -> None:
        """Portfolio state for empty portfolio."""
        rm = RiskManager(Policy(bankroll_eur=10_000.0))
        state = rm.get_portfolio_state([])
        assert state["total_exposure_eur"] == 0.0
        assert state["available_cash_eur"] == 10_000.0
        assert state["num_positions"] == 0
        assert state["exposure_frac"] == 0.0

    def test_get_portfolio_state_with_positions(self) -> None:
        """Portfolio state correctly sums position sizes."""
        rm = RiskManager(Policy(bankroll_eur=10_000.0))
        positions = [
            _make_position(size_eur=200.0),
            _make_position(size_eur=300.0),
        ]
        state = rm.get_portfolio_state(positions)
        assert state["total_exposure_eur"] == pytest.approx(500.0)
        assert state["available_cash_eur"] == pytest.approx(9_500.0)
        assert state["num_positions"] == 2
        assert state["exposure_frac"] == pytest.approx(0.05)

    def test_get_portfolio_state_zero_bankroll(self) -> None:
        """Zero bankroll does not cause division by zero."""
        rm = RiskManager(Policy(bankroll_eur=0.0))
        state = rm.get_portfolio_state([])
        assert state["exposure_frac"] == 0.0


# ---- Lifecycle --------------------------------------------------------------


class TestLifecycleAction:
    """LifecycleAction enum values."""

    def test_values(self) -> None:
        assert LifecycleAction.HOLD == "HOLD"
        assert LifecycleAction.REDUCE == "REDUCE"
        assert LifecycleAction.CLOSE == "CLOSE"
        assert LifecycleAction.ADD == "ADD"


class TestPositionLifecycle:
    """Position lifecycle evaluation tests."""

    def test_hold_normal_conditions(self) -> None:
        """Position is held under normal conditions."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES", size_eur=100.0)
        snap = _make_snapshot(mid=0.52)
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=100.0)
        assert decision.action == LifecycleAction.HOLD

    def test_close_no_snapshot(self) -> None:
        """Position is closed when no snapshot is available."""
        lc = PositionLifecycle(Policy())
        pos = _make_position()
        decision = lc.evaluate(pos, None, entry_snapshot_mid=0.50, hours_to_resolution=100.0)
        assert decision.action == LifecycleAction.CLOSE
        assert any("no current market snapshot" in r for r in decision.reasons)

    def test_close_no_mid(self) -> None:
        """Position is closed when snapshot has no mid price."""
        lc = PositionLifecycle(Policy())
        pos = _make_position()
        snap = _make_snapshot(mid=None)
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=100.0)
        assert decision.action == LifecycleAction.CLOSE
        assert any("no mid price" in r for r in decision.reasons)

    def test_close_approaching_resolution(self) -> None:
        """Position is closed when too close to resolution."""
        policy = Policy(min_hours_to_resolution=24)
        lc = PositionLifecycle(policy)
        pos = _make_position()
        snap = _make_snapshot(mid=0.50)
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=12.0)
        assert decision.action == LifecycleAction.CLOSE
        assert any("hours_to_resolution" in r for r in decision.reasons)

    def test_close_stop_loss_buy_yes(self) -> None:
        """BUY_YES position is closed when mid drops significantly."""
        policy = Policy(edge_threshold=0.05)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES", size_eur=100.0)
        snap = _make_snapshot(mid=0.35)  # dropped 0.15 from entry 0.50
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=100.0)
        assert decision.action == LifecycleAction.CLOSE
        assert any("stop loss" in r for r in decision.reasons)

    def test_close_stop_loss_buy_no(self) -> None:
        """BUY_NO position is closed when mid rises significantly."""
        policy = Policy(edge_threshold=0.05)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_NO", size_eur=100.0)
        snap = _make_snapshot(mid=0.65)  # rose 0.15 from entry 0.50
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=100.0)
        assert decision.action == LifecycleAction.CLOSE
        assert any("stop loss" in r for r in decision.reasons)

    def test_reduce_partial_stop_buy_yes(self) -> None:
        """BUY_YES position is reduced on moderate unrealized loss."""
        policy = Policy(edge_threshold=0.05)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES", size_eur=100.0)
        # Mid drops enough for >3% loss but not enough for stop loss.
        # entry=0.50, need pnl < -3.0 => (mid - 0.50)/0.50 * 100 < -3.0
        # mid < 0.485.  But stop loss requires 0.50 - mid > 0.10, so mid < 0.40.
        # Use mid=0.48 => pnl = 100*(0.48-0.50)/0.50 = -4.0 < -3.0.  Move against = 0.02 < 0.10.
        snap = _make_snapshot(mid=0.48)
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=100.0)
        assert decision.action == LifecycleAction.REDUCE
        assert any("partial stop" in r for r in decision.reasons)

    def test_reduce_partial_stop_buy_no(self) -> None:
        """BUY_NO position is reduced on moderate unrealized loss."""
        policy = Policy(edge_threshold=0.05)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_NO", size_eur=100.0)
        # entry=0.50, mid=0.52 => pnl = 100*(0.50-0.52)/0.50 = -4.0 < -3.0
        # move against = 0.52-0.50 = 0.02 < 0.10
        snap = _make_snapshot(mid=0.52)
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=100.0)
        assert decision.action == LifecycleAction.REDUCE
        assert any("partial stop" in r for r in decision.reasons)

    def test_hold_when_hours_to_resolution_none(self) -> None:
        """Position is held when hours_to_resolution is unknown."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        decision = lc.evaluate(pos, snap, entry_snapshot_mid=0.50, hours_to_resolution=None)
        assert decision.action == LifecycleAction.HOLD


# ---- Lifecycle with Aggregation (M4) ----------------------------------------


class TestPositionLifecycleWithAggregation:
    """evaluate_with_aggregation() tests."""

    def _make_agg(
        self,
        p_consensus: float = 0.60,
        confidence: float = 0.80,
        disagreement: float = 0.02,
        veto: bool = False,
        trade_allowed: bool = True,
    ) -> AggregationResult:
        return AggregationResult(
            p_consensus=p_consensus,
            confidence=confidence,
            disagreement=disagreement,
            veto=veto,
            trade_allowed=trade_allowed,
        )

    def test_edge_flip_close(self) -> None:
        """Consensus implies opposite side → CLOSE."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.40)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE
        assert any("edge flip" in r for r in decision.reasons)

    def test_edge_flip_buy_no(self) -> None:
        """BUY_NO with consensus > 0.50 → edge flip → CLOSE."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_NO")
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.60)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE

    def test_ambiguity_veto_first_strike_holds(self) -> None:
        """First aggregation veto → HOLD (consecutive veto required)."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.HOLD
        assert "first_veto_hold" in decision.reasons

    def test_ambiguity_veto_consecutive_closes(self) -> None:
        """Two consecutive vetoes → CLOSE."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        current_agg = self._make_agg(veto=True, trade_allowed=False)
        prior_agg = self._make_agg(veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.CLOSE
        assert any("consecutive" in r for r in decision.reasons)

    def test_trade_not_allowed_first_strike_holds(self) -> None:
        """First trade_allowed=False → HOLD."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.HOLD

    def test_trade_not_allowed_consecutive_closes(self) -> None:
        """Two consecutive trade_allowed=False → CLOSE."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        current_agg = self._make_agg(trade_allowed=False)
        prior_agg = self._make_agg(trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.CLOSE

    def test_take_profit_close(self) -> None:
        """Small edge remaining + confidence drop → CLOSE."""
        policy = Policy(take_profit_band=0.02, confidence_drop_threshold=0.15)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.61)
        current_agg = self._make_agg(p_consensus=0.60, confidence=0.60)
        prior_agg = self._make_agg(p_consensus=0.60, confidence=0.80)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.CLOSE
        assert any("take-profit" in r for r in decision.reasons)

    def test_take_profit_needs_both_conditions(self) -> None:
        """Take-profit requires BOTH small edge AND confidence drop."""
        policy = Policy(take_profit_band=0.02, confidence_drop_threshold=0.15)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.61)
        # Confidence didn't drop enough.
        current_agg = self._make_agg(p_consensus=0.60, confidence=0.75)
        prior_agg = self._make_agg(p_consensus=0.60, confidence=0.80)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.HOLD

    def test_disagreement_block_close(self) -> None:
        """Disagreement >= block threshold → CLOSE."""
        policy = Policy(disagreement_block_threshold=0.15)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.60, disagreement=0.16)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE

    def test_disagreement_increase_reduce(self) -> None:
        """Disagreement increased above penalty start → REDUCE."""
        policy = Policy(disagreement_block_threshold=0.15, disagreement_size_penalty_start=0.08)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        current_agg = self._make_agg(p_consensus=0.60, disagreement=0.10)
        prior_agg = self._make_agg(p_consensus=0.60, disagreement=0.05)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.REDUCE

    def test_disagreement_no_reduce_if_below_penalty_start(self) -> None:
        """Disagreement increased but below penalty start → HOLD."""
        policy = Policy(disagreement_block_threshold=0.15, disagreement_size_penalty_start=0.08)
        lc = PositionLifecycle(policy)
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        current_agg = self._make_agg(p_consensus=0.60, disagreement=0.06)
        prior_agg = self._make_agg(p_consensus=0.60, disagreement=0.03)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.HOLD

    def test_hold_normal(self) -> None:
        """No issues → HOLD."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.55)
        agg = self._make_agg(p_consensus=0.60, disagreement=0.02)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.HOLD

    def test_deterministic_stop_loss_takes_priority(self) -> None:
        """Stop loss fires before aggregation checks."""
        lc = PositionLifecycle(Policy(edge_threshold=0.05))
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.35)  # big drop
        agg = self._make_agg(p_consensus=0.60)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE
        assert any("stop loss" in r for r in decision.reasons)

    def test_no_snapshot_closes_before_aggregation(self) -> None:
        """No snapshot → CLOSE (deterministic path)."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        agg = self._make_agg(p_consensus=0.60)
        decision = lc.evaluate_with_aggregation(pos, None, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE


# ---- Minimum Hold Time (new) -----------------------------------------------


class TestMinHoldTime:
    """Minimum hold time blocks aggregation-based exits for young positions."""

    def _make_agg(
        self,
        p_consensus: float = 0.60,
        confidence: float = 0.80,
        disagreement: float = 0.02,
        veto: bool = False,
        trade_allowed: bool = True,
    ) -> AggregationResult:
        return AggregationResult(
            p_consensus=p_consensus,
            confidence=confidence,
            disagreement=disagreement,
            veto=veto,
            trade_allowed=trade_allowed,
        )

    def test_young_position_veto_holds(self) -> None:
        """Position < min_hold_minutes old, veto → HOLD."""
        policy = Policy(min_hold_minutes=60)
        lc = PositionLifecycle(policy)
        pos = _make_position(
            side="BUY_YES",
            opened_ts_utc=datetime.now(timezone.utc) - timedelta(minutes=30),
        )
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(veto=True, trade_allowed=False)
        prior_agg = self._make_agg(veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, prior_agg)
        assert decision.action == LifecycleAction.HOLD
        assert "min_hold_not_met" in decision.reasons

    def test_old_position_veto_closes(self) -> None:
        """Position > min_hold_minutes old, consecutive veto → CLOSE."""
        policy = Policy(min_hold_minutes=60)
        lc = PositionLifecycle(policy)
        pos = _make_position(
            side="BUY_YES",
            opened_ts_utc=datetime.now(timezone.utc) - timedelta(minutes=120),
        )
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(veto=True, trade_allowed=False)
        prior_agg = self._make_agg(veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, prior_agg)
        assert decision.action == LifecycleAction.CLOSE

    def test_young_position_hard_safety_still_fires(self) -> None:
        """Stop loss fires even for young positions (deterministic check)."""
        policy = Policy(min_hold_minutes=60, edge_threshold=0.05)
        lc = PositionLifecycle(policy)
        pos = _make_position(
            side="BUY_YES",
            opened_ts_utc=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        snap = _make_snapshot(mid=0.35)  # big drop → stop loss
        agg = self._make_agg(p_consensus=0.60)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE
        assert any("stop loss" in r for r in decision.reasons)

    def test_young_position_approaching_resolution_still_fires(self) -> None:
        """Approaching resolution fires even for young positions."""
        policy = Policy(min_hold_minutes=60, min_hours_to_resolution=24)
        lc = PositionLifecycle(policy)
        pos = _make_position(
            side="BUY_YES",
            opened_ts_utc=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(p_consensus=0.60)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 12.0, agg, None)
        assert decision.action == LifecycleAction.CLOSE
        assert any("hours_to_resolution" in r for r in decision.reasons)

    def test_min_hold_zero_disables_check(self) -> None:
        """min_hold_minutes=0 disables the hold check."""
        policy = Policy(min_hold_minutes=0)
        lc = PositionLifecycle(policy)
        pos = _make_position(
            side="BUY_YES",
            opened_ts_utc=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(veto=True, trade_allowed=False)
        prior_agg = self._make_agg(veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, prior_agg)
        # With min_hold=0, position is old enough → consecutive veto → CLOSE.
        assert decision.action == LifecycleAction.CLOSE


# ---- Consecutive Veto (new) -------------------------------------------------


class TestConsecutiveVeto:
    """Consecutive veto requirement for closing positions."""

    def _make_agg(
        self,
        veto: bool = False,
        trade_allowed: bool = True,
        **kwargs,
    ) -> AggregationResult:
        return AggregationResult(
            p_consensus=kwargs.get("p_consensus", 0.60),
            confidence=kwargs.get("confidence", 0.80),
            disagreement=kwargs.get("disagreement", 0.02),
            veto=veto,
            trade_allowed=trade_allowed,
            veto_reasons=kwargs.get("veto_reasons", []),
        )

    def test_first_veto_holds(self) -> None:
        """First veto → HOLD."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        agg = self._make_agg(veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, agg, None)
        assert decision.action == LifecycleAction.HOLD
        assert "first_veto_hold" in decision.reasons

    def test_second_consecutive_veto_closes(self) -> None:
        """Two consecutive vetoes → CLOSE."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        current_agg = self._make_agg(veto=True, trade_allowed=False)
        prior_agg = self._make_agg(veto=True, trade_allowed=False)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.CLOSE

    def test_non_veto_between_resets_counter(self) -> None:
        """Non-veto prior → first veto → HOLD (counter reset)."""
        lc = PositionLifecycle(Policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50)
        current_agg = self._make_agg(veto=True, trade_allowed=False)
        # Prior was NOT a veto.
        prior_agg = self._make_agg(veto=False, trade_allowed=True)
        decision = lc.evaluate_with_aggregation(pos, snap, 0.50, 100.0, current_agg, prior_agg)
        assert decision.action == LifecycleAction.HOLD
        assert "first_veto_hold" in decision.reasons


# ---- Effective Confidence (Change 1 soft gate) ------------------------------


class TestEffectiveConfidence:
    """Effective confidence replaces raw confidence in sizing formula."""

    def test_full_confidence_unchanged(self) -> None:
        """Confidence >= min_confidence_full → effective_conf=1.0."""
        policy = Policy(min_confidence_hard=0.25, min_confidence_full=0.60)
        inp = SizingInput(p_consensus=0.70, p_market=0.50, confidence=0.80, disagreement=0.0)
        result = compute_size(inp, policy)
        # effective_conf=1.0 → same as confidence=1.0
        inp_one = SizingInput(p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.0)
        result_one = compute_size(inp_one, policy)
        assert result.raw_size_eur == pytest.approx(result_one.raw_size_eur)

    def test_mid_confidence_ramps(self) -> None:
        """Confidence between hard and full → effective_conf in (0, 1)."""
        policy = Policy(min_confidence_hard=0.25, min_confidence_full=0.60)
        inp = SizingInput(p_consensus=0.70, p_market=0.50, confidence=0.425, disagreement=0.0)
        result = compute_size(inp, policy)
        # effective_conf = (0.425 - 0.25) / 0.35 = 0.5
        inp_one = SizingInput(p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.0)
        result_one = compute_size(inp_one, policy)
        assert result.raw_size_eur == pytest.approx(result_one.raw_size_eur * 0.5, rel=0.01)

    def test_below_hard_min_zero_size(self) -> None:
        """Confidence below min_confidence_hard → effective_conf=0 → tiny size."""
        policy = Policy(min_confidence_hard=0.25, min_confidence_full=0.60)
        inp = SizingInput(p_consensus=0.70, p_market=0.50, confidence=0.20, disagreement=0.0)
        result = compute_size(inp, policy)
        # effective_conf clamped to 0.0 → scaled=0 → skip below minimum
        assert result.raw_size_eur == pytest.approx(0.0)

    def test_at_hard_min_boundary(self) -> None:
        """Confidence exactly at hard min → effective_conf=0."""
        policy = Policy(min_confidence_hard=0.25, min_confidence_full=0.60)
        inp = SizingInput(p_consensus=0.70, p_market=0.50, confidence=0.25, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.raw_size_eur == pytest.approx(0.0)

    def test_at_full_boundary(self) -> None:
        """Confidence exactly at full min → effective_conf=1.0."""
        policy = Policy(min_confidence_hard=0.25, min_confidence_full=0.60)
        inp = SizingInput(p_consensus=0.70, p_market=0.50, confidence=0.60, disagreement=0.0)
        result = compute_size(inp, policy)
        inp_one = SizingInput(p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.0)
        result_one = compute_size(inp_one, policy)
        assert result.raw_size_eur == pytest.approx(result_one.raw_size_eur)


# ---- Entry Price Filter (Change 2) -----------------------------------------


class TestEntryPriceFilter:
    """Entry price filter blocks expensive entries."""

    def test_buy_yes_high_price_blocked(self) -> None:
        """BUY_YES at 0.95 (entry_price=0.95 > 0.90) → blocked."""
        policy = Policy(max_entry_price=0.90, edge_threshold=0.01)
        inp = SizingInput(p_consensus=0.99, p_market=0.95, confidence=0.80, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "entry_price" in result.skip_reason

    def test_buy_no_high_price_blocked(self) -> None:
        """BUY_NO at 0.05 (entry_price=0.95 > 0.90) → blocked."""
        policy = Policy(max_entry_price=0.90, edge_threshold=0.01)
        inp = SizingInput(p_consensus=0.01, p_market=0.05, confidence=0.80, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "entry_price" in result.skip_reason

    def test_normal_price_not_blocked(self) -> None:
        """Entry price 0.50 → passes filter."""
        policy = Policy(max_entry_price=0.90)
        inp = SizingInput(p_consensus=0.70, p_market=0.50, confidence=0.80, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is None

    def test_massive_edge_override(self) -> None:
        """High entry price allowed with massive edge + full confidence + low disagreement."""
        policy = Policy(
            max_entry_price=0.90,
            min_confidence_full=0.60,
            disagreement_size_penalty_start=0.08,
            edge_threshold=0.01,
        )
        # entry_price=0.92, edge=0.07+0.92=0.99-0.92? No:
        # p_consensus=0.99, p_market=0.92 → edge_yes=0.07, edge_no=-0.07
        # We need edge >= 0.15. p_consensus=0.99, p_market=0.80 → edge=0.19 > 0.15.
        # entry_price for BUY_YES = 0.80... that's not > 0.90. Need p_market > 0.90.
        # p_market=0.92, p_consensus=0.99 → edge_yes=0.07. Not >= 0.15.
        # p_market=0.92, edge_yes = p_consensus - 0.92 >= 0.15 → p_consensus >= 1.07. Not possible.
        # So massive edge override at very high prices is extremely rare.
        # Test with BUY_NO: p_market=0.92, entry_price=0.08 < 0.90. That passes normally.
        # Use: p_market=0.08, side=BUY_NO, entry_price=1-0.08=0.92 > 0.90.
        # edge_no = 0.08 - p_consensus. Need edge >= 0.15 → p_consensus <= -0.07. Not possible.
        # Conclusion: massive edge override at entry_price > 0.90 is nearly impossible
        # because edge >= 0.15 requires the other side to have huge edge.
        # Still test the logic with a custom max_entry_price to make override feasible.
        policy2 = Policy(
            max_entry_price=0.60,
            min_confidence_full=0.60,
            disagreement_size_penalty_start=0.08,
            edge_threshold=0.01,
        )
        inp = SizingInput(
            p_consensus=0.90, p_market=0.70, confidence=0.80, disagreement=0.02,
        )
        # entry_price=0.70 > 0.60, edge=0.20 >= 0.15, conf=0.80 >= 0.60, disagree=0.02 <= 0.08
        result = compute_size(inp, policy2)
        assert result.skip_reason is None

    def test_massive_edge_override_fails_low_confidence(self) -> None:
        """Override denied when confidence is below min_confidence_full."""
        policy = Policy(
            max_entry_price=0.60,
            min_confidence_full=0.60,
            disagreement_size_penalty_start=0.08,
            edge_threshold=0.01,
        )
        inp = SizingInput(
            p_consensus=0.90, p_market=0.70, confidence=0.50, disagreement=0.02,
        )
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "entry_price" in result.skip_reason

    def test_massive_edge_override_fails_high_disagreement(self) -> None:
        """Override denied when disagreement exceeds penalty start."""
        policy = Policy(
            max_entry_price=0.60,
            min_confidence_full=0.60,
            disagreement_size_penalty_start=0.08,
            edge_threshold=0.01,
        )
        inp = SizingInput(
            p_consensus=0.90, p_market=0.70, confidence=0.80, disagreement=0.10,
        )
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "entry_price" in result.skip_reason


# ---- Execution Price EV (new) -----------------------------------------------


class TestExecutionPriceEV:
    """EV-based edge computation using execution prices instead of mid."""

    def test_wide_spread_eats_edge(self) -> None:
        """Gross edge 0.06, spread=0.08 → net EV ~0.02 → skipped at 5% threshold."""
        policy = Policy(edge_threshold=0.05)
        # p_consensus=0.56, mid=0.50 → gross edge 0.06
        # ask=0.54 → ev_yes = 0.56 - (0.54 + 0.005) = 0.015 < 0.05
        inp = SizingInput(
            p_consensus=0.56, p_market=0.50, confidence=1.0, disagreement=0.0,
            best_bid=0.46, best_ask=0.54,
        )
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "below threshold" in result.skip_reason
        assert result.gross_edge == pytest.approx(0.06)

    def test_narrow_spread_preserves_edge(self) -> None:
        """Gross edge 0.08, narrow spread → net EV above threshold → allowed."""
        policy = Policy(edge_threshold=0.05)
        # ask=0.51 → ev_yes = 0.58 - (0.51 + 0.005) = 0.065 > 0.05
        inp = SizingInput(
            p_consensus=0.58, p_market=0.50, confidence=1.0, disagreement=0.0,
            best_bid=0.49, best_ask=0.51,
        )
        result = compute_size(inp, policy)
        assert result.skip_reason is None
        assert result.edge == pytest.approx(0.065)

    def test_buy_no_correctly_penalized(self) -> None:
        """BUY_NO: p_consensus=0.90, bid=0.93 → ev_no = 0.93-0.005-0.90 = 0.025."""
        policy = Policy(edge_threshold=0.01)
        inp = SizingInput(
            p_consensus=0.90, p_market=0.93, confidence=1.0, disagreement=0.0,
            best_bid=0.93, best_ask=0.95,
        )
        result = compute_size(inp, policy)
        assert result.side == "BUY_NO"
        assert result.edge == pytest.approx(0.025)

    def test_zero_bid_ask_fallback_to_mid(self) -> None:
        """Zero bid/ask falls back to mid; edge = gross - slippage."""
        policy = Policy(edge_threshold=0.01)
        inp = SizingInput(
            p_consensus=0.60, p_market=0.50, confidence=1.0, disagreement=0.0,
            best_bid=0.0, best_ask=0.0,
        )
        result = compute_size(inp, policy)
        assert result.side == "BUY_YES"
        assert result.edge == pytest.approx(0.095)  # 0.10 - 0.005
        assert result.gross_edge == pytest.approx(0.10)

    def test_sizing_uses_net_edge(self) -> None:
        """Sizing formula uses net edge (not gross) for amount calculation."""
        policy = Policy(bankroll_eur=10_000.0, base_risk_frac=0.02, edge_scale=0.20)
        inp = SizingInput(
            p_consensus=0.70, p_market=0.50, confidence=1.0, disagreement=0.0,
            best_bid=0.49, best_ask=0.51,
        )
        result = compute_size(inp, policy)
        # net edge = 0.70 - (0.51 + 0.005) = 0.185
        net_edge = 0.185
        expected_raw = 10_000.0 * 0.02 * (net_edge / 0.20)
        assert result.raw_size_eur == pytest.approx(expected_raw)


# ---- Ratio Tail Filter (new) ------------------------------------------------


class TestRatioTailFilter:
    """Ratio-based tail filter: entry < 0.10, model must >= max(0.08, 3x entry)."""

    def test_entry_003_model_008_blocked(self) -> None:
        """Entry 0.03, model 0.08 → min_model = max(0.08, 0.09) = 0.09 → blocked."""
        policy = Policy(edge_threshold=0.01)
        inp = SizingInput(p_consensus=0.08, p_market=0.03, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "tail_bet" in result.skip_reason

    def test_entry_003_model_012_allowed(self) -> None:
        """Entry 0.03, model 0.12 → min_model = 0.09 → 0.12 >= 0.09 → allowed."""
        policy = Policy(edge_threshold=0.01)
        inp = SizingInput(p_consensus=0.12, p_market=0.03, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is None or "tail_bet" not in result.skip_reason

    def test_entry_005_model_014_blocked(self) -> None:
        """Entry 0.05, model 0.14 → min_model = max(0.08, 0.15) = 0.15 → blocked."""
        policy = Policy(edge_threshold=0.01)
        inp = SizingInput(p_consensus=0.14, p_market=0.05, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "tail_bet" in result.skip_reason

    def test_entry_011_above_threshold(self) -> None:
        """Entry 0.11 → above 0.10 threshold → filter doesn't apply."""
        policy = Policy(edge_threshold=0.01)
        inp = SizingInput(p_consensus=0.15, p_market=0.11, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is None or "tail_bet" not in result.skip_reason

    def test_buy_no_at_097_entry_003(self) -> None:
        """BUY_NO at p_market=0.97 → NO entry=0.03, model_no checked."""
        policy = Policy(edge_threshold=0.01)
        # p_consensus=0.95, p_market=0.97 → BUY_NO, NO entry=0.03, model_no=0.05
        # min_model = max(0.08, 0.09) = 0.09; 0.05 < 0.09 → blocked
        inp = SizingInput(p_consensus=0.95, p_market=0.97, confidence=1.0, disagreement=0.0)
        result = compute_size(inp, policy)
        assert result.skip_reason is not None
        assert "tail_bet" in result.skip_reason

"""Tests for enhanced lifecycle rules (evaluate_with_engine)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.config.policy import Policy
from src.portfolio.lifecycle import LifecycleAction, PositionLifecycle


# ---- Helpers ----------------------------------------------------------------


def _make_position(
    market_id: str = "mkt-1",
    side: str = "BUY_YES",
    size_eur: float = 100.0,
    avg_entry_price: float = 0.50,
    opened_ts_utc: datetime | None = None,
) -> MagicMock:
    pos = MagicMock()
    pos.market_id = market_id
    pos.side = side
    pos.size_eur = size_eur
    pos.avg_entry_price = avg_entry_price
    pos.opened_ts_utc = opened_ts_utc or (
        datetime.now(timezone.utc) - timedelta(hours=2)
    )
    return pos


def _make_snapshot(
    mid: float = 0.50,
    best_bid: float = 0.48,
    best_ask: float = 0.52,
) -> MagicMock:
    snap = MagicMock()
    snap.mid = mid
    snap.best_bid = best_bid
    snap.best_ask = best_ask
    return snap


def _default_policy(**overrides) -> Policy:
    defaults = dict(
        edge_threshold=0.05,
        min_hold_minutes=30,
        exit_flip_threshold=0.02,
        take_profit_band=0.02,
        min_confidence_hard=0.25,
        slippage_bps=50,
        fee_rate=0.0,
    )
    defaults.update(overrides)
    return Policy(**defaults)


# ---- evaluate_with_engine: base rules still fire ----------------------------


class TestEvaluateWithEngineBaseRules:
    """Base deterministic rules (1-4) still trigger via evaluate_with_engine."""

    def test_close_no_snapshot(self) -> None:
        lc = PositionLifecycle(_default_policy())
        pos = _make_position()
        d = lc.evaluate_with_engine(
            pos, None, 0.50, 100.0, engine_p_yes=0.60, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("no current market snapshot" in r for r in d.reasons)

    def test_close_approaching_resolution(self) -> None:
        lc = PositionLifecycle(_default_policy(min_hours_to_resolution=24))
        pos = _make_position()
        snap = _make_snapshot(mid=0.50)
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 12.0, engine_p_yes=0.60, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE

    def test_close_stop_loss(self) -> None:
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.35)
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.60, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("stop loss" in r for r in d.reasons)


# ---- Rule 5: Min hold time gate ---------------------------------------------


class TestMinHoldTimeGate:
    """Engine-based exits are skipped for fresh positions."""

    def test_fresh_position_holds(self) -> None:
        """Position opened 5 minutes ago should HOLD even if edge flipped."""
        lc = PositionLifecycle(_default_policy(min_hold_minutes=30))
        pos = _make_position(
            side="BUY_YES",
            opened_ts_utc=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        snap = _make_snapshot(mid=0.50)
        # Engine says p_yes=0.30 (edge flipped for BUY_YES), but position is fresh.
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.30, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.HOLD

    def test_old_position_closes_on_edge_flip(self) -> None:
        """Position opened 2 hours ago should close on edge flip."""
        lc = PositionLifecycle(_default_policy(min_hold_minutes=30))
        pos = _make_position(
            side="BUY_YES",
            opened_ts_utc=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # Engine says p_yes=0.30 => signed_edge = 0.30 - 0.50 = -0.20
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.30, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("edge_flip" in r for r in d.reasons)


# ---- Rule 6: Edge flip (signed edge) ----------------------------------------


class TestEdgeFlip:
    """Engine view has flipped against position direction."""

    def test_buy_yes_edge_flip(self) -> None:
        """BUY_YES: engine_p_yes well below market => CLOSE."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # signed_edge = 0.30 - 0.50 = -0.20, threshold ~ -(0.02 + 0.01 + 0.005) = -0.035
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.30, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("edge_flip" in r for r in d.reasons)

    def test_buy_no_edge_flip(self) -> None:
        """BUY_NO: engine_p_yes well above market => CLOSE."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_NO")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # signed_edge = 0.50 - 0.70 = -0.20
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.70, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("edge_flip" in r for r in d.reasons)

    def test_no_flip_when_edge_positive(self) -> None:
        """Positive signed edge should not trigger flip."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # signed_edge = 0.70 - 0.50 = +0.20 (good for BUY_YES)
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.70, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.HOLD


# ---- Rule 7: Take profit (convergence) --------------------------------------


class TestTakeProfit:
    """Market converged to fair value — remaining edge won't cover exit costs."""

    def test_take_profit_buy_yes(self) -> None:
        """BUY_YES where engine ≈ market => CLOSE on convergence."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # signed_edge = 0.51 - 0.50 = 0.01, cost_buffer = 0.01 + 0.005 + 0.0 = 0.015
        # abs(0.01) < 0.02 + 0.015 = 0.035 => take profit
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.51, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("take_profit" in r for r in d.reasons)

    def test_take_profit_buy_no(self) -> None:
        """BUY_NO where engine ≈ market => CLOSE on convergence."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_NO")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # signed_edge = 0.50 - 0.49 = 0.01
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.49, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("take_profit" in r for r in d.reasons)

    def test_no_take_profit_large_edge(self) -> None:
        """Large remaining edge should not trigger take profit."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # signed_edge = 0.70 - 0.50 = 0.20, well above band + costs
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.70, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.HOLD


# ---- Rule 8: Confidence collapse --------------------------------------------


class TestConfidenceCollapse:
    """Engine confidence below hard minimum triggers CLOSE."""

    def test_low_confidence_closes(self) -> None:
        lc = PositionLifecycle(_default_policy(min_confidence_hard=0.25))
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # Engine has strong edge but very low confidence.
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.70, engine_confidence=0.15,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("confidence_collapse" in r for r in d.reasons)

    def test_adequate_confidence_holds(self) -> None:
        lc = PositionLifecycle(_default_policy(min_confidence_hard=0.25))
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.70, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.HOLD


# ---- Rule priority order ----------------------------------------------------


class TestRulePriority:
    """Rules fire in order: base (1-4) > min_hold (5) > flip (6) > profit (7) > confidence (8)."""

    def test_base_rule_beats_engine(self) -> None:
        """Stop loss fires before engine rules."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.35)
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.70, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("stop loss" in r for r in d.reasons)

    def test_edge_flip_before_take_profit(self) -> None:
        """Edge flip fires before take profit when signed_edge is very negative."""
        lc = PositionLifecycle(_default_policy())
        pos = _make_position(side="BUY_YES")
        snap = _make_snapshot(mid=0.50, best_bid=0.49, best_ask=0.51)
        # signed_edge = 0.20 - 0.50 = -0.30, clearly a flip
        d = lc.evaluate_with_engine(
            pos, snap, 0.50, 100.0, engine_p_yes=0.20, engine_confidence=0.80,
        )
        assert d.action == LifecycleAction.CLOSE
        assert any("edge_flip" in r for r in d.reasons)

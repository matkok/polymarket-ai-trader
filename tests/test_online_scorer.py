"""Tests for src.evaluation.online_scorer — edge capture computation."""

from __future__ import annotations

import pytest

from src.evaluation.online_scorer import compute_edge_capture


class TestComputeEdgeCapture:
    """Edge capture and direction correctness computation."""

    def test_buy_yes_market_moves_toward_consensus(self) -> None:
        """BUY_YES: market moves in the predicted direction → positive capture."""
        edge_capture, direction_correct = compute_edge_capture(
            side="BUY_YES",
            p_consensus_at_entry=0.60,
            p_market_at_entry=0.50,
            p_market_now=0.55,
        )
        # Edge = 0.60 - 0.50 = 0.10
        # Market move = 0.55 - 0.50 = 0.05
        # Edge capture = 0.05 / 0.10 = 0.50
        assert edge_capture == pytest.approx(0.50)
        assert direction_correct is True

    def test_buy_yes_market_moves_against(self) -> None:
        """BUY_YES: market moves against prediction → negative capture."""
        edge_capture, direction_correct = compute_edge_capture(
            side="BUY_YES",
            p_consensus_at_entry=0.60,
            p_market_at_entry=0.50,
            p_market_now=0.45,
        )
        # Edge = 0.10, market move = -0.05
        # Edge capture = -0.05 / 0.10 = -0.50
        assert edge_capture == pytest.approx(-0.50)
        assert direction_correct is False

    def test_buy_no_market_moves_toward_consensus(self) -> None:
        """BUY_NO: consensus says price is too high, market drops → positive."""
        edge_capture, direction_correct = compute_edge_capture(
            side="BUY_NO",
            p_consensus_at_entry=0.40,  # consensus < market → BUY_NO
            p_market_at_entry=0.50,
            p_market_now=0.45,
        )
        # For BUY_NO: edge = -(0.40 - 0.50) = 0.10, move = -(0.45 - 0.50) = 0.05
        # Edge capture = 0.05 / 0.10 = 0.50
        assert edge_capture == pytest.approx(0.50)
        assert direction_correct is True

    def test_buy_no_market_moves_against(self) -> None:
        """BUY_NO: market rises instead of dropping → negative capture."""
        edge_capture, direction_correct = compute_edge_capture(
            side="BUY_NO",
            p_consensus_at_entry=0.40,
            p_market_at_entry=0.50,
            p_market_now=0.55,
        )
        # For BUY_NO: edge = 0.10, move = -(0.55 - 0.50) = -0.05
        # Edge capture = -0.05 / 0.10 = -0.50
        assert edge_capture == pytest.approx(-0.50)
        assert direction_correct is False

    def test_zero_edge_returns_zero(self) -> None:
        """When consensus equals market price (no edge), returns 0."""
        edge_capture, direction_correct = compute_edge_capture(
            side="BUY_YES",
            p_consensus_at_entry=0.50,
            p_market_at_entry=0.50,
            p_market_now=0.55,
        )
        assert edge_capture == pytest.approx(0.0)
        assert direction_correct is False

    def test_full_edge_capture(self) -> None:
        """Market fully converges to consensus → capture = 1.0."""
        edge_capture, direction_correct = compute_edge_capture(
            side="BUY_YES",
            p_consensus_at_entry=0.70,
            p_market_at_entry=0.50,
            p_market_now=0.70,
        )
        # Edge = 0.20, move = 0.20
        assert edge_capture == pytest.approx(1.0)
        assert direction_correct is True

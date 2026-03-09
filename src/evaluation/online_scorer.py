"""Online edge-capture scoring for open positions.

Measures how well the system's consensus probability forecast is being
validated by subsequent market price movements, without waiting for
resolution.
"""

from __future__ import annotations


def compute_edge_capture(
    side: str,
    p_consensus_at_entry: float,
    p_market_at_entry: float,
    p_market_now: float,
) -> tuple[float, bool]:
    """Compute edge capture and direction correctness for an open position.

    Parameters
    ----------
    side:
        ``"BUY_YES"`` or ``"BUY_NO"``.
    p_consensus_at_entry:
        Panel consensus probability at the time the trade was opened.
    p_market_at_entry:
        Market mid price at the time the trade was opened.
    p_market_now:
        Current market mid price.

    Returns
    -------
    (edge_capture, direction_correct)
        ``edge_capture`` is the fraction of the predicted edge that the
        market has moved in the direction of the trade.  Values > 0 mean
        the market is moving towards the consensus.  ``direction_correct``
        is True if the current price has moved in the predicted direction.
    """
    edge = p_consensus_at_entry - p_market_at_entry
    if abs(edge) < 1e-9:
        return 0.0, False

    market_move = p_market_now - p_market_at_entry

    # For BUY_NO positions, the beneficial direction is price falling.
    if side == "BUY_NO":
        market_move = -market_move
        edge = -edge

    edge_capture = market_move / edge if abs(edge) > 1e-9 else 0.0
    direction_correct = market_move > 0

    return edge_capture, direction_correct

"""Fill recording and PnL calculation.

Provides functions to compute unrealized and realized PnL for positions
based on entry price, current price, and fees.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PnLSnapshot:
    """Snapshot of a position's profit and loss."""

    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float


def calculate_unrealized_pnl(
    side: str,
    size_eur: float,
    avg_entry_price: float,
    current_price: float,
) -> float:
    """Calculate unrealized PnL for an open position.

    Models the return if the position were to be exited at
    *current_price*.

    For ``BUY_YES``: ``size_eur * (current_price - avg_entry_price) / avg_entry_price``
    For ``BUY_NO``:  ``size_eur * (avg_entry_price - current_price) / avg_entry_price``
    """
    if avg_entry_price <= 0:
        return 0.0

    if side == "BUY_YES":
        return size_eur * (current_price - avg_entry_price) / avg_entry_price
    else:
        return size_eur * (avg_entry_price - current_price) / avg_entry_price


def calculate_realized_pnl(
    side: str,
    size_eur: float,
    avg_entry_price: float,
    exit_price: float,
    fee_eur: float,
) -> float:
    """Calculate realized PnL on a closed position.

    Same formula as :func:`calculate_unrealized_pnl` but subtracting
    total fees incurred.
    """
    if avg_entry_price <= 0:
        return -fee_eur

    if side == "BUY_YES":
        gross = size_eur * (exit_price - avg_entry_price) / avg_entry_price
    else:
        gross = size_eur * (avg_entry_price - exit_price) / avg_entry_price

    return gross - fee_eur

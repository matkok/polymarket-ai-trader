"""Paper trading executor.

Simulates order fills at best bid/ask with configurable slippage and
fees.  No real orders are placed; the executor is used for back-testing
and paper-trading runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.config.policy import Policy


@dataclass
class PaperFill:
    """Simulated fill result."""

    side: str
    price: float
    size_eur: float
    fee_eur: float
    slippage_applied: float


class PaperExecutor:
    """Simulate order execution with slippage and fees."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.logger = structlog.get_logger(__name__)

    def execute(
        self,
        side: str,
        size_eur: float,
        best_bid: float,
        best_ask: float,
        fee_rate_override: float | None = None,
        order_book: list[tuple[float, float]] | None = None,
    ) -> PaperFill:
        """Simulate a fill for a paper trade.

        For buy orders the fill price is ``best_ask + slippage``.
        For sell orders the fill price is ``best_bid - slippage``.
        The price is clamped to the ``[0.01, 0.99]`` range.

        Parameters
        ----------
        side:
            ``"BUY_YES"`` or ``"BUY_NO"``.
        size_eur:
            Notional size in EUR.
        best_bid:
            Current best bid price.
        best_ask:
            Current best ask price.
        fee_rate_override:
            If provided, use this fee rate instead of the policy default.
        order_book:
            Optional list of (price, size_eur) levels from the CLOB.
            If provided, slippage is computed via VWAP walk instead of
            the constant ``slippage_bps``.
        """
        # Determine fill price.
        if order_book is not None:
            vwap = walk_order_book(order_book, size_eur)
            if vwap is not None:
                raw_price = vwap
            else:
                # Insufficient depth — fall back to constant slippage.
                slippage = self.policy.slippage_bps / 10_000
                if side in ("BUY_YES", "BUY_NO"):
                    raw_price = best_ask + slippage
                else:
                    raw_price = best_bid - slippage
        else:
            slippage = self.policy.slippage_bps / 10_000
            if side in ("BUY_YES", "BUY_NO"):
                raw_price = best_ask + slippage
            else:
                raw_price = best_bid - slippage

        # Clamp to valid probability range.
        price = max(0.01, min(0.99, raw_price))

        fee_rate = fee_rate_override if fee_rate_override is not None else self.policy.fee_rate
        fee_eur = size_eur * fee_rate

        ob_used = order_book is not None
        self.logger.info(
            "paper_fill",
            side=side,
            size_eur=size_eur,
            price=price,
            fee_eur=fee_eur,
            fee_rate=fee_rate,
            ob_depth_used=ob_used,
        )

        return PaperFill(
            side=side,
            price=price,
            size_eur=size_eur,
            fee_eur=fee_eur,
            slippage_applied=abs(price - (best_ask if side in ("BUY_YES", "BUY_NO") else best_bid)),
        )


def walk_order_book(
    entries: list[tuple[float, float]],
    target_size_eur: float,
) -> float | None:
    """Walk order book levels computing VWAP for *target_size_eur*.

    Parameters
    ----------
    entries:
        List of ``(price, size_eur)`` levels sorted by price
        (ascending for asks, descending for bids).
    target_size_eur:
        Notional amount to fill.

    Returns
    -------
    VWAP price if sufficient depth, otherwise ``None``.
    """
    if not entries or target_size_eur <= 0:
        return None

    filled = 0.0
    cost = 0.0

    for price, level_size in entries:
        take = min(level_size, target_size_eur - filled)
        cost += take * price
        filled += take
        if filled >= target_size_eur - 1e-9:
            break

    if filled < target_size_eur - 1e-9:
        return None  # Insufficient depth.

    return cost / filled

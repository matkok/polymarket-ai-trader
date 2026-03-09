"""Position sizing based on edge, confidence, and disagreement.

Implements the Kelly-inspired sizing formula from spec section 9.1.
Given a consensus probability and current market price, computes the
optimal position size subject to policy constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.policy import Policy


@dataclass
class SizingInput:
    """Inputs required to compute a position size."""

    p_consensus: float  # panel consensus probability (or market mid for M1)
    p_market: float  # current market price
    confidence: float  # consensus confidence (1.0 for M1)
    disagreement: float  # panel disagreement std dev (0.0 for M1)
    best_bid: float = 0.0  # best bid price (0 = unknown, falls back to mid)
    best_ask: float = 0.0  # best ask price (0 = unknown, falls back to mid)


@dataclass
class SizingResult:
    """Output of the sizing computation."""

    raw_size_eur: float
    clamped_size_eur: float
    edge: float  # net EV after execution costs
    side: str  # "BUY_YES" or "BUY_NO"
    skip_reason: str | None  # None if trade is valid
    gross_edge: float = 0.0  # mid-based edge before execution costs


def compute_size(inp: SizingInput, policy: Policy) -> SizingResult:
    """Compute position size based on edge, confidence, and disagreement.

    The edge is the net EV after accounting for spread and slippage using
    actual execution prices.  If below ``policy.edge_threshold`` the trade
    is skipped.

    The raw size is scaled by edge magnitude, confidence, and optionally
    penalised for high panel disagreement.  The final size is clamped to
    the per-market exposure limit and a minimum of 1.0 EUR.
    """
    slippage_frac = policy.slippage_bps / 10_000
    bid = inp.best_bid if inp.best_bid > 0 else inp.p_market
    ask = inp.best_ask if inp.best_ask > 0 else inp.p_market

    # Direction-aware EV using execution prices.
    ev_yes = inp.p_consensus - (ask + slippage_frac)
    ev_no = (bid - slippage_frac) - inp.p_consensus

    if ev_yes >= ev_no:
        side = "BUY_YES"
        edge = ev_yes
    else:
        side = "BUY_NO"
        edge = ev_no

    # Gross edge (mid-based, for logging).
    gross_edge_yes = inp.p_consensus - inp.p_market
    gross_edge_no = inp.p_market - inp.p_consensus
    gross_edge = max(gross_edge_yes, gross_edge_no)

    def skip(reason: str) -> SizingResult:
        return SizingResult(
            raw_size_eur=0.0,
            clamped_size_eur=0.0,
            edge=edge,
            side=side,
            skip_reason=reason,
            gross_edge=gross_edge,
        )

    # Ratio-based tail bet filter.
    TAIL_ENTRY_THRESHOLD = 0.10
    TAIL_MULTIPLE = 3.0
    TAIL_MIN_MODEL_PROB = 0.08

    side_entry_price = inp.p_market if side == "BUY_YES" else (1.0 - inp.p_market)
    side_model_prob = inp.p_consensus if side == "BUY_YES" else (1.0 - inp.p_consensus)

    if side_entry_price < TAIL_ENTRY_THRESHOLD:
        min_model = max(TAIL_MIN_MODEL_PROB, TAIL_MULTIPLE * side_entry_price)
        if side_model_prob < min_model:
            return skip(
                f"tail_bet: entry {side_entry_price:.4f},"
                f" model {side_model_prob:.4f} < min {min_model:.4f}"
            )

    # Entry price filter.
    entry_price = inp.p_market if side == "BUY_YES" else (1.0 - inp.p_market)
    if entry_price > policy.max_entry_price:
        massive_edge_override = (
            edge >= 0.15
            and inp.confidence >= policy.min_confidence_full
            and inp.disagreement <= policy.disagreement_size_penalty_start
        )
        if not massive_edge_override:
            return skip(f"entry_price {entry_price:.4f} > max {policy.max_entry_price}")

    # Skip if edge is below threshold.
    if edge < policy.edge_threshold:
        return skip(f"edge {edge:.4f} below threshold {policy.edge_threshold}")

    # Skip if disagreement exceeds block threshold.
    if inp.disagreement >= policy.disagreement_block_threshold:
        return skip(
            f"disagreement {inp.disagreement:.4f} "
            f">= block threshold {policy.disagreement_block_threshold}"
        )

    # Effective confidence: soft gate ramps from 0→1 between hard and full.
    if inp.confidence < policy.min_confidence_full:
        effective_conf = (inp.confidence - policy.min_confidence_hard) / (
            policy.min_confidence_full - policy.min_confidence_hard
        )
        effective_conf = max(0.0, min(1.0, effective_conf))
    else:
        effective_conf = 1.0

    # Base size scaled by edge magnitude and effective confidence.
    base = policy.bankroll_eur * policy.base_risk_frac
    scaled = base * (edge / policy.edge_scale) * effective_conf

    # Apply disagreement penalty if above the start threshold.
    if inp.disagreement > policy.disagreement_size_penalty_start:
        penalty_range = (
            policy.disagreement_block_threshold
            - policy.disagreement_size_penalty_start
        )
        if penalty_range > 0:
            penalty = (
                inp.disagreement - policy.disagreement_size_penalty_start
            ) / penalty_range
            scaled *= max(0.0, 1.0 - penalty)

    raw_size_eur = scaled

    # Clamp to per-market exposure limit.
    max_per_market = policy.bankroll_eur * policy.max_exposure_per_market_frac
    clamped = min(raw_size_eur, max_per_market)

    # Skip if below minimum trade size.
    if clamped < 1.0:
        return SizingResult(
            raw_size_eur=raw_size_eur,
            clamped_size_eur=0.0,
            edge=edge,
            side=side,
            skip_reason=f"clamped size {clamped:.2f} EUR below minimum 1.00 EUR",
            gross_edge=gross_edge,
        )

    return SizingResult(
        raw_size_eur=raw_size_eur,
        clamped_size_eur=clamped,
        edge=edge,
        side=side,
        skip_reason=None,
        gross_edge=gross_edge,
    )

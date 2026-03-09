"""Model attribution metrics for evaluating panel member contributions.

Measures how each model's proposals contributed to trading outcomes:
support value, dissent value, veto value, and sizing error.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TradeOutcome:
    """A resolved trade with model proposals for attribution."""

    market_id: str
    position_side: str  # "BUY_YES" or "BUY_NO"
    realized_pnl: float
    profitable: bool
    model_proposals: list[ProposalSummary] = field(default_factory=list)


@dataclass
class ProposalSummary:
    """Subset of a ModelProposal needed for attribution."""

    model_id: str
    direction: str  # "BUY_YES" or "BUY_NO"
    p_true: float
    confidence: float
    ambiguity_score: float
    recommended_max_exposure_frac: float


@dataclass
class ModelAttribution:
    """Attribution metrics for a single model across resolved trades."""

    model_id: str
    trades_evaluated: int
    support_value: float  # fraction of profitable trades where model agreed
    dissent_value: float  # fraction of losing trades where model dissented
    veto_value: float  # fraction of losing trades where model flagged ambiguity
    sizing_error: float  # average excess exposure recommendation on losing trades


def compute_attribution(
    model_id: str,
    outcomes: list[TradeOutcome],
    ambiguity_threshold: float = 0.70,
) -> ModelAttribution:
    """Compute attribution metrics for a single model.

    Parameters
    ----------
    model_id:
        The model to evaluate.
    outcomes:
        List of resolved trades with model proposals.
    ambiguity_threshold:
        Ambiguity score at or above which a model is considered to have
        flagged the trade as ambiguous.
    """
    profitable_trades: list[TradeOutcome] = []
    losing_trades: list[TradeOutcome] = []
    trades_with_model = 0

    for outcome in outcomes:
        model_props = [p for p in outcome.model_proposals if p.model_id == model_id]
        if not model_props:
            continue
        trades_with_model += 1
        if outcome.profitable:
            profitable_trades.append(outcome)
        else:
            losing_trades.append(outcome)

    if trades_with_model == 0:
        return ModelAttribution(
            model_id=model_id,
            trades_evaluated=0,
            support_value=0.0,
            dissent_value=0.0,
            veto_value=0.0,
            sizing_error=0.0,
        )

    # Support value: fraction of profitable trades where model direction matched.
    support_count = 0
    for outcome in profitable_trades:
        for p in outcome.model_proposals:
            if p.model_id == model_id and p.direction == outcome.position_side:
                support_count += 1
                break
    support_value = support_count / len(profitable_trades) if profitable_trades else 0.0

    # Dissent value: fraction of losing trades where model direction differed.
    dissent_count = 0
    for outcome in losing_trades:
        for p in outcome.model_proposals:
            if p.model_id == model_id and p.direction != outcome.position_side:
                dissent_count += 1
                break
    dissent_value = dissent_count / len(losing_trades) if losing_trades else 0.0

    # Veto value: fraction of losing trades where model flagged high ambiguity.
    veto_count = 0
    for outcome in losing_trades:
        for p in outcome.model_proposals:
            if p.model_id == model_id and p.ambiguity_score >= ambiguity_threshold:
                veto_count += 1
                break
    veto_value = veto_count / len(losing_trades) if losing_trades else 0.0

    # Sizing error: average excess exposure recommendation on losing trades.
    sizing_errors: list[float] = []
    for outcome in losing_trades:
        for p in outcome.model_proposals:
            if p.model_id == model_id:
                sizing_errors.append(p.recommended_max_exposure_frac)
                break
    sizing_error = sum(sizing_errors) / len(sizing_errors) if sizing_errors else 0.0

    return ModelAttribution(
        model_id=model_id,
        trades_evaluated=trades_with_model,
        support_value=support_value,
        dissent_value=dissent_value,
        veto_value=veto_value,
        sizing_error=sizing_error,
    )

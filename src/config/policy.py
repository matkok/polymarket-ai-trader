"""Trading policy model and helpers.

The policy file (``policy.yaml``) contains deterministic portfolio and risk
limits that govern every trading decision.  A version hash is stored alongside
each decision for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel


class Policy(BaseModel):
    """Deterministic portfolio and risk limits (spec section 4.2).

    All fractional fields are expressed as fractions of ``bankroll_eur``
    unless otherwise noted.  Currency is EUR.
    """

    # Capital
    bankroll_eur: float = 10_000.0
    cash_reserve_target_frac: float = 0.20
    max_total_exposure_frac: float = 0.60
    max_exposure_per_market_frac: float = 0.08
    max_open_positions: int = 15
    max_daily_loss_frac: float = 0.05
    max_daily_drawdown_frac: float = 0.08
    min_liquidity_eur: float = 5_000.0

    # Edge and sizing
    edge_threshold: float = 0.05
    base_risk_frac: float = 0.02
    edge_scale: float = 0.20

    # Confidence gate
    min_confidence_hard: float = 0.25
    min_confidence_full: float = 0.60

    # Entry price filter
    max_entry_price: float = 0.90

    # Disagreement
    disagreement_block_threshold: float = 0.15
    disagreement_size_penalty_start: float = 0.08

    # Ambiguity
    ambiguity_veto_threshold: float = 0.85
    veto_quorum: int = 2

    # Position reentry cooldown
    position_reentry_cooldown_hours: float = 6

    # Position stability
    min_hold_minutes: int = 60
    veto_exit_consecutive_required: int = 2
    no_add_if_recent_veto_minutes: int = 120

    # Recheck triggers
    odds_move_recheck_threshold: float = 0.05
    new_evidence_recheck_window_minutes: int = 120

    # Panel cooldown
    panel_cooldown_hours: float = 8  # Skip re-paneling a market for N hours after last panel

    # Cycle limits
    max_candidates_per_cycle: int = 15
    max_panel_markets_per_day: int = 20

    # Position review (M4)
    reduce_fraction: float = 0.50
    dust_position_eur: float = 1.0
    take_profit_band: float = 0.02
    confidence_drop_threshold: float = 0.15

    # Execution
    slippage_bps: int = 50
    fee_rate: float = 0.0
    crypto_15min_fee_rate: float = 0.02

    # Market filters
    min_volume: float = 10_000.0
    min_hours_to_resolution: int = 24
    max_hours_to_resolution: int = 2160  # 90 days
    min_rules_text_length: int = 50

    # Evidence & packets
    max_evidence_items_per_packet: int = 10
    evidence_excerpt_max_chars: int = 500
    evidence_similarity_threshold: float = 0.35

    # LLM token limits
    max_input_tokens_per_call: int = 4000
    max_output_tokens_per_call: int = 1000

    # Triage
    triage_panel_threshold: float = 0.30
    triage_odds_move_threshold: float = 0.08
    triage_volume_surge_threshold: float = 1.5
    triage_trends_spike_threshold: float = 2.0
    triage_wiki_spike_threshold: float = 2.0
    triage_max_spread: float = 0.10
    triage_wide_spread_threshold: float = 0.08
    triage_spread_widening_threshold: float = 2.0
    triage_max_markets_for_signals: int = 50


def load_policy(path: str) -> Policy:
    """Load a :class:`Policy` from a YAML file at *path*.

    Raises :class:`FileNotFoundError` if the file does not exist and
    :class:`pydantic.ValidationError` if the content is invalid.
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        data = {}
    return Policy.model_validate(data)


def policy_version_hash(policy: Policy) -> str:
    """Return the SHA-256 hex digest of the canonical JSON representation.

    The canonical form uses sorted keys and no extra whitespace so that
    the hash is stable across Python versions and serialisation order.
    """
    canonical = policy.model_dump_json(indent=None)
    # model_dump_json already produces deterministic output; re-serialise
    # through json to guarantee sorted keys.
    obj = json.loads(canonical)
    canonical_sorted = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_sorted.encode("utf-8")).hexdigest()

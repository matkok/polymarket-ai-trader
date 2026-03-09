"""Trading policy model and helpers.

The policy file (``policy.yaml``) contains deterministic portfolio and risk
limits that govern every trading decision.  A version hash is stored alongside
each decision for reproducibility.

Supports per-category overrides via ``CategoryPolicy``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class CategoryPolicy(BaseModel):
    """Per-category policy overrides.

    Fields set to ``None`` inherit from the global :class:`Policy`.
    """

    enabled: bool = True
    bankroll_eur: float | None = None
    max_total_exposure_frac: float | None = None
    max_exposure_per_market_frac: float | None = None
    max_open_positions: int | None = None
    max_daily_loss_frac: float | None = None
    edge_threshold: float | None = None
    min_liquidity_eur: float | None = None
    min_volume: float | None = None
    max_hours_to_resolution: int | None = None
    take_profit_band: float | None = None
    min_hold_minutes: int | None = None
    exit_flip_threshold: float | None = None
    reentry_cooldown_hours: float | None = None
    engine_stale_hours: float | None = None
    engine_params: dict[str, Any] = {}


class Policy(BaseModel):
    """Deterministic portfolio and risk limits.

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

    # Position review
    reduce_fraction: float = 0.50
    dust_position_eur: float = 1.0

    # Lifecycle
    take_profit_band: float = 0.02
    min_hold_minutes: int = 30
    exit_flip_threshold: float = 0.02
    reentry_cooldown_hours: float = 6.0
    engine_stale_hours: float = 1.0

    # Execution
    slippage_bps: int = 50
    fee_rate: float = 0.0

    # Market filters
    min_volume: float = 10_000.0
    min_hours_to_resolution: int = 24
    max_hours_to_resolution: int = 2160  # 90 days

    # Category overrides
    categories: dict[str, CategoryPolicy] = {}

    def for_category(self, category: str) -> Policy:
        """Return a copy of this policy with category overrides applied.

        Fields in the matching :class:`CategoryPolicy` that are not ``None``
        replace the corresponding global values.  The returned policy has
        an empty ``categories`` dict (no nesting).
        """
        cat_policy = self.categories.get(category)
        if cat_policy is None:
            return self.model_copy(update={"categories": {}})

        overrides: dict[str, Any] = {}
        for field_name in [
            "bankroll_eur",
            "max_total_exposure_frac",
            "max_exposure_per_market_frac",
            "max_open_positions",
            "max_daily_loss_frac",
            "edge_threshold",
            "min_liquidity_eur",
            "min_volume",
            "max_hours_to_resolution",
            "take_profit_band",
            "min_hold_minutes",
            "exit_flip_threshold",
            "reentry_cooldown_hours",
            "engine_stale_hours",
        ]:
            value = getattr(cat_policy, field_name)
            if value is not None:
                overrides[field_name] = value

        overrides["categories"] = {}
        return self.model_copy(update=overrides)


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

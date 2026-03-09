"""LLM panel schemas — ModelProposal, panel agents, and legacy tier mappings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class EvidenceCitation(BaseModel):
    """A single evidence citation within a model proposal."""

    url: str
    claim: str
    strength: float = Field(ge=0.0, le=1.0)


class ModelProposal(BaseModel):
    """Structured output from a single LLM panel member."""

    # Metadata — injected by the orchestrator, not by the LLM.
    model_id: str = ""
    run_id: str = ""
    market_id: str = ""
    ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Probability vote (soft vote)
    p_true: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    direction: str  # "BUY_YES" or "BUY_NO"

    # Ambiguity vote (hard vote) — two separate dimensions
    rules_ambiguity: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_ambiguity: float = Field(ge=0.0, le=1.0, default=0.0)

    # Sizing & horizon
    recommended_max_exposure_frac: float = Field(ge=0.0, le=1.0)
    hold_horizon_hours: float = Field(ge=0.0)

    # Analysis
    thesis: str
    key_risks: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    exit_triggers: list[str] = Field(default_factory=list)
    notes: str = ""


class EscalationTrigger(str, Enum):
    """Reason for escalating to a stronger model."""

    DISAGREEMENT = "disagreement"
    RULES_AMBIGUITY = "rules_ambiguity"
    HIGH_STAKES = "high_stakes"
    FAST_ODDS_MOVE = "fast_odds_move"


@dataclass
class PanelAgent:
    """Configuration for a single panel agent."""

    agent_id: str       # "rules_lawyer", "probabilist", etc.
    provider: str       # "anthropic", "openai", "gemini", "xai"
    model: str          # Exact API model name
    role: str           # "rules", "probability", "sanity", "signals", "arbiter"
    always_on: bool     # True = runs every cycle


@dataclass
class ModelTierConfig:
    """A single tier in a provider's model ladder."""

    model: str      # API model name
    cap_eur: float  # Cumulative spend cap to switch away from this tier


MODEL_LADDER: dict[str, list[ModelTierConfig]] = {
    "openai": [
        ModelTierConfig("gpt-5-mini", 2.00),
        ModelTierConfig("gpt-4.1-mini", 4.00),
    ],
    "anthropic": [
        ModelTierConfig("claude-sonnet-4-20250514", 2.50),
        ModelTierConfig("claude-haiku-4-5-20251001", 4.50),
    ],
    "google": [
        ModelTierConfig("gemini-2.5-flash", 1.00),
        ModelTierConfig("gemini-2.5-flash", 3.00),
    ],
    "xai": [
        ModelTierConfig("grok-3-fast", 0.50),
        ModelTierConfig("grok-3-fast", 2.50),
    ],
}


DEFAULT_PANEL: list[PanelAgent] = [
    PanelAgent("rules_lawyer",   "anthropic", "claude-sonnet-4-20250514",  "rules",       True),
    PanelAgent("probabilist",    "openai",    "gpt-5-mini",               "probability", True),
    PanelAgent("sanity_checker", "gemini",    "gemini-2.5-flash",         "sanity",      True),
    PanelAgent("x_signals",     "xai",        "grok-3-fast",              "signals",     False),
]

ESCALATION_AGENTS: list[PanelAgent] = [
    PanelAgent("escalation_openai",    "openai",    "gpt-5.2",                "probability", False),
    PanelAgent("escalation_anthropic", "anthropic", "claude-sonnet-4-20250514", "rules",       False),
    PanelAgent("escalation_google",    "gemini",    "gemini-2.5-pro",         "arbiter",     False),
]


@dataclass
class PanelResult:
    """Result from running the full LLM panel for a single market."""

    proposals: list[ModelProposal]
    agents_used: list[str] = field(default_factory=list)
    skipped_agents: list[str] = field(default_factory=list)
    escalation_trigger: EscalationTrigger | None = None
    escalation_agent: str | None = None
    total_cost_eur: float = 0.0


# ---- Legacy tier mappings (deprecated, kept for backward compat) -----------


class ModelTier(str, Enum):
    """LLM panel tier — controls model selection and cost. DEPRECATED."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


TIER_MODELS: dict[str, dict[str, str]] = {
    "openai": {
        "tier_1": "gpt-4.1-mini",
        "tier_2": "gpt-4.1",
        "tier_3": "o4-mini",
    },
    "anthropic": {
        "tier_1": "claude-haiku-4-5-20251001",
        "tier_2": "claude-sonnet-4-5-20250929",
        "tier_3": "claude-sonnet-4-5-20250929",
    },
    "gemini": {
        "tier_1": "gemini-2.5-flash",
        "tier_2": "gemini-2.5-pro",
        "tier_3": "gemini-2.5-pro",
    },
    "xai": {
        "tier_1": "grok-3-fast",
        "tier_2": "grok-3",
        "tier_3": "grok-3",
    },
}

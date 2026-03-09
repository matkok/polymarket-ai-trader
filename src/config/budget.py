"""LLM provider budget tracking.

Tracks cumulative EUR spend per provider against daily caps to prevent
runaway API costs.  Methods are synchronous because the Python GIL
protects dict operations within asyncio.
"""

from pydantic import BaseModel


class DailyBudget(BaseModel):
    """Daily EUR spend caps per LLM provider."""

    daily_eur_cap_openai: float = 4.00
    daily_eur_cap_anthropic: float = 4.50
    daily_eur_cap_google: float = 3.00
    daily_eur_cap_xai: float = 2.50
    daily_eur_cap_total: float = 14.00


class BudgetTracker:
    """Track cumulative spend per provider against daily caps."""

    # Map provider names used at runtime to budget field suffixes.
    _PROVIDER_ALIASES: dict[str, str] = {"gemini": "google"}

    def __init__(self, budget: DailyBudget) -> None:
        self.budget = budget
        self.spent: dict[str, float] = {}

    def _cap_key(self, provider: str) -> str:
        """Resolve provider name to DailyBudget field suffix."""
        return self._PROVIDER_ALIASES.get(provider, provider)

    def can_spend(self, provider: str, estimated_cost: float) -> bool:
        """Return whether *provider* can afford *estimated_cost* EUR."""
        current = self.spent.get(provider, 0.0)
        cap = getattr(self.budget, f"daily_eur_cap_{self._cap_key(provider)}", 0.0)
        if current + estimated_cost > cap:
            return False
        if sum(self.spent.values()) + estimated_cost > self.budget.daily_eur_cap_total:
            return False
        return True

    def select_model(self, provider: str) -> str | None:
        """Return the model to use for *provider* based on current spend tier.

        Walks the provider's model ladder and returns the model for the
        current spend level.  Returns None if all tiers are exhausted.
        """
        from src.llm.schemas import MODEL_LADDER

        cap_key = self._cap_key(provider)
        ladder = MODEL_LADDER.get(cap_key, [])
        if not ladder:
            return None
        current = self.spent.get(provider, 0.0)
        for tier in ladder:
            if current < tier.cap_eur:
                return tier.model
        return None

    def record_spend(self, provider: str, cost: float) -> None:
        """Record *cost* EUR spent by *provider*."""
        self.spent[provider] = self.spent.get(provider, 0.0) + cost

    def reset_daily(self) -> None:
        """Reset all provider spend counters for a new day."""
        self.spent.clear()

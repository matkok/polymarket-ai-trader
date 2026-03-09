"""Tests for src.config.budget — BudgetTracker."""

from __future__ import annotations

from src.config.budget import BudgetTracker, DailyBudget


# ---- BudgetTracker -----------------------------------------------------------


class TestBudgetTracker:
    """Budget enforcement and spend recording."""

    def test_can_spend_within_provider_cap(self) -> None:
        bt = BudgetTracker(DailyBudget())
        assert bt.can_spend("openai", 1.00) is True

    def test_can_spend_exceeds_provider_cap(self) -> None:
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 3.90)
        assert bt.can_spend("openai", 0.20) is False

    def test_can_spend_exceeds_total_cap(self) -> None:
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 4.00)
        bt.record_spend("anthropic", 4.50)
        bt.record_spend("google", 3.00)
        # Total spent: 11.50, total cap: 14.00, asking for 3.00 → over
        assert bt.can_spend("xai", 3.00) is False

    def test_can_spend_unknown_provider(self) -> None:
        bt = BudgetTracker(DailyBudget())
        # Unknown provider has 0.0 cap.
        assert bt.can_spend("unknown_provider", 0.01) is False

    def test_record_spend_accumulates(self) -> None:
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 0.50)
        bt.record_spend("openai", 0.30)
        assert bt.spent["openai"] == 0.80

    def test_record_spend_multiple_providers(self) -> None:
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 0.50)
        bt.record_spend("anthropic", 0.30)
        assert bt.spent["openai"] == 0.50
        assert bt.spent["anthropic"] == 0.30

    def test_reset_clears_all(self) -> None:
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 1.00)
        bt.record_spend("anthropic", 0.50)
        bt.reset_daily()
        assert bt.spent == {}
        assert bt.can_spend("openai", 1.00) is True

    def test_can_spend_at_exact_cap(self) -> None:
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 4.00)
        # At cap exactly, trying to spend 0.01 more should fail.
        assert bt.can_spend("openai", 0.01) is False

    def test_can_spend_zero_cost(self) -> None:
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 4.00)
        assert bt.can_spend("openai", 0.0) is True

    def test_total_cap_with_all_providers(self) -> None:
        budget = DailyBudget(daily_eur_cap_total=2.00)
        bt = BudgetTracker(budget)
        bt.record_spend("openai", 1.00)
        bt.record_spend("anthropic", 0.90)
        # Total: 1.90, asking for 0.20 → exceeds 2.00
        assert bt.can_spend("google", 0.20) is False

    def test_gemini_alias_uses_google_cap(self) -> None:
        """Provider 'gemini' should resolve to daily_eur_cap_google."""
        bt = BudgetTracker(DailyBudget())
        # google cap is 3.00 EUR by default
        assert bt.can_spend("gemini", 2.50) is True
        assert bt.can_spend("gemini", 3.50) is False


# ---- select_model (tiered ladder) -------------------------------------------


class TestSelectModel:
    """BudgetTracker.select_model() tiered model ladder."""

    def test_select_model_tier_a(self) -> None:
        """Fresh tracker returns Tier A model."""
        bt = BudgetTracker(DailyBudget())
        assert bt.select_model("openai") == "gpt-5-mini"

    def test_select_model_tier_b(self) -> None:
        """After spending past Tier A cap, returns Tier B model."""
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 2.00)
        assert bt.select_model("openai") == "gpt-4.1-mini"

    def test_select_model_exhausted(self) -> None:
        """After spending past all tiers, returns None."""
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 4.00)
        assert bt.select_model("openai") is None

    def test_select_model_anthropic_tiers(self) -> None:
        """Anthropic: sonnet under 2.50, haiku at 2.50+, None at 4.50+."""
        bt = BudgetTracker(DailyBudget())
        assert bt.select_model("anthropic") == "claude-sonnet-4-20250514"

        bt.record_spend("anthropic", 2.50)
        assert bt.select_model("anthropic") == "claude-haiku-4-5-20251001"

        bt.record_spend("anthropic", 2.00)  # total 4.50
        assert bt.select_model("anthropic") is None

    def test_select_model_gemini_alias(self) -> None:
        """Provider 'gemini' resolves to google ladder."""
        bt = BudgetTracker(DailyBudget())
        assert bt.select_model("gemini") == "gemini-2.5-flash"

        bt.record_spend("gemini", 1.00)
        # Still returns flash (Tier B)
        assert bt.select_model("gemini") == "gemini-2.5-flash"

        bt.record_spend("gemini", 2.00)  # total 3.00
        assert bt.select_model("gemini") is None

    def test_select_model_unknown_provider(self) -> None:
        """Unknown provider returns None."""
        bt = BudgetTracker(DailyBudget())
        assert bt.select_model("unknown_provider") is None

"""Abstract base class for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base class for all LLM provider clients."""

    # Per-model pricing in EUR per 1K tokens: {model: {input, output}}
    PRICING: dict[str, dict[str, float]] = {}

    def __init__(self, api_key: str, provider: str) -> None:
        self.api_key = api_key
        self.provider = provider

    @abstractmethod
    async def call(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, dict]:
        """Call the LLM and return (raw_response_text, usage_metadata)."""

    def estimate_cost_for_model(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> float:
        """Rough cost estimate in EUR for a specific model."""
        pricing = self.PRICING.get(model)
        if not pricing:
            return 0.0
        input_tokens = (len(system_prompt) + len(user_prompt)) / 4
        # Assume max output tokens for the estimate.
        output_tokens = 1000
        cost = (input_tokens / 1000) * pricing["input"] + (
            output_tokens / 1000
        ) * pricing["output"]
        return cost

    def actual_cost(self, model: str, usage: dict) -> float:
        """Compute actual cost in EUR from API usage metadata."""
        pricing = self.PRICING.get(model)
        if not pricing:
            return 0.0
        input_tokens = usage.get("prompt_tokens", 0) or usage.get(
            "input_tokens", 0
        )
        output_tokens = usage.get("completion_tokens", 0) or usage.get(
            "output_tokens", 0
        )
        return (input_tokens / 1000) * pricing["input"] + (
            output_tokens / 1000
        ) * pricing["output"]

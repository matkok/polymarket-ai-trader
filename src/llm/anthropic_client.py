"""Anthropic LLM client for the trading panel."""

from __future__ import annotations

import anthropic

from src.llm.base import BaseLLMClient


class AnthropicLLMClient(BaseLLMClient):
    """Anthropic API client with prompt caching support."""

    PRICING: dict[str, dict[str, float]] = {
        "claude-haiku-4-5-20251001": {"input": 0.00080, "output": 0.0040},
        "claude-sonnet-4-5-20250929": {"input": 0.0030, "output": 0.0150},
        "claude-sonnet-4-20250514": {"input": 0.0030, "output": 0.0150},
        "claude-opus-4-20250514": {"input": 0.0150, "output": 0.0750},
    }

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, provider="anthropic")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def call(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, dict]:
        response = await self.client.messages.create(
            model=model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text if response.content else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage["cache_creation_input_tokens"] = (
                response.usage.cache_creation_input_tokens
            )
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage["cache_read_input_tokens"] = (
                response.usage.cache_read_input_tokens
            )
        return content, usage

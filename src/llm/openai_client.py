"""OpenAI LLM client for the trading panel."""

from __future__ import annotations

import openai

from src.llm.base import BaseLLMClient


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client (gpt-4.1-mini, gpt-4.1, o4-mini)."""

    PRICING: dict[str, dict[str, float]] = {
        "gpt-4.1-mini": {"input": 0.00040, "output": 0.0016},
        "gpt-4.1": {"input": 0.0020, "output": 0.0080},
        "o4-mini": {"input": 0.0011, "output": 0.0044},
        "gpt-5-mini": {"input": 0.00060, "output": 0.0024},
        "gpt-5.2": {"input": 0.0030, "output": 0.0120},
    }

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, provider="openai")
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def call(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, dict]:
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if model == "o4-mini":
            kwargs["reasoning_effort"] = "medium"

        response = await self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return content, usage

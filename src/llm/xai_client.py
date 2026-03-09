"""xAI (Grok) LLM client for the trading panel."""

from __future__ import annotations

import openai

from src.llm.base import BaseLLMClient


class XAILLMClient(BaseLLMClient):
    """xAI API client using the OpenAI-compatible endpoint."""

    PRICING: dict[str, dict[str, float]] = {
        "grok-3-fast": {"input": 0.0005, "output": 0.0020},
        "grok-3": {"input": 0.0030, "output": 0.0150},
    }

    def __init__(self, api_key: str, base_url: str = "https://api.x.ai/v1") -> None:
        super().__init__(api_key=api_key, provider="xai")
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def call(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, dict]:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return content, usage

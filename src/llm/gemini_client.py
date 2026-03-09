"""Google Gemini LLM client for the trading panel."""

from __future__ import annotations

import json
import re

import structlog
from google import genai
from google.genai import types

from src.llm.base import BaseLLMClient

logger = structlog.get_logger(__name__)


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini API client."""

    PRICING: dict[str, dict[str, float]] = {
        "gemini-2.5-flash": {"input": 0.00015, "output": 0.00060},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01000},
    }

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, provider="gemini")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Attempt to repair JSON truncated mid-generation.

        Closes any open string literals, then balances braces/brackets.
        """
        # Strip trailing whitespace.
        text = text.rstrip()

        # If it already parses, return as-is.
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Close an unterminated string literal.
        # Count unescaped quotes — if odd, the last string is open.
        quote_count = len(re.findall(r'(?<!\\)"', text))
        if quote_count % 2 == 1:
            text += '"'

        # Balance braces and brackets.
        opens = 0
        open_brackets = 0
        for ch in text:
            if ch == "{":
                opens += 1
            elif ch == "}":
                opens -= 1
            elif ch == "[":
                open_brackets += 1
            elif ch == "]":
                open_brackets -= 1

        text += "]" * max(open_brackets, 0)
        text += "}" * max(opens, 0)

        return text

    async def call(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, dict]:
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                max_output_tokens=8192,
            ),
        )
        content = response.text or ""
        usage: dict = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            }

        # Detect and repair truncated responses.
        try:
            json.loads(content)
        except json.JSONDecodeError:
            if content:
                logger.warning(
                    "gemini_truncated_response",
                    model=model,
                    content_len=len(content),
                )
                content = self._repair_truncated_json(content)

        return content, usage

"""One-off connectivity test for all external APIs and LLM provider keys.

Run: conda run -n agent-trader python tests/test_connectivity.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


async def check_gamma_api(base_url: str) -> bool:
    """Polymarket Gamma API — fetch 1 market."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{base_url}/markets", params={"limit": 1, "active": True})
        r.raise_for_status()
        data = r.json()
        print(f"  {PASS}  Gamma API — fetched {len(data)} market(s)")
        return True


async def check_clob_api(base_url: str) -> bool:
    """Polymarket CLOB API — hit the time endpoint."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{base_url}/time")
        r.raise_for_status()
        print(f"  {PASS}  CLOB API — server time: {r.text.strip()[:40]}")
        return True


async def check_openai(api_key: str) -> bool:
    """OpenAI — list models to verify key."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        r.raise_for_status()
        models = r.json().get("data", [])
        print(f"  {PASS}  OpenAI — key valid, {len(models)} models available")
        return True


async def check_anthropic(api_key: str) -> bool:
    """Anthropic — send a minimal message to verify key."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Say OK"}],
            },
        )
        r.raise_for_status()
        text = r.json()["content"][0]["text"]
        print(f"  {PASS}  Anthropic — key valid, response: {text[:30]}")
        return True


async def check_google(api_key: str) -> bool:
    """Google Gemini — list models to verify key."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
        )
        r.raise_for_status()
        models = r.json().get("models", [])
        print(f"  {PASS}  Google Gemini — key valid, {len(models)} models available")
        return True


async def check_xai(api_key: str) -> bool:
    """xAI Grok — list models to verify key."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.x.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        r.raise_for_status()
        models = r.json().get("data", [])
        print(f"  {PASS}  xAI Grok — key valid, {len(models)} models available")
        return True


async def main() -> int:
    load_dotenv()

    gamma_url = os.getenv("GAMMA_API_BASE_URL", "https://gamma-api.polymarket.com")
    clob_url = os.getenv("CLOB_API_BASE_URL", "https://clob.polymarket.com")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")
    xai_key = os.getenv("XAI_API_KEY", "")

    checks: list[tuple[str, ...]] = []
    failures = 0

    print("\n--- Polymarket APIs (no key needed) ---")

    for name, coro in [
        ("Gamma API", check_gamma_api(gamma_url)),
        ("CLOB API", check_clob_api(clob_url)),
    ]:
        try:
            await coro
        except Exception as e:
            print(f"  {FAIL}  {name} — {e}")
            failures += 1

    print("\n--- LLM Provider Keys ---")

    for name, key, check_fn in [
        ("OpenAI", openai_key, check_openai),
        ("Anthropic", anthropic_key, check_anthropic),
        ("Google Gemini", google_key, check_google),
        ("xAI Grok", xai_key, check_xai),
    ]:
        if not key:
            print(f"  {FAIL}  {name} — key not set in .env")
            failures += 1
            continue
        try:
            await check_fn(key)
        except httpx.HTTPStatusError as e:
            print(f"  {FAIL}  {name} — HTTP {e.response.status_code}: {e.response.text[:100]}")
            failures += 1
        except Exception as e:
            print(f"  {FAIL}  {name} — {e}")
            failures += 1

    print()
    if failures == 0:
        print(f"All checks passed. Ready to run.")
    else:
        print(f"{failures} check(s) failed.")
    print()
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

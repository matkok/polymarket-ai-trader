"""Application settings loaded from environment variables and .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the structured trader system.

    Values are loaded from environment variables first, then from a .env file
    at the project root.  Env vars take precedence over file values.
    """

    # Database
    database_url: str = (
        "postgresql+asyncpg://agent_trader_structured:devpassword@localhost:5433/agent_trader_structured"
    )

    # Polymarket
    gamma_api_base_url: str = "https://gamma-api.polymarket.com"
    clob_api_base_url: str = "https://clob.polymarket.com"

    # Source API keys
    nws_user_agent: str = ""
    bls_api_key: str = ""
    fred_api_key: str = ""

    # Logging
    log_level: str = "INFO"

    # Paths
    policy_path: str = "policy.yaml"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` instance."""
    return Settings()

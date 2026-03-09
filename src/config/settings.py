"""Application settings loaded from environment variables and .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the agent-trader system.

    Values are loaded from environment variables first, then from a .env file
    at the project root.  Env vars take precedence over file values.
    """

    # Database
    database_url: str = (
        "postgresql+asyncpg://agent_trader:devpassword@localhost:5432/agent_trader"
    )

    # Polymarket
    gamma_api_base_url: str = "https://gamma-api.polymarket.com"
    clob_api_base_url: str = "https://clob.polymarket.com"

    # API Keys (M3)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    xai_api_key: str = ""

    # xAI
    xai_base_url: str = "https://api.x.ai/v1"

    # Logging
    log_level: str = "INFO"

    # Evidence ingestion
    rss_fetch_timeout: int = 30
    evidence_max_age_hours: int = 72

    # Google Trends
    google_trends_enabled: bool = False
    google_trends_trailing_days: int = 30

    # Wikipedia
    wikipedia_enabled: bool = False
    wikipedia_trailing_days: int = 30

    # Paths
    policy_path: str = "policy.yaml"
    charter_path: str = "charter.md"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` instance."""
    return Settings()

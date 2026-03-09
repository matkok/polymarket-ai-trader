"""Tests for src.config — settings, policy, and categories modules."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config.categories import Category, CategoryPortfolio
from src.config.policy import CategoryPolicy, Policy, load_policy, policy_version_hash
from src.config.settings import Settings, get_settings


# ---- Settings --------------------------------------------------------------


class TestSettings:
    """Settings loading from env vars and defaults."""

    def test_defaults(self) -> None:
        """Default values are applied when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.log_level == "INFO"
        assert s.gamma_api_base_url == "https://gamma-api.polymarket.com"
        assert s.clob_api_base_url == "https://clob.polymarket.com"
        assert s.policy_path == "policy.yaml"
        assert s.nws_user_agent == ""
        assert s.bls_api_key == ""
        assert s.fred_api_key == ""

    def test_env_override(self) -> None:
        """Environment variables override defaults."""
        overrides = {
            "LOG_LEVEL": "DEBUG",
            "NWS_USER_AGENT": "test-agent",
            "DATABASE_URL": "postgresql+asyncpg://u:p@host/db",
        }
        with patch.dict(os.environ, overrides, clear=True):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.log_level == "DEBUG"
        assert s.nws_user_agent == "test-agent"
        assert s.database_url == "postgresql+asyncpg://u:p@host/db"

    def test_get_settings_returns_same_instance(self) -> None:
        """get_settings() returns the same cached object on repeated calls."""
        get_settings.cache_clear()
        a = get_settings()
        b = get_settings()
        assert a is b
        get_settings.cache_clear()

    def test_no_llm_keys(self) -> None:
        """Settings should not have LLM-related keys."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert not hasattr(s, "openai_api_key")
        assert not hasattr(s, "anthropic_api_key")
        assert not hasattr(s, "google_api_key")
        assert not hasattr(s, "xai_api_key")


# ---- Policy -----------------------------------------------------------------


class TestPolicy:
    """Policy model validation and serialisation."""

    def test_defaults(self) -> None:
        """All default values match the spec."""
        p = Policy()
        assert p.bankroll_eur == 10_000.0
        assert p.cash_reserve_target_frac == 0.20
        assert p.max_total_exposure_frac == 0.60
        assert p.max_exposure_per_market_frac == 0.08
        assert p.max_open_positions == 15
        assert p.max_daily_loss_frac == 0.05
        assert p.min_liquidity_eur == 5_000.0
        assert p.edge_threshold == 0.05
        assert p.base_risk_frac == 0.02
        assert p.edge_scale == 0.20
        assert p.disagreement_block_threshold == 0.15
        assert p.disagreement_size_penalty_start == 0.08
        assert p.slippage_bps == 50
        assert p.fee_rate == 0.0
        assert p.min_volume == 10_000.0
        assert p.min_hours_to_resolution == 24
        assert p.max_hours_to_resolution == 2160
        assert p.max_daily_drawdown_frac == 0.08
        assert p.min_confidence_hard == 0.25
        assert p.min_confidence_full == 0.60
        assert p.max_entry_price == 0.90
        assert p.categories == {}

    def test_custom_values(self) -> None:
        """Policy accepts overridden values."""
        p = Policy(bankroll_eur=5_000.0, max_open_positions=8)
        assert p.bankroll_eur == 5_000.0
        assert p.max_open_positions == 8

    def test_load_policy_from_yaml(self, tmp_path: Path) -> None:
        """load_policy reads a YAML file and returns a validated Policy."""
        data = {"bankroll_eur": 20_000.0, "edge_threshold": 0.10}
        yaml_path = tmp_path / "policy.yaml"
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

        p = load_policy(str(yaml_path))
        assert p.bankroll_eur == 20_000.0
        assert p.edge_threshold == 0.10
        # Non-overridden fields keep defaults.
        assert p.max_open_positions == 15

    def test_load_policy_empty_yaml(self, tmp_path: Path) -> None:
        """An empty YAML file produces a Policy with all defaults."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("", encoding="utf-8")

        p = load_policy(str(yaml_path))
        assert p == Policy()

    def test_load_policy_missing_file(self) -> None:
        """load_policy raises FileNotFoundError for a missing path."""
        with pytest.raises(FileNotFoundError):
            load_policy("/nonexistent/path/policy.yaml")

    def test_load_policy_project_root(self) -> None:
        """The project-root policy.yaml loads without errors."""
        root_policy = Path(__file__).resolve().parent.parent / "policy.yaml"
        if root_policy.exists():
            p = load_policy(str(root_policy))
            assert p.bankroll_eur == 40_000.0
            assert len(p.categories) >= 2
            assert "weather" in p.categories
            assert "macro" in p.categories

    def test_policy_version_hash_deterministic(self) -> None:
        """The same policy always produces the same hash."""
        p = Policy()
        h1 = policy_version_hash(p)
        h2 = policy_version_hash(p)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_policy_version_hash_changes_with_values(self) -> None:
        """Different policy values produce different hashes."""
        p1 = Policy()
        p2 = Policy(bankroll_eur=5_000.0)
        assert policy_version_hash(p1) != policy_version_hash(p2)

    def test_policy_version_hash_matches_manual_sha256(self) -> None:
        """Hash matches a manually computed SHA-256 of sorted canonical JSON."""
        p = Policy()
        obj = json.loads(p.model_dump_json())
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert policy_version_hash(p) == expected


# ---- CategoryPolicy ---------------------------------------------------------


class TestCategoryPolicy:
    """CategoryPolicy model and overrides."""

    def test_defaults(self) -> None:
        """CategoryPolicy defaults to enabled with all None overrides."""
        cp = CategoryPolicy()
        assert cp.enabled is True
        assert cp.bankroll_eur is None
        assert cp.max_open_positions is None
        assert cp.edge_threshold is None
        assert cp.engine_params == {}

    def test_with_overrides(self) -> None:
        """CategoryPolicy accepts specific overrides."""
        cp = CategoryPolicy(
            bankroll_eur=3000.0,
            max_open_positions=5,
            engine_params={"forecast_horizon_hours": 168},
        )
        assert cp.bankroll_eur == 3000.0
        assert cp.max_open_positions == 5
        assert cp.engine_params["forecast_horizon_hours"] == 168


# ---- Policy.for_category ---------------------------------------------------


class TestPolicyForCategory:
    """Policy.for_category() returns effective policy with overrides."""

    def test_unknown_category_returns_copy(self) -> None:
        """Unknown category returns a copy of the global policy."""
        p = Policy(bankroll_eur=10_000.0)
        effective = p.for_category("unknown")
        assert effective.bankroll_eur == 10_000.0
        assert effective.categories == {}

    def test_category_override_bankroll(self) -> None:
        """Category override replaces bankroll_eur."""
        p = Policy(
            bankroll_eur=10_000.0,
            categories={"weather": CategoryPolicy(bankroll_eur=3000.0)},
        )
        effective = p.for_category("weather")
        assert effective.bankroll_eur == 3000.0
        # Non-overridden fields keep global values.
        assert effective.max_open_positions == 15
        assert effective.categories == {}

    def test_category_override_multiple_fields(self) -> None:
        """Multiple fields can be overridden at once."""
        p = Policy(
            bankroll_eur=10_000.0,
            max_open_positions=15,
            edge_threshold=0.05,
            categories={
                "macro": CategoryPolicy(
                    bankroll_eur=3000.0,
                    max_open_positions=5,
                    edge_threshold=0.08,
                ),
            },
        )
        effective = p.for_category("macro")
        assert effective.bankroll_eur == 3000.0
        assert effective.max_open_positions == 5
        assert effective.edge_threshold == 0.08
        # Non-overridden fields keep global values.
        assert effective.max_daily_loss_frac == 0.05

    def test_none_overrides_are_skipped(self) -> None:
        """None values in CategoryPolicy are not applied."""
        p = Policy(
            bankroll_eur=10_000.0,
            categories={"weather": CategoryPolicy(bankroll_eur=3000.0, max_open_positions=None)},
        )
        effective = p.for_category("weather")
        assert effective.bankroll_eur == 3000.0
        assert effective.max_open_positions == 15  # Global default

    def test_disabled_category_still_returns_policy(self) -> None:
        """Disabled category still returns a policy (callers check enabled)."""
        p = Policy(
            categories={"weather": CategoryPolicy(enabled=False, bankroll_eur=1000.0)},
        )
        effective = p.for_category("weather")
        assert effective.bankroll_eur == 1000.0

    def test_load_policy_with_categories(self, tmp_path: Path) -> None:
        """load_policy correctly parses category overrides from YAML."""
        data = {
            "bankroll_eur": 10_000.0,
            "categories": {
                "weather": {
                    "enabled": True,
                    "bankroll_eur": 3000.0,
                    "edge_threshold": 0.06,
                    "engine_params": {"forecast_horizon_hours": 168},
                },
            },
        }
        yaml_path = tmp_path / "policy.yaml"
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

        p = load_policy(str(yaml_path))
        assert "weather" in p.categories
        assert p.categories["weather"].bankroll_eur == 3000.0
        assert p.categories["weather"].engine_params["forecast_horizon_hours"] == 168

        effective = p.for_category("weather")
        assert effective.bankroll_eur == 3000.0
        assert effective.edge_threshold == 0.06


# ---- Categories -------------------------------------------------------------


class TestCategory:
    """Category enum values."""

    def test_values(self) -> None:
        assert Category.WEATHER == "weather"
        assert Category.MACRO == "macro"

    def test_str_enum(self) -> None:
        """Category is a string enum."""
        assert str(Category.WEATHER) == "Category.WEATHER"
        assert Category.WEATHER.value == "weather"


class TestCategoryPortfolio:
    """CategoryPortfolio dataclass."""

    def test_defaults(self) -> None:
        cp = CategoryPortfolio(category=Category.WEATHER, bankroll_eur=3000.0)
        assert cp.category == Category.WEATHER
        assert cp.bankroll_eur == 3000.0
        assert cp.exposure_eur == 0.0
        assert cp.realized_pnl_eur == 0.0
        assert cp.unrealized_pnl_eur == 0.0
        assert cp.daily_realized_pnl_eur == 0.0

    def test_custom_values(self) -> None:
        cp = CategoryPortfolio(
            category=Category.MACRO,
            bankroll_eur=5000.0,
            exposure_eur=1000.0,
            realized_pnl_eur=50.0,
        )
        assert cp.category == Category.MACRO
        assert cp.bankroll_eur == 5000.0
        assert cp.exposure_eur == 1000.0
        assert cp.realized_pnl_eur == 50.0

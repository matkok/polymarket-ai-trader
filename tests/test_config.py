"""Tests for src.config — settings, policy, and budget modules."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config.budget import BudgetTracker, DailyBudget
from src.config.policy import Policy, load_policy, policy_version_hash
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
        assert s.charter_path == "charter.md"
        assert s.openai_api_key == ""

    def test_env_override(self) -> None:
        """Environment variables override defaults."""
        overrides = {
            "LOG_LEVEL": "DEBUG",
            "OPENAI_API_KEY": "sk-test-123",
            "DATABASE_URL": "postgresql+asyncpg://u:p@host/db",
        }
        with patch.dict(os.environ, overrides, clear=True):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.log_level == "DEBUG"
        assert s.openai_api_key == "sk-test-123"
        assert s.database_url == "postgresql+asyncpg://u:p@host/db"

    def test_get_settings_returns_same_instance(self) -> None:
        """get_settings() returns the same cached object on repeated calls."""
        get_settings.cache_clear()
        a = get_settings()
        b = get_settings()
        assert a is b
        get_settings.cache_clear()


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
        assert p.ambiguity_veto_threshold == 0.85
        assert p.veto_quorum == 2
        assert p.min_hold_minutes == 60
        assert p.veto_exit_consecutive_required == 2
        assert p.no_add_if_recent_veto_minutes == 120
        assert p.odds_move_recheck_threshold == 0.05
        assert p.new_evidence_recheck_window_minutes == 120
        assert p.panel_cooldown_hours == 8
        assert p.max_candidates_per_cycle == 15
        assert p.max_panel_markets_per_day == 20
        assert p.slippage_bps == 50
        assert p.fee_rate == 0.0
        assert p.crypto_15min_fee_rate == 0.02
        assert p.min_volume == 10_000.0
        assert p.min_hours_to_resolution == 24
        assert p.max_hours_to_resolution == 2160
        assert p.crypto_15min_fee_rate == 0.02
        assert p.max_daily_drawdown_frac == 0.08
        assert p.evidence_similarity_threshold == 0.35
        assert p.min_confidence_hard == 0.25
        assert p.min_confidence_full == 0.60
        assert p.max_entry_price == 0.90
        assert p.position_reentry_cooldown_hours == 6

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
            assert p.edge_threshold == 0.06
            assert p.max_panel_markets_per_day == 60
            assert p.triage_panel_threshold == 0.40

    def test_load_calibration_policy(self) -> None:
        """policy.calibration.yaml loads and has conservative params."""
        cal_path = Path(__file__).resolve().parent.parent / "policy.calibration.yaml"
        if cal_path.exists():
            p = load_policy(str(cal_path))
            assert p.base_risk_frac == 0.005
            assert p.max_open_positions == 10
            assert p.edge_threshold == 0.07
            assert p.max_daily_drawdown_frac == 0.05
            assert p.fee_rate == 0.0

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


# ---- Budget -----------------------------------------------------------------


class TestDailyBudget:
    """DailyBudget model defaults."""

    def test_defaults(self) -> None:
        b = DailyBudget()
        assert b.daily_eur_cap_openai == 4.00
        assert b.daily_eur_cap_anthropic == 4.50
        assert b.daily_eur_cap_google == 3.00
        assert b.daily_eur_cap_xai == 2.50
        assert b.daily_eur_cap_total == 14.00


class TestBudgetTracker:
    """BudgetTracker enforcement behaviour."""

    def test_can_spend_within_cap(self) -> None:
        """Returns True when spend is within provider cap."""
        bt = BudgetTracker(DailyBudget())
        assert bt.can_spend("openai", 1.00) is True

    def test_can_spend_exceeds_cap(self) -> None:
        """Returns False when spend exceeds provider cap."""
        bt = BudgetTracker(DailyBudget())
        assert bt.can_spend("openai", 100.0) is False

    def test_record_spend_accumulates(self) -> None:
        """record_spend accumulates cost."""
        bt = BudgetTracker(DailyBudget())
        bt.record_spend("openai", 0.50)
        assert bt.spent["openai"] == 0.50

    def test_reset_daily_clears_spent(self) -> None:
        """reset_daily empties the spent dict."""
        bt = BudgetTracker(DailyBudget())
        bt.spent["openai"] = 1.0
        bt.reset_daily()
        assert bt.spent == {}

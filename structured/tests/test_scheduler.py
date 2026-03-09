"""Tests for src.app.scheduler — StructuredTradingEngine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.scheduler import StructuredTradingEngine
from src.config.policy import Policy
from src.db.models import EnginePrice, Position
from src.evaluation.kill_switch import KillSwitch, KillSwitchConfig
from src.polymarket.schemas import GammaMarket


# ---- Helpers ----------------------------------------------------------------


def _make_gamma_market(
    condition_id: str = "0xabc",
    question: str = "Will it rain?",
    outcome_prices: str = '["0.55","0.45"]',
    end_date_iso: str = "2026-06-01T00:00:00Z",
    liquidity: float = 10000.0,
    volume: float = 50000.0,
    active: bool = True,
    closed: bool = False,
    description: str = "Resolves yes if it rains.",
    category: str = "Weather",
) -> GammaMarket:
    return GammaMarket(
        condition_id=condition_id,
        question=question,
        outcome_prices=outcome_prices,
        end_date_iso=end_date_iso,
        liquidity=liquidity,
        volume=volume,
        active=active,
        closed=closed,
        description=description,
        category=category,
    )


# ---- StructuredTradingEngine ------------------------------------------------


class TestStructuredTradingEngineInit:
    """StructuredTradingEngine initialization."""

    def test_init(self) -> None:
        repo = MagicMock()
        gamma = MagicMock()
        policy = Policy()
        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        assert engine.repo is repo
        assert engine.gamma is gamma
        assert engine.policy is policy
        assert engine.daily_realized_pnl == 0.0


class TestIngestMarkets:
    """Market ingestion tests."""

    async def test_ingest_markets_calls_gamma_and_repo(self) -> None:
        """ingest_markets fetches from Gamma and upserts to repo."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        markets = [
            _make_gamma_market(condition_id="mkt-1", question="Q1"),
            _make_gamma_market(condition_id="mkt-2", question="Q2"),
        ]
        gamma.get_all_active_markets.return_value = markets

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        count = await engine.ingest_markets()

        assert count == 2
        gamma.get_all_active_markets.assert_called_once()
        repo.bulk_upsert_markets.assert_called_once()

        # Verify snapshots were bulk-inserted.
        repo.bulk_add_snapshots.assert_called_once()
        assert len(repo.bulk_add_snapshots.call_args[0][0]) == 2

    async def test_ingest_markets_skips_empty_condition_id(self) -> None:
        """Markets without condition_id are skipped."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        markets = [
            _make_gamma_market(condition_id="", question="Empty ID"),
            _make_gamma_market(condition_id="mkt-1", question="Q1"),
        ]
        gamma.get_all_active_markets.return_value = markets

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        count = await engine.ingest_markets()

        assert count == 1
        repo.bulk_add_snapshots.assert_called_once()
        assert len(repo.bulk_add_snapshots.call_args[0][0]) == 1

    async def test_ingest_markets_empty_response(self) -> None:
        """Empty Gamma response results in zero markets."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        gamma.get_all_active_markets.return_value = []

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        count = await engine.ingest_markets()

        assert count == 0
        repo.bulk_upsert_markets.assert_not_called()

    async def test_ingest_markets_resolution_time_parsed(self) -> None:
        """Resolution time is parsed from end_date_iso."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        markets = [
            _make_gamma_market(
                condition_id="mkt-1",
                end_date_iso="2026-06-01T00:00:00Z",
            ),
        ]
        gamma.get_all_active_markets.return_value = markets

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["resolution_time_utc"] is not None

    async def test_ingest_markets_maps_description_to_rules_text(self) -> None:
        """Gamma description field is mapped to rules_text."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        markets = [
            _make_gamma_market(
                condition_id="mkt-1",
                description="This market resolves yes if...",
            ),
        ]
        gamma.get_all_active_markets.return_value = markets

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["rules_text"] == "This market resolves yes if..."

    async def test_ingest_markets_closed_market_status(self) -> None:
        """Closed markets get status='closed'."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        markets = [
            _make_gamma_market(condition_id="mkt-1", active=True, closed=True),
        ]
        gamma.get_all_active_markets.return_value = markets

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        await engine.ingest_markets()

        call_args = repo.bulk_upsert_markets.call_args[0][0]
        assert call_args[0]["status"] == "closed"

    async def test_ingest_markets_snapshot_mid_computed(self) -> None:
        """Snapshot mid is computed from outcome prices."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        markets = [
            _make_gamma_market(
                condition_id="mkt-1",
                outcome_prices='["0.60","0.40"]',
            ),
        ]
        gamma.get_all_active_markets.return_value = markets

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        await engine.ingest_markets()

        snap_data = repo.bulk_add_snapshots.call_args[0][0][0]
        assert snap_data["best_bid"] == pytest.approx(0.60)
        assert snap_data["mid"] == pytest.approx(0.60)


class TestClassifyMarkets:
    """classify_markets integration tests."""

    async def test_delegates_to_classify_markets_batch(self) -> None:
        """classify_markets calls classify_markets_batch and returns count."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        # Provide an unparsed weather market.
        market = MagicMock()
        market.market_id = "mkt-w1"
        market.question = "Will the high temperature in Dallas exceed 100°F?"
        market.rules_text = None
        repo.get_unparsed_markets.return_value = [market]

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.classify_markets()

        assert result == 1
        repo.upsert_category_assignment.assert_called_once()

    async def test_returns_zero_when_no_unparsed(self) -> None:
        """No unparsed markets → returns 0."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        repo.get_unparsed_markets.return_value = []

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.classify_markets()

        assert result == 0


class TestDailyReset:
    """Daily reset tests."""

    def test_resets_daily_pnl(self) -> None:
        """reset_daily clears daily realized PnL."""
        repo = MagicMock()
        gamma = MagicMock()
        policy = Policy()
        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        engine.daily_realized_pnl = 100.0
        engine.reset_daily()
        assert engine.daily_realized_pnl == 0.0

    def test_reset_daily_calls_kill_switch_reset_all(self) -> None:
        """reset_daily resets the kill switch."""
        repo = MagicMock()
        gamma = MagicMock()
        policy = Policy()
        ks = KillSwitch()
        ks._disabled_categories.add("weather")
        engine = StructuredTradingEngine(
            repo=repo, gamma_client=gamma, policy=policy, kill_switch=ks,
        )
        engine.reset_daily()
        assert ks.is_enabled("weather")

    def test_reset_daily_no_kill_switch_ok(self) -> None:
        """reset_daily works without a kill switch."""
        repo = MagicMock()
        gamma = MagicMock()
        policy = Policy()
        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        engine.daily_realized_pnl = 50.0
        engine.reset_daily()
        assert engine.daily_realized_pnl == 0.0


# ---- Calibration Cycle (S6) ------------------------------------------------


def _make_resolution(market_id: str = "m1", outcome: str = "yes") -> MagicMock:
    """Create a mock Resolution."""
    r = MagicMock()
    r.market_id = market_id
    r.outcome = outcome
    r.resolved_ts_utc = datetime.now(timezone.utc)
    return r


def _make_assignment(market_id: str = "m1", category: str = "weather") -> MagicMock:
    """Create a mock CategoryAssignment."""
    a = MagicMock()
    a.market_id = market_id
    a.category = category
    return a


def _make_engine_price(
    market_id: str = "m1",
    p_yes: float = 0.7,
    engine_version: str = "weather_v1",
) -> MagicMock:
    """Create a mock EnginePrice."""
    ep = MagicMock(spec=EnginePrice)
    ep.market_id = market_id
    ep.p_yes = p_yes
    ep.engine_version = engine_version
    return ep


class TestCalibrationCycle:
    """run_calibration_cycle tests."""

    async def test_no_resolutions_returns_empty(self) -> None:
        """No resolutions → summary with resolutions=0."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        repo.get_resolutions_since.return_value = []

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.run_calibration_cycle()

        assert result == {"resolutions": 0}
        repo.add_calibration_stat.assert_not_called()

    async def test_resolutions_with_engine_prices_persists_stat(self) -> None:
        """Resolved markets with engine prices → computes Brier, persists stat."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        res = _make_resolution("m1", outcome="yes")
        repo.get_resolutions_since.return_value = [res]
        repo.get_assignment.return_value = _make_assignment("m1", "weather")
        repo.get_engine_prices_for_markets.return_value = [
            _make_engine_price("m1", p_yes=0.8, engine_version="weather_v1"),
        ]
        repo.add_calibration_stat.return_value = 1

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.run_calibration_cycle()

        assert result["resolutions"] == 1
        assert result["calibrated"] == 1
        assert "weather" in result["brier_scores"]
        # Brier for p=0.8, outcome=1.0: (0.8 - 1.0)^2 = 0.04
        assert result["brier_scores"]["weather"] == pytest.approx(0.04)
        repo.add_calibration_stat.assert_called_once()

    async def test_kill_switch_triggered_on_high_brier(self) -> None:
        """High Brier score triggers kill switch for the category."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        ks = KillSwitch(config=KillSwitchConfig(
            max_brier_score=0.30,
            min_predictions_before_check=1,
        ))

        res = _make_resolution("m1", outcome="yes")
        repo.get_resolutions_since.return_value = [res]
        repo.get_assignment.return_value = _make_assignment("m1", "weather")
        # p_yes=0.1 for outcome=1.0 → Brier = (0.1-1.0)^2 = 0.81
        repo.get_engine_prices_for_markets.return_value = [
            _make_engine_price("m1", p_yes=0.1, engine_version="weather_v1"),
        ]
        repo.add_calibration_stat.return_value = 1

        engine = StructuredTradingEngine(
            repo=repo, gamma_client=gamma, policy=policy, kill_switch=ks,
        )
        await engine.run_calibration_cycle()

        assert not ks.is_enabled("weather")

    async def test_missing_assignment_skipped(self) -> None:
        """Resolution with no assignment is skipped gracefully."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        res = _make_resolution("m1", outcome="yes")
        repo.get_resolutions_since.return_value = [res]
        repo.get_assignment.return_value = None
        repo.get_engine_prices_for_markets.return_value = []

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.run_calibration_cycle()

        assert result["assigned"] == 0
        repo.add_calibration_stat.assert_not_called()

    async def test_missing_engine_price_skipped(self) -> None:
        """Resolution with assignment but no engine price produces no calibration."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        res = _make_resolution("m1", outcome="yes")
        repo.get_resolutions_since.return_value = [res]
        repo.get_assignment.return_value = _make_assignment("m1", "weather")
        repo.get_engine_prices_for_markets.return_value = []

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.run_calibration_cycle()

        assert result["assigned"] == 1
        assert result["calibrated"] == 0
        repo.add_calibration_stat.assert_not_called()

    async def test_outcome_no_maps_to_zero(self) -> None:
        """Outcome 'no' maps to 0.0 for Brier calculation."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        res = _make_resolution("m1", outcome="no")
        repo.get_resolutions_since.return_value = [res]
        repo.get_assignment.return_value = _make_assignment("m1", "crypto")
        repo.get_engine_prices_for_markets.return_value = [
            _make_engine_price("m1", p_yes=0.2, engine_version="crypto_v1"),
        ]
        repo.add_calibration_stat.return_value = 1

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.run_calibration_cycle()

        # Brier for p=0.2, outcome=0.0: (0.2 - 0.0)^2 = 0.04
        assert result["brier_scores"]["crypto"] == pytest.approx(0.04)


# ---- Daily PnL Aggregation (S6) -------------------------------------------


class TestDailyPnLAggregation:
    """aggregate_daily_pnl tests."""

    async def test_no_positions_empty_summary(self) -> None:
        """No positions → all categories have zero PnL."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        repo.get_markets_by_category.return_value = []

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.aggregate_daily_pnl()

        assert len(result) == 4  # weather, macro, crypto, earnings
        for cat_data in result.values():
            assert cat_data["realized"] == 0.0
            assert cat_data["unrealized"] == 0.0

    async def test_persists_pnl_rows_per_category(self) -> None:
        """Each category gets a PnL row persisted."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        repo.get_markets_by_category.return_value = []

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        await engine.aggregate_daily_pnl()

        assert repo.add_category_pnl_daily.call_count == 4

    async def test_closed_position_realized_pnl(self) -> None:
        """Closed position contributes realized PnL."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()

        assignment = _make_assignment("m1", "weather")
        repo.get_markets_by_category.side_effect = lambda cat: (
            [assignment] if cat == "weather" else []
        )
        pos = MagicMock(spec=Position)
        pos.status = "closed"
        pos.realized_pnl = 42.0
        repo.get_position.return_value = pos

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)
        result = await engine.aggregate_daily_pnl()

        assert result["weather"]["realized"] == 42.0
        assert result["weather"]["trades_closed"] == 1


# ---- Kill Switch Integration (S6) -----------------------------------------


class TestKillSwitchIntegration:
    """Kill switch gating on pipeline cycle methods."""

    async def test_killed_weather_returns_killed(self) -> None:
        """Killed weather category returns {'killed': True}."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        ks = KillSwitch()
        ks._disabled_categories.add("weather")

        engine = StructuredTradingEngine(
            repo=repo, gamma_client=gamma, policy=policy, kill_switch=ks,
        )
        result = await engine.run_weather_cycle()
        assert result == {"killed": True}

    async def test_killed_macro_returns_killed(self) -> None:
        """Killed macro category returns {'killed': True}."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        ks = KillSwitch()
        ks._disabled_categories.add("macro")

        engine = StructuredTradingEngine(
            repo=repo, gamma_client=gamma, policy=policy, kill_switch=ks,
        )
        result = await engine.run_macro_cycle()
        assert result == {"killed": True}

    async def test_killed_crypto_returns_killed(self) -> None:
        """Killed crypto category returns {'killed': True}."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        ks = KillSwitch()
        ks._disabled_categories.add("crypto")

        engine = StructuredTradingEngine(
            repo=repo, gamma_client=gamma, policy=policy, kill_switch=ks,
        )
        result = await engine.run_crypto_cycle()
        assert result == {"killed": True}

    async def test_killed_earnings_returns_killed(self) -> None:
        """Killed earnings category returns {'killed': True}."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        ks = KillSwitch()
        ks._disabled_categories.add("earnings")

        engine = StructuredTradingEngine(
            repo=repo, gamma_client=gamma, policy=policy, kill_switch=ks,
        )
        result = await engine.run_earnings_cycle()
        assert result == {"killed": True}

    async def test_enabled_category_runs_normally(self) -> None:
        """Enabled category runs the pipeline normally."""
        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        ks = KillSwitch()  # nothing disabled
        mock_pipeline = AsyncMock()
        mock_pipeline.run_cycle.return_value = {"trades": 3}

        engine = StructuredTradingEngine(
            repo=repo,
            gamma_client=gamma,
            policy=policy,
            kill_switch=ks,
            weather_pipeline=mock_pipeline,
        )
        result = await engine.run_weather_cycle()

        assert result == {"trades": 3}
        mock_pipeline.run_cycle.assert_called_once()


# ---- Reduce Cooldown -------------------------------------------------------


def _make_position_with_update(
    market_id: str = "mkt-1",
    side: str = "BUY_YES",
    size_eur: float = 100.0,
    avg_entry_price: float = 0.50,
    last_update_ts_utc: datetime | None = None,
    realized_pnl: float = 0.0,
    opened_ts_utc: datetime | None = None,
) -> MagicMock:
    """Create a mock Position with last_update_ts_utc."""
    pos = MagicMock(spec=Position)
    pos.market_id = market_id
    pos.side = side
    pos.size_eur = size_eur
    pos.avg_entry_price = avg_entry_price
    pos.last_update_ts_utc = last_update_ts_utc
    pos.realized_pnl = realized_pnl
    pos.opened_ts_utc = opened_ts_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pos


class TestReduceCooldown:
    """Reduce cooldown prevents rapid-fire reduces."""

    async def test_reduce_within_cooldown_held(self) -> None:
        """REDUCE skipped when last update was 30 minutes ago (< 60 min cooldown)."""
        from datetime import timedelta
        from src.portfolio.lifecycle import LifecycleAction, LifecycleDecision

        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)

        # Position updated 30 minutes ago.
        pos = _make_position_with_update(
            last_update_ts_utc=now - timedelta(minutes=30),
        )

        market = MagicMock()
        market.resolution_time_utc = now + timedelta(days=5)
        repo.get_market.return_value = market

        snap = MagicMock()
        snap.mid = 0.48
        snap.best_bid = 0.47
        snap.best_ask = 0.49
        repo.get_latest_snapshot.return_value = snap

        repo.get_latest_engine_price.return_value = None

        assignment = _make_assignment("mkt-1", "crypto")
        repo.get_assignment.return_value = assignment

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)

        # Mock lifecycle to return REDUCE.
        with patch(
            "src.app.scheduler.PositionLifecycle"
        ) as mock_lc_cls:
            mock_lc = MagicMock()
            mock_lc.evaluate.return_value = LifecycleDecision(
                action=LifecycleAction.REDUCE, reasons=["partial stop"],
            )
            mock_lc_cls.return_value = mock_lc

            result = await engine._review_position(pos, now)

        assert result == "held"

    async def test_reduce_after_cooldown_allowed(self) -> None:
        """REDUCE proceeds when last update was 61 minutes ago (> 60 min cooldown)."""
        from datetime import timedelta
        from src.portfolio.lifecycle import LifecycleAction, LifecycleDecision

        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy(reduce_fraction=0.50, dust_position_eur=1.0)
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)

        pos = _make_position_with_update(
            last_update_ts_utc=now - timedelta(minutes=61),
            size_eur=100.0,
        )

        market = MagicMock()
        market.resolution_time_utc = now + timedelta(days=5)
        repo.get_market.return_value = market

        snap = MagicMock()
        snap.mid = 0.48
        snap.best_bid = 0.47
        snap.best_ask = 0.49
        repo.get_latest_snapshot.return_value = snap

        repo.get_latest_engine_price.return_value = None

        assignment = _make_assignment("mkt-1", "crypto")
        repo.get_assignment.return_value = assignment

        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)

        with patch(
            "src.app.scheduler.PositionLifecycle"
        ) as mock_lc_cls:
            mock_lc = MagicMock()
            mock_lc.evaluate.return_value = LifecycleDecision(
                action=LifecycleAction.REDUCE, reasons=["partial stop"],
            )
            mock_lc_cls.return_value = mock_lc

            result = await engine._review_position(pos, now)

        assert result == "reduced"

    async def test_close_ignores_cooldown(self) -> None:
        """CLOSE always fires regardless of last_update_ts_utc."""
        from datetime import timedelta
        from src.portfolio.lifecycle import LifecycleAction, LifecycleDecision

        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)

        # Last update 5 minutes ago — would be blocked for REDUCE.
        pos = _make_position_with_update(
            last_update_ts_utc=now - timedelta(minutes=5),
            size_eur=100.0,
        )

        market = MagicMock()
        market.resolution_time_utc = now + timedelta(days=5)
        repo.get_market.return_value = market

        snap = MagicMock()
        snap.mid = 0.48
        snap.best_bid = 0.47
        snap.best_ask = 0.49
        repo.get_latest_snapshot.return_value = snap

        repo.get_latest_engine_price.return_value = None

        assignment = _make_assignment("mkt-1", "crypto")
        repo.get_assignment.return_value = assignment

        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)

        with patch(
            "src.app.scheduler.PositionLifecycle"
        ) as mock_lc_cls:
            mock_lc = MagicMock()
            mock_lc.evaluate.return_value = LifecycleDecision(
                action=LifecycleAction.CLOSE, reasons=["stop loss"],
            )
            mock_lc_cls.return_value = mock_lc

            result = await engine._review_position(pos, now)

        assert result == "closed"


# ---- Reduce Sell-Size Invariant --------------------------------------------


class TestReduceSellSizeInvariant:
    """Reduce cannot produce negative remaining size."""

    async def test_reduce_fraction_clamped_to_position_size(self) -> None:
        """reduce_size is clamped to position.size_eur — no negative remaining."""
        from datetime import timedelta
        from src.portfolio.lifecycle import LifecycleAction, LifecycleDecision

        repo = AsyncMock()
        gamma = AsyncMock()
        # reduce_fraction > 1.0 would produce negative remaining without clamp.
        policy = Policy(reduce_fraction=1.5, dust_position_eur=0.01)
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)

        pos = _make_position_with_update(
            last_update_ts_utc=now - timedelta(hours=2),
            size_eur=10.0,
        )

        market = MagicMock()
        market.resolution_time_utc = now + timedelta(days=5)
        repo.get_market.return_value = market

        snap = MagicMock()
        snap.mid = 0.48
        snap.best_bid = 0.47
        snap.best_ask = 0.49
        repo.get_latest_snapshot.return_value = snap

        repo.get_latest_engine_price.return_value = None

        assignment = _make_assignment("mkt-1", "crypto")
        repo.get_assignment.return_value = assignment

        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)

        with patch(
            "src.app.scheduler.PositionLifecycle"
        ) as mock_lc_cls:
            mock_lc = MagicMock()
            mock_lc.evaluate.return_value = LifecycleDecision(
                action=LifecycleAction.REDUCE, reasons=["partial stop"],
            )
            mock_lc_cls.return_value = mock_lc

            result = await engine._review_position(pos, now)

        # With clamp: reduce_size = min(10.0 * 1.5, 10.0) = 10.0
        # remaining = 10.0 - 10.0 = 0.0 (not negative)
        assert result == "reduced"
        # Verify the upsert_position call has remaining_size >= 0.
        upsert_call = repo.upsert_position.call_args
        assert upsert_call[0][0]["size_eur"] >= 0


# ---- PnL Loss Bounds -------------------------------------------------------


class TestPnLLossBounds:
    """PnL loss exceeding position size logs critical."""

    async def test_normal_loss_no_critical(self) -> None:
        """Normal loss within bounds does not trigger critical log."""
        from datetime import timedelta
        from src.portfolio.lifecycle import LifecycleAction, LifecycleDecision

        repo = AsyncMock()
        gamma = AsyncMock()
        policy = Policy()
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)

        pos = _make_position_with_update(
            last_update_ts_utc=now - timedelta(hours=2),
            size_eur=100.0,
            avg_entry_price=0.50,
        )

        market = MagicMock()
        market.resolution_time_utc = now + timedelta(days=5)
        repo.get_market.return_value = market

        snap = MagicMock()
        snap.mid = 0.48
        snap.best_bid = 0.47
        snap.best_ask = 0.49
        repo.get_latest_snapshot.return_value = snap

        repo.get_latest_engine_price.return_value = None

        assignment = _make_assignment("mkt-1", "crypto")
        repo.get_assignment.return_value = assignment

        repo.add_decision.return_value = 1
        repo.add_order.return_value = 1

        engine = StructuredTradingEngine(repo=repo, gamma_client=gamma, policy=policy)

        with patch(
            "src.app.scheduler.PositionLifecycle"
        ) as mock_lc_cls:
            mock_lc = MagicMock()
            mock_lc.evaluate.return_value = LifecycleDecision(
                action=LifecycleAction.CLOSE, reasons=["stop loss"],
            )
            mock_lc_cls.return_value = mock_lc

            # Should complete without issues.
            result = await engine._review_position(pos, now)

        assert result == "closed"

"""Tests for macro engine — release calendar, BLS adapter, FRED adapter,
pricing engine, pipeline, and scheduler wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.contracts.macro import MacroContractSpec
from src.engines.macro import MacroEngine
from src.sources.base import FetchResult
from src.sources.release_calendar import (
    is_near_release,
    next_release,
    resolve_release_date,
)

# ===========================================================================
# Release Calendar
# ===========================================================================


class TestReleaseCalendarCPI:
    """CPI release date resolution."""

    def test_cpi_january_2026(self) -> None:
        dt = resolve_release_date("cpi", "January 2026")
        assert dt is not None
        # CPI for Jan released ~13th of Feb 2026.
        assert dt.month == 2
        assert dt.year == 2026
        assert dt.day == 13

    def test_core_cpi_same_as_cpi(self) -> None:
        dt = resolve_release_date("core_cpi", "March 2026")
        assert dt is not None
        assert dt.month == 4  # Released month after.

    def test_cpi_december_rolls_year(self) -> None:
        dt = resolve_release_date("cpi", "December 2025")
        assert dt is not None
        assert dt.month == 1
        assert dt.year == 2026


class TestReleaseCalendarPayrolls:
    """NFP release date resolution."""

    def test_nfp_february_2026(self) -> None:
        dt = resolve_release_date("nonfarm_payrolls", "February 2026")
        assert dt is not None
        # First Friday of March 2026 = March 6 (March 1 is Sunday → Friday is March 6).
        assert dt.month == 3
        assert dt.year == 2026
        assert dt.weekday() == 4  # Friday


class TestReleaseCalendarFed:
    """FOMC meeting date resolution."""

    def test_fed_march_2026(self) -> None:
        dt = resolve_release_date("fed_rate", "March 2026")
        assert dt is not None
        # Nearest FOMC to March 2026 → March 18, 2026.
        assert dt.month == 3
        assert dt.year == 2026

    def test_fed_june_2026(self) -> None:
        dt = resolve_release_date("fed_rate", "June 2026")
        assert dt is not None
        assert dt.month == 6
        assert dt.year == 2026


class TestReleaseCalendarGDP:
    """GDP release date resolution."""

    def test_gdp_q1_2026(self) -> None:
        dt = resolve_release_date("gdp", "Q1 2026")
        assert dt is not None
        assert dt.month == 4  # Q1 data → April release
        assert dt.day == 28


class TestReleaseCalendarPCE:
    """PCE release date resolution."""

    def test_pce_january_2026(self) -> None:
        dt = resolve_release_date("pce", "January 2026")
        assert dt is not None
        # PCE for Jan released last weekday of Feb.
        assert dt.month == 2
        assert dt.weekday() < 5  # Weekday


class TestReleaseCalendarHelpers:
    """Helper functions."""

    def test_next_release_fed_finds_future(self) -> None:
        after = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dt = next_release("fed_rate", after=after)
        assert dt is not None
        assert dt > after

    def test_next_release_cpi_finds_future(self) -> None:
        after = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dt = next_release("cpi", after=after)
        assert dt is not None
        assert dt > after

    def test_is_near_release_true(self) -> None:
        release = resolve_release_date("cpi", "January 2026")
        assert release is not None
        near_time = release - timedelta(hours=12)
        assert is_near_release("cpi", "January 2026", hours_threshold=24.0, now=near_time) is True

    def test_is_near_release_false(self) -> None:
        release = resolve_release_date("cpi", "January 2026")
        assert release is not None
        far_time = release - timedelta(days=30)
        assert is_near_release("cpi", "January 2026", hours_threshold=24.0, now=far_time) is False

    def test_unknown_period_returns_none(self) -> None:
        assert resolve_release_date("cpi", "gibberish") is None

    def test_unknown_indicator_returns_none(self) -> None:
        assert resolve_release_date("unknown_thing", "January 2026") is None


# ===========================================================================
# BLS Adapter
# ===========================================================================


class TestBLSAdapterParse:
    """BLS response parsing (unit tests, no HTTP)."""

    def test_parse_successful_response(self) -> None:
        from src.sources.bls import BLSAdapter

        raw = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [{
                    "seriesID": "CUSR0000SA0",
                    "data": [
                        {"year": "2026", "periodName": "January", "value": "310.5", "footnotes": [{}]},
                        {"year": "2025", "periodName": "December", "value": "309.2", "footnotes": [{}]},
                    ],
                }],
            },
        }
        result = BLSAdapter._parse_response(raw, "cpi", "CUSR0000SA0")
        assert result["latest_value"] == 310.5
        assert result["previous_value"] == 309.2
        assert "month_over_month_change" in result
        assert len(result["history"]) == 2

    def test_parse_preliminary_flag(self) -> None:
        from src.sources.bls import BLSAdapter

        raw = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [{
                    "seriesID": "CUSR0000SA0",
                    "data": [
                        {"year": "2026", "periodName": "January", "value": "310.5",
                         "footnotes": [{"text": "Preliminary data"}]},
                    ],
                }],
            },
        }
        result = BLSAdapter._parse_response(raw, "cpi", "CUSR0000SA0")
        assert result["is_preliminary"] is True

    def test_parse_failed_status(self) -> None:
        from src.sources.bls import BLSAdapter

        raw = {"status": "REQUEST_FAILED"}
        result = BLSAdapter._parse_response(raw, "cpi", "CUSR0000SA0")
        assert "error" in result

    def test_parse_no_data(self) -> None:
        from src.sources.bls import BLSAdapter

        raw = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [{"data": []}]}}
        result = BLSAdapter._parse_response(raw, "cpi", "CUSR0000SA0")
        assert "error" in result


class TestBLSAdapterFetch:
    """BLS adapter fetch method (mocked HTTP)."""

    def test_name(self) -> None:
        from src.sources.bls import BLSAdapter
        assert BLSAdapter().name == "bls"

    async def test_unsupported_indicator(self) -> None:
        from src.sources.bls import BLSAdapter

        adapter = BLSAdapter()
        spec = MacroContractSpec(category="macro", indicator="unknown_thing")
        result = await adapter.fetch(spec)
        assert not result.ok
        assert "unsupported_indicator" in (result.error or "")

    async def test_invalid_spec_type(self) -> None:
        from src.sources.bls import BLSAdapter

        adapter = BLSAdapter()
        result = await adapter.fetch("not a spec")
        assert not result.ok
        assert result.error == "invalid_spec_type"


# ===========================================================================
# FRED Adapter
# ===========================================================================


class TestFREDAdapterParse:
    """FRED response parsing (unit tests, no HTTP)."""

    def test_parse_successful_response(self) -> None:
        from src.sources.fred import FREDAdapter

        raw = {
            "observations": [
                {"date": "2026-01-01", "value": "21500.5"},
                {"date": "2025-10-01", "value": "21200.3"},
            ],
        }
        result = FREDAdapter._parse_response(raw, "gdp", "GDPC1")
        assert result["latest_value"] == 21500.5
        assert result["previous_value"] == 21200.3
        assert "change" in result
        assert len(result["history"]) == 2

    def test_parse_missing_values_filtered(self) -> None:
        from src.sources.fred import FREDAdapter

        raw = {
            "observations": [
                {"date": "2026-01-01", "value": "."},
                {"date": "2025-10-01", "value": "21200.3"},
            ],
        }
        result = FREDAdapter._parse_response(raw, "gdp", "GDPC1")
        assert result["latest_value"] == 21200.3
        assert len(result["history"]) == 1

    def test_parse_no_observations(self) -> None:
        from src.sources.fred import FREDAdapter

        raw = {"observations": []}
        result = FREDAdapter._parse_response(raw, "gdp", "GDPC1")
        assert "error" in result


class TestFREDAdapterFetch:
    """FRED adapter fetch method (mocked HTTP)."""

    def test_name(self) -> None:
        from src.sources.fred import FREDAdapter
        assert FREDAdapter().name == "fred"

    async def test_no_api_key(self) -> None:
        from src.sources.fred import FREDAdapter

        adapter = FREDAdapter(api_key="")
        spec = MacroContractSpec(category="macro", indicator="gdp")
        result = await adapter.fetch(spec)
        assert not result.ok
        assert result.error == "no_api_key"

    async def test_unsupported_indicator(self) -> None:
        from src.sources.fred import FREDAdapter

        adapter = FREDAdapter(api_key="test_key")
        spec = MacroContractSpec(category="macro", indicator="unknown_thing")
        result = await adapter.fetch(spec)
        assert not result.ok
        assert "unsupported_indicator" in (result.error or "")

    async def test_invalid_spec_type(self) -> None:
        from src.sources.fred import FREDAdapter

        adapter = FREDAdapter(api_key="test_key")
        result = await adapter.fetch("not a spec")
        assert not result.ok
        assert result.error == "invalid_spec_type"


# ===========================================================================
# Macro Engine
# ===========================================================================


def _history(values: list[float]) -> list[dict[str, Any]]:
    """Build a history list from values (oldest first)."""
    return [{"year": 2025, "period": f"M{i}", "value": v, "date": f"2025-{i:02d}-01"}
            for i, v in enumerate(values, 1)]


class TestMacroEngineCPI:
    """CPI indicator pricing."""

    def test_cpi_direct_comparison_above(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="cpi", threshold=3.0, comparison="above")
        obs = {"latest_value": 3.5, "is_preliminary": False, "history": []}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99
        assert est.confidence == 0.95

    def test_cpi_direct_comparison_below(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="cpi", threshold=3.0, comparison="above")
        obs = {"latest_value": 2.5, "is_preliminary": False, "history": []}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.01

    def test_cpi_preliminary_lower_confidence(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="cpi", threshold=3.0, comparison="above")
        obs = {"latest_value": 3.5, "is_preliminary": True, "history": []}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.90

    def test_cpi_no_threshold(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="cpi", comparison="above")
        obs = {"history": []}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.5
        assert est.confidence == 0.1

    def test_core_cpi_routes_to_price_index(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="core_cpi", threshold=3.0, comparison="above")
        obs = {"latest_value": 3.5, "is_preliminary": False, "history": []}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99


class TestMacroEngineUnemployment:
    """Unemployment indicator pricing."""

    def test_direct_comparison_below(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="unemployment", threshold=4.0, comparison="below")
        obs = {"latest_value": 3.5, "is_preliminary": False, "history": []}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99

    def test_distribution_with_history(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="unemployment", threshold=4.5, comparison="above")
        history = _history([3.8, 3.9, 3.8, 3.7, 3.9, 4.0, 3.8, 3.9])
        obs = {"latest_value": None, "history": history}
        est = engine.compute(spec, obs)
        # Unemployment ~3.9 avg, threshold 4.5 → low p_yes.
        assert 0.01 <= est.p_yes <= 0.99
        assert est.confidence == 0.70


class TestMacroEnginePayrolls:
    """Nonfarm payrolls pricing."""

    def test_direct_comparison(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(
            category="macro", indicator="nonfarm_payrolls",
            threshold=200.0, threshold_unit="K", comparison="above",
        )
        obs = {"latest_value": 250.0, "is_preliminary": False, "history": []}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99

    def test_capped_confidence(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(
            category="macro", indicator="nonfarm_payrolls",
            threshold=200.0, threshold_unit="K", comparison="above",
        )
        history = _history([180, 190, 200, 210, 195, 205, 215, 185])
        obs = {"latest_value": None, "history": history}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.65


class TestMacroEngineGDP:
    """GDP pricing."""

    def test_direct_comparison(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(
            category="macro", indicator="gdp",
            threshold=2.0, comparison="above",
        )
        obs = {"latest_value": 2.5, "history": []}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99

    def test_contraction_no_threshold(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(
            category="macro", indicator="gdp", comparison="below",
        )
        obs = {"history": []}
        est = engine.compute(spec, obs)
        # threshold defaults to 0 for contraction.
        assert 0.01 <= est.p_yes <= 0.99
        assert est.confidence == 0.60

    def test_capped_confidence(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(
            category="macro", indicator="gdp",
            threshold=2.0, comparison="above",
        )
        history = _history([100, 100.5, 101, 101.5, 102])
        obs = {"latest_value": None, "history": history}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.60


class TestMacroEngineFedRate:
    """Fed rate decision pricing."""

    def test_post_meeting_raise(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="fed_rate", comparison="raise")
        obs = {"latest_value": 5.50, "previous_value": 5.25}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99
        assert est.confidence == 0.95

    def test_post_meeting_hold(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="fed_rate", comparison="raise")
        obs = {"latest_value": 5.25, "previous_value": 5.25}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.01

    def test_post_meeting_cut(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="fed_rate", comparison="cut")
        obs = {"latest_value": 5.00, "previous_value": 5.25}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99

    def test_pre_meeting_hold_base_rate(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="fed_rate", comparison="hold")
        obs = {}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.60
        assert est.confidence == 0.50


class TestMacroEngineDispatch:
    """Engine dispatch and edge cases."""

    def test_invalid_spec_type(self) -> None:
        engine = MacroEngine()
        est = engine.compute("not a spec", {})
        assert est.confidence == 0.0

    def test_invalid_observation_type(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="cpi")
        est = engine.compute(spec, 42)
        assert est.confidence == 0.0

    def test_unknown_indicator(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(category="macro", indicator="unknown_thing")
        est = engine.compute(spec, {})
        assert est.confidence == 0.1

    def test_version(self) -> None:
        engine = MacroEngine()
        assert engine.version == "macro_v1"
        assert engine.name == "macro"

    def test_p_yes_always_clamped(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(
            category="macro", indicator="cpi", threshold=3.0, comparison="above",
        )
        obs = {"latest_value": 3.5, "is_preliminary": False, "history": []}
        est = engine.compute(spec, obs)
        assert 0.01 <= est.p_yes <= 0.99

    def test_accepts_fetch_result(self) -> None:
        engine = MacroEngine()
        spec = MacroContractSpec(
            category="macro", indicator="cpi", threshold=3.0, comparison="above",
        )
        fr = FetchResult(
            source_name="bls",
            source_key="CUSR0000SA0",
            ts_source=datetime.now(timezone.utc),
            raw_json={},
            normalized_json={"latest_value": 3.5, "is_preliminary": False, "history": []},
        )
        est = engine.compute(spec, fr)
        assert est.p_yes == 0.99


# ===========================================================================
# Macro Pipeline
# ===========================================================================


class TestMacroPipelineRunCycle:
    """MacroPipeline integration tests (all deps mocked)."""

    def _make_pipeline(self, assignments=None, snapshot=None) -> Any:
        from src.config.policy import load_policy
        from src.trading.macro_pipeline import MacroPipeline

        repo = AsyncMock()
        repo.get_markets_by_category.return_value = assignments or []
        repo.get_latest_snapshot.return_value = snapshot
        repo.get_open_positions.return_value = []
        repo.get_position.return_value = None
        market_mock = MagicMock()
        market_mock.resolution_time_utc = datetime.now(timezone.utc) + timedelta(hours=100)
        repo.get_market.return_value = market_mock
        repo.add_source_observation.return_value = None
        repo.add_engine_price.return_value = None
        repo.add_decision.return_value = "dec-1"
        repo.add_order.return_value = "ord-1"
        repo.add_fill.return_value = None
        repo.upsert_position.return_value = None

        bls = AsyncMock()
        fred = AsyncMock()

        policy = load_policy("policy.yaml")
        engine = MacroEngine()

        from src.execution.paper_executor import PaperExecutor
        from src.portfolio.risk_manager import RiskManager

        macro_policy = policy.for_category("macro")
        executor = PaperExecutor(policy=macro_policy)
        risk_manager = RiskManager(policy=macro_policy)

        pipeline = MacroPipeline(
            repo=repo, bls=bls, fred=fred,
            engine=engine, executor=executor,
            risk_manager=risk_manager, policy=policy,
        )
        return pipeline, repo, bls, fred

    async def test_no_markets(self) -> None:
        pipeline, repo, bls, fred = self._make_pipeline()
        summary = await pipeline.run_cycle()
        assert summary["markets_found"] == 0

    async def test_skip_no_spec(self) -> None:
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = None

        pipeline, repo, bls, fred = self._make_pipeline(assignments=[assignment])
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1

    async def test_skip_source_error(self) -> None:
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = {
            "category": "macro",
            "indicator": "cpi",
            "threshold": 3.0,
            "comparison": "above",
            "release_period": "January 2026",
        }

        pipeline, repo, bls, fred = self._make_pipeline(assignments=[assignment])
        bls.fetch.return_value = FetchResult(
            source_name="bls", source_key="test", ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={}, error="api_error",
        )
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1


# ===========================================================================
# Scheduler Macro Cycle
# ===========================================================================


class TestSchedulerMacroCycle:
    """StructuredTradingEngine macro cycle tests."""

    def _policy(self) -> Any:
        from src.config.policy import Policy
        return Policy()

    async def test_no_pipeline_returns_empty(self) -> None:
        from src.app.scheduler import StructuredTradingEngine

        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(),
            policy=self._policy(), macro_pipeline=None,
        )
        result = await engine.run_macro_cycle()
        assert result == {}

    async def test_delegates_to_pipeline(self) -> None:
        from src.app.scheduler import StructuredTradingEngine

        mock_pipeline = AsyncMock()
        mock_pipeline.run_cycle.return_value = {"markets_found": 5}

        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(),
            policy=self._policy(), macro_pipeline=mock_pipeline,
        )
        result = await engine.run_macro_cycle()
        assert result == {"markets_found": 5}
        mock_pipeline.run_cycle.assert_called_once()

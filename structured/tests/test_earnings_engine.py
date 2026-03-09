"""Tests for earnings engine — ticker resolver, EDGAR adapter, pricing engine,
pipeline, and scheduler wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.contracts.earnings import EarningsContractSpec
from src.engines.earnings import EarningsEngine, _scale_threshold
from src.sources.base import FetchResult
from src.sources.ticker_resolver import resolve_company, resolve_ticker


# ===========================================================================
# Ticker Resolver
# ===========================================================================


class TestTickerResolver:
    """Static ticker → CIK resolution."""

    def test_resolve_known_ticker(self) -> None:
        result = resolve_ticker("AAPL")
        assert result is not None
        cik, name = result
        assert cik == "0000320193"
        assert "Apple" in name

    def test_resolve_lowercase_ticker(self) -> None:
        result = resolve_ticker("aapl")
        assert result is not None
        assert result[0] == "0000320193"

    def test_resolve_unknown_ticker(self) -> None:
        result = resolve_ticker("ZZZZZZ")
        assert result is None

    def test_resolve_msft(self) -> None:
        result = resolve_ticker("MSFT")
        assert result is not None
        assert result[0] == "0000789019"
        assert "Microsoft" in result[1]

    def test_resolve_tsla(self) -> None:
        result = resolve_ticker("TSLA")
        assert result is not None
        assert "Tesla" in result[1]

    def test_resolve_nvda(self) -> None:
        result = resolve_ticker("NVDA")
        assert result is not None
        assert "NVIDIA" in result[1]

    def test_resolve_company_apple(self) -> None:
        result = resolve_company("Apple")
        assert result is not None
        cik, ticker = result
        assert cik == "0000320193"
        assert ticker == "AAPL"

    def test_resolve_company_case_insensitive(self) -> None:
        result = resolve_company("apple")
        assert result is not None
        assert result[1] == "AAPL"

    def test_resolve_company_partial(self) -> None:
        result = resolve_company("Tesla")
        assert result is not None
        assert result[1] == "TSLA"

    def test_resolve_company_unknown(self) -> None:
        result = resolve_company("Nonexistent Corp")
        assert result is None


# ===========================================================================
# EDGAR Adapter
# ===========================================================================


class TestEDGARAdapterBasic:
    """EDGARAdapter unit tests."""

    def test_name(self) -> None:
        from src.sources.edgar import EDGARAdapter
        assert EDGARAdapter().name == "edgar"

    async def test_invalid_spec_type(self) -> None:
        from src.sources.edgar import EDGARAdapter
        adapter = EDGARAdapter()
        result = await adapter.fetch(MagicMock(spec=[]))
        assert not result.ok
        assert "invalid_spec_type" in (result.error or "")

    async def test_cik_not_resolved(self) -> None:
        from src.sources.edgar import EDGARAdapter
        adapter = EDGARAdapter()
        spec = EarningsContractSpec(
            category="earnings", company="", ticker="ZZZZZZ", metric="eps",
        )
        result = await adapter.fetch(spec)
        assert not result.ok
        assert "cik_not_resolved" in (result.error or "")

    async def test_resolves_ticker_to_cik(self) -> None:
        from src.sources.edgar import EDGARAdapter
        adapter = EDGARAdapter()
        spec = EarningsContractSpec(
            category="earnings", ticker="AAPL", metric="eps",
        )
        # Mock the HTTP call
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {"val": 1.52, "end": "2025-12-31", "filed": "2026-01-30", "fp": "Q1"},
                            ]
                        }
                    }
                }
            }
        }
        with patch("src.sources.edgar.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await adapter.fetch(spec)
            assert result.ok
            assert result.normalized_json["latest_value"] == 1.52


class TestEDGARParserFacts:
    """EDGARAdapter._parse_facts tests."""

    def test_parse_eps_facts(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {"val": 1.52, "end": "2025-12-31", "filed": "2026-01-30", "fp": "Q1"},
                                {"val": 1.40, "end": "2025-09-30", "filed": "2025-10-30", "fp": "Q4"},
                            ]
                        }
                    }
                }
            }
        }
        spec = EarningsContractSpec(category="earnings", metric="eps")
        result = EDGARAdapter._parse_facts(raw, spec)
        assert result["metric"] == "eps"
        assert result["latest_value"] == 1.52
        assert result["has_filed"] is True
        assert len(result["history"]) == 2

    def test_parse_revenue_facts(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"val": 94836000000, "end": "2025-12-31", "filed": "2026-01-30", "fp": "Q1"},
                            ]
                        }
                    }
                }
            }
        }
        spec = EarningsContractSpec(category="earnings", metric="revenue")
        result = EDGARAdapter._parse_facts(raw, spec)
        assert result["metric"] == "revenue"
        assert result["latest_value"] == 94836000000
        assert result["has_filed"] is True

    def test_parse_no_xbrl_data(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {"facts": {"us-gaap": {}}}
        spec = EarningsContractSpec(category="earnings", metric="eps")
        result = EDGARAdapter._parse_facts(raw, spec)
        assert result["error"] == "no_xbrl_data"
        assert result["has_filed"] is False

    def test_parse_unsupported_metric(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {"facts": {"us-gaap": {}}}
        spec = EarningsContractSpec(category="earnings", metric="guidance")
        result = EDGARAdapter._parse_facts(raw, spec)
        assert "unsupported_metric" in result["error"]

    def test_parse_empty_values_list(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {
                        "units": {"USD/shares": []}
                    }
                }
            }
        }
        spec = EarningsContractSpec(category="earnings", metric="eps")
        result = EDGARAdapter._parse_facts(raw, spec)
        assert result["error"] == "no_xbrl_data"

    def test_parse_history_limited_to_12(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {"val": i * 0.1, "end": f"2025-{(i % 12) + 1:02d}-30", "fp": "Q1"}
                                for i in range(20)
                            ]
                        }
                    }
                }
            }
        }
        spec = EarningsContractSpec(category="earnings", metric="eps")
        result = EDGARAdapter._parse_facts(raw, spec)
        assert len(result["history"]) == 12


class TestEDGARParserSubmissions:
    """EDGARAdapter._parse_submissions tests."""

    def test_filing_found(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "name": "Apple Inc.",
            "cik": "0000320193",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K"],
                    "filingDate": ["2025-11-01", "2025-08-01", "2025-07-15"],
                }
            }
        }
        spec = EarningsContractSpec(category="earnings", filing_type="10-K")
        result = EDGARAdapter._parse_submissions(raw, spec)
        assert result["filing_found"] is True
        assert result["filing_type"] == "10-K"
        assert result["filing_date"] == "2025-11-01"

    def test_filing_not_found(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "name": "Apple Inc.",
            "cik": "0000320193",
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K"],
                    "filingDate": ["2025-08-01", "2025-07-15"],
                }
            }
        }
        spec = EarningsContractSpec(category="earnings", filing_type="10-K")
        result = EDGARAdapter._parse_submissions(raw, spec)
        assert result["filing_found"] is False
        assert result["filing_type"] == "10-K"

    def test_default_filing_type_10k(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "name": "Test Corp",
            "cik": "0000000001",
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2025-11-01"],
                }
            }
        }
        spec = EarningsContractSpec(category="earnings")  # No filing_type
        result = EDGARAdapter._parse_submissions(raw, spec)
        assert result["filing_found"] is True

    def test_case_insensitive_match(self) -> None:
        from src.sources.edgar import EDGARAdapter
        raw = {
            "name": "Test Corp",
            "cik": "0000000001",
            "filings": {
                "recent": {
                    "form": ["10-k"],  # lowercase
                    "filingDate": ["2025-11-01"],
                }
            }
        }
        spec = EarningsContractSpec(category="earnings", filing_type="10-K")
        result = EDGARAdapter._parse_submissions(raw, spec)
        assert result["filing_found"] is True


class TestEDGARHealthCheck:
    """EDGARAdapter.health_check tests."""

    async def test_health_check_success(self) -> None:
        from src.sources.edgar import EDGARAdapter
        adapter = EDGARAdapter()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("src.sources.edgar.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await adapter.health_check()
            assert result is True

    async def test_health_check_failure(self) -> None:
        from src.sources.edgar import EDGARAdapter
        adapter = EDGARAdapter()

        with patch("src.sources.edgar.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("connection error"))
            mock_client_cls.return_value = mock_client

            result = await adapter.health_check()
            assert result is False


# ===========================================================================
# Scale Threshold Helper
# ===========================================================================


class TestScaleThreshold:
    """Test _scale_threshold helper."""

    def test_billions(self) -> None:
        assert _scale_threshold(25.0, "B") == 25_000_000_000.0

    def test_millions(self) -> None:
        assert _scale_threshold(100.0, "M") == 100_000_000.0

    def test_thousands(self) -> None:
        assert _scale_threshold(500.0, "K") == 500_000.0

    def test_usd_no_scale(self) -> None:
        assert _scale_threshold(1.50, "USD") == 1.50

    def test_empty_no_scale(self) -> None:
        assert _scale_threshold(1.50, "") == 1.50


# ===========================================================================
# Earnings Engine
# ===========================================================================


class TestEarningsEngineBasic:
    """Basic EarningsEngine tests."""

    def test_version(self) -> None:
        engine = EarningsEngine()
        assert engine.version == "earnings_v1"
        assert engine.name == "earnings"

    def test_invalid_spec_type(self) -> None:
        engine = EarningsEngine()
        est = engine.compute("not a spec", {})
        assert est.confidence == 0.0

    def test_invalid_observation_type(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(category="earnings", metric="eps")
        est = engine.compute(spec, 42)
        assert est.confidence == 0.0


class TestEarningsEngineFinancialMetric:
    """EarningsEngine financial metric (EPS/revenue) tests."""

    def test_post_filing_eps_above(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            threshold=1.50, comparison="above",
        )
        obs = {"has_filed": True, "latest_value": 1.75}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99
        assert est.confidence == 0.95

    def test_post_filing_eps_below(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            threshold=1.50, comparison="above",
        )
        obs = {"has_filed": True, "latest_value": 1.20}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.01
        assert est.confidence == 0.95

    def test_post_filing_eps_equal(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            threshold=1.50, comparison="above",
        )
        obs = {"has_filed": True, "latest_value": 1.50}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.50

    def test_post_filing_below_comparison(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            threshold=1.50, comparison="below",
        )
        obs = {"has_filed": True, "latest_value": 1.20}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99

    def test_post_filing_revenue_billions(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="revenue",
            threshold=25.0, threshold_unit="B", comparison="above",
        )
        obs = {"has_filed": True, "latest_value": 30_000_000_000.0}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99

    def test_post_filing_revenue_below_threshold(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="revenue",
            threshold=25.0, threshold_unit="B", comparison="above",
        )
        obs = {"has_filed": True, "latest_value": 20_000_000_000.0}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.01

    def test_pre_filing_returns_uniform(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            threshold=1.50, comparison="above",
        )
        obs = {"has_filed": False}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.50
        assert est.confidence == 0.30

    def test_pre_filing_custom_confidence(self) -> None:
        engine = EarningsEngine(pre_filing_confidence=0.20)
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            threshold=1.50, comparison="above",
        )
        obs = {"has_filed": False}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.20

    def test_no_threshold_returns_low_conf(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            comparison="above",  # No threshold
        )
        obs = {"has_filed": True, "latest_value": 1.75}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.50
        assert est.confidence == 0.30

    def test_accepts_fetch_result(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="eps",
            threshold=1.50, comparison="above",
        )
        fr = FetchResult(
            source_name="edgar", source_key="0000320193",
            ts_source=datetime.now(timezone.utc),
            raw_json={},
            normalized_json={"has_filed": True, "latest_value": 1.75},
        )
        est = engine.compute(spec, fr)
        assert est.p_yes == 0.99


class TestEarningsEngineFilingExistence:
    """EarningsEngine filing existence tests."""

    def test_filing_confirmed(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="filing_10k",
            filing_type="10-K",
        )
        obs = {
            "filing_found": True,
            "filing_type": "10-K",
            "filing_date": "2025-11-01",
            "company": "Apple Inc.",
        }
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.99
        assert est.confidence == 0.95

    def test_filing_not_found(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="filing_10k",
            filing_type="10-K",
        )
        obs = {
            "filing_found": False,
            "filing_type": "10-K",
            "company": "Apple Inc.",
        }
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.85
        assert est.confidence == 0.40

    def test_unsupported_metric(self) -> None:
        engine = EarningsEngine()
        spec = EarningsContractSpec(
            category="earnings", metric="guidance",
        )
        est = engine.compute(spec, {})
        assert est.confidence == 0.1


# ===========================================================================
# Earnings Pipeline
# ===========================================================================


class TestEarningsPipelineRunCycle:
    """EarningsPipeline integration tests (all deps mocked)."""

    def _make_pipeline(self, assignments=None, snapshot=None) -> Any:
        from src.config.policy import load_policy
        from src.execution.paper_executor import PaperExecutor
        from src.portfolio.risk_manager import RiskManager
        from src.sources.edgar import EDGARAdapter
        from src.trading.earnings_pipeline import EarningsPipeline

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

        edgar = AsyncMock(spec=EDGARAdapter)
        edgar.name = "edgar"

        policy = load_policy("policy.yaml")
        engine = EarningsEngine()

        earnings_policy = policy.for_category("earnings")
        executor = PaperExecutor(policy=earnings_policy)
        risk_manager = RiskManager(policy=earnings_policy)

        pipeline = EarningsPipeline(
            repo=repo, edgar=edgar,
            engine=engine, executor=executor,
            risk_manager=risk_manager, policy=policy,
        )
        return pipeline, repo, edgar

    async def test_no_markets(self) -> None:
        pipeline, repo, edgar = self._make_pipeline()
        summary = await pipeline.run_cycle()
        assert summary["markets_found"] == 0

    async def test_skip_no_spec(self) -> None:
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = None

        pipeline, repo, edgar = self._make_pipeline(assignments=[assignment])
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1

    async def test_skip_source_error(self) -> None:
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = {
            "category": "earnings",
            "company": "Apple",
            "ticker": "AAPL",
            "metric": "eps",
            "threshold": 1.50,
            "comparison": "above",
        }

        pipeline, repo, edgar = self._make_pipeline(assignments=[assignment])
        edgar.fetch.return_value = FetchResult(
            source_name="edgar", source_key="0000320193",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={}, error="api_error",
        )
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1


# ===========================================================================
# Scheduler Earnings Cycle
# ===========================================================================


class TestSchedulerEarningsCycle:
    """StructuredTradingEngine earnings cycle tests."""

    def _policy(self) -> Any:
        from src.config.policy import Policy
        return Policy()

    async def test_no_pipeline_returns_empty(self) -> None:
        from src.app.scheduler import StructuredTradingEngine

        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(),
            policy=self._policy(), earnings_pipeline=None,
        )
        result = await engine.run_earnings_cycle()
        assert result == {}

    async def test_delegates_to_pipeline(self) -> None:
        from src.app.scheduler import StructuredTradingEngine

        mock_pipeline = AsyncMock()
        mock_pipeline.run_cycle.return_value = {"markets_found": 3}

        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(),
            policy=self._policy(), earnings_pipeline=mock_pipeline,
        )
        result = await engine.run_earnings_cycle()
        assert result == {"markets_found": 3}
        mock_pipeline.run_cycle.assert_called_once()

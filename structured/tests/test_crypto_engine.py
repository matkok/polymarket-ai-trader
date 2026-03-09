"""Tests for crypto engine — exchange adapters, pricing engine, pipeline,
and scheduler wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.contracts.crypto import CryptoContractSpec
from src.engines.crypto import CryptoEngine
from src.sources.base import FetchResult


# ===========================================================================
# Exchange Adapters (unit tests)
# ===========================================================================


class TestCoinbaseAdapter:
    def test_name(self) -> None:
        from src.sources.coinbase import CoinbaseAdapter
        assert CoinbaseAdapter().name == "coinbase"

    async def test_unsupported_asset(self) -> None:
        from src.sources.coinbase import CoinbaseAdapter
        adapter = CoinbaseAdapter()
        spec = MagicMock()
        spec.asset = "UNKNOWN"
        result = await adapter.fetch(spec)
        assert not result.ok
        assert "unsupported_asset" in (result.error or "")

    async def test_invalid_spec(self) -> None:
        from src.sources.coinbase import CoinbaseAdapter
        adapter = CoinbaseAdapter()
        spec = MagicMock(spec=[])  # No 'asset' attribute.
        result = await adapter.fetch(spec)
        assert not result.ok


class TestBinanceAdapter:
    def test_name(self) -> None:
        from src.sources.binance import BinanceAdapter
        assert BinanceAdapter().name == "binance"

    async def test_unsupported_asset(self) -> None:
        from src.sources.binance import BinanceAdapter
        adapter = BinanceAdapter()
        spec = MagicMock()
        spec.asset = "UNKNOWN"
        result = await adapter.fetch(spec)
        assert not result.ok


class TestKrakenAdapter:
    def test_name(self) -> None:
        from src.sources.kraken import KrakenAdapter
        assert KrakenAdapter().name == "kraken"

    async def test_unsupported_asset(self) -> None:
        from src.sources.kraken import KrakenAdapter
        adapter = KrakenAdapter()
        spec = MagicMock()
        spec.asset = "UNKNOWN"
        result = await adapter.fetch(spec)
        assert not result.ok


class TestExchangeRouter:
    def test_name(self) -> None:
        from src.sources.exchange_router import ExchangeRouter
        assert ExchangeRouter().name == "exchange_router"

    def test_get_adapter_coinbase(self) -> None:
        from src.sources.exchange_router import ExchangeRouter
        router = ExchangeRouter()
        adapter = router.get_adapter("coinbase")
        assert adapter.name == "coinbase"

    def test_get_adapter_binance(self) -> None:
        from src.sources.exchange_router import ExchangeRouter
        router = ExchangeRouter()
        adapter = router.get_adapter("binance")
        assert adapter.name == "binance"

    def test_get_adapter_kraken(self) -> None:
        from src.sources.exchange_router import ExchangeRouter
        router = ExchangeRouter()
        adapter = router.get_adapter("kraken")
        assert adapter.name == "kraken"

    def test_get_adapter_default_coinbase(self) -> None:
        from src.sources.exchange_router import ExchangeRouter
        router = ExchangeRouter()
        adapter = router.get_adapter("")
        assert adapter.name == "coinbase"

    async def test_routes_to_correct_adapter(self) -> None:
        from src.sources.exchange_router import ExchangeRouter
        mock_coinbase = AsyncMock()
        mock_coinbase.name = "coinbase"
        mock_coinbase.fetch.return_value = FetchResult(
            source_name="coinbase", source_key="BTC-USD",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={"price": 100000.0},
        )
        router = ExchangeRouter(coinbase=mock_coinbase)
        spec = CryptoContractSpec(category="crypto", asset="BTC", exchange="coinbase")
        result = await router.fetch(spec)
        assert result.ok
        mock_coinbase.fetch.assert_called_once()


# ===========================================================================
# Crypto Engine
# ===========================================================================


class TestCryptoEngineBasic:
    """Basic CDF pricing tests."""

    def test_btc_well_above_threshold(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=80000.0, comparison="above",
        )
        obs = {"price": 100000.0, "hours_remaining": 24 * 7}
        est = engine.compute(spec, obs)
        assert est.p_yes > 0.80
        assert est.confidence == 0.70

    def test_btc_well_below_threshold(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=150000.0, comparison="above",
        )
        obs = {"price": 100000.0, "hours_remaining": 24 * 7}
        est = engine.compute(spec, obs)
        assert est.p_yes < 0.30

    def test_eth_below_comparison(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="ETH",
            threshold=5000.0, comparison="below",
        )
        obs = {"price": 3000.0, "hours_remaining": 24 * 7}
        est = engine.compute(spec, obs)
        assert est.p_yes > 0.60

    def test_near_resolution_above(self) -> None:
        engine = CryptoEngine(near_resolution_hours=1.0)
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 105000.0, "hours_remaining": 0.5}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.95
        assert est.confidence == 0.90

    def test_near_resolution_below(self) -> None:
        engine = CryptoEngine(near_resolution_hours=1.0)
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 95000.0, "hours_remaining": 0.5}
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.05

    def test_near_resolution_too_close(self) -> None:
        engine = CryptoEngine(near_resolution_hours=1.0)
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 99900.0, "hours_remaining": 0.5}  # Within 2%
        est = engine.compute(spec, obs)
        assert est.p_yes == 0.50

    def test_p_yes_clamped(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 100000.0, "hours_remaining": 24 * 30}
        est = engine.compute(spec, obs)
        assert 0.01 <= est.p_yes <= 0.99


class TestCryptoEngineConfidence:
    """Confidence levels at various time horizons."""

    def test_confidence_within_1_day(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 100000.0, "hours_remaining": 20}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.85

    def test_confidence_within_7_days(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 100000.0, "hours_remaining": 24 * 5}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.70

    def test_confidence_within_30_days(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 100000.0, "hours_remaining": 24 * 20}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.55

    def test_confidence_beyond_30_days(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        obs = {"price": 100000.0, "hours_remaining": 24 * 60}
        est = engine.compute(spec, obs)
        assert est.confidence == 0.50


class TestCryptoEngineDispatch:
    """Engine dispatch and edge cases."""

    def test_invalid_spec_type(self) -> None:
        engine = CryptoEngine()
        est = engine.compute("not a spec", {})
        assert est.confidence == 0.0

    def test_invalid_observation_type(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(category="crypto", asset="BTC")
        est = engine.compute(spec, 42)
        assert est.confidence == 0.0

    def test_no_price(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=100000.0, comparison="above",
        )
        est = engine.compute(spec, {})
        assert est.confidence == 0.1

    def test_no_threshold(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(category="crypto", asset="BTC", comparison="above")
        est = engine.compute(spec, {"price": 100000.0})
        assert est.confidence == 0.1

    def test_version(self) -> None:
        engine = CryptoEngine()
        assert engine.version == "crypto_v1"
        assert engine.name == "crypto"

    def test_accepts_fetch_result(self) -> None:
        engine = CryptoEngine()
        spec = CryptoContractSpec(
            category="crypto", asset="BTC",
            threshold=80000.0, comparison="above",
        )
        fr = FetchResult(
            source_name="coinbase", source_key="BTC-USD",
            ts_source=datetime.now(timezone.utc),
            raw_json={},
            normalized_json={"price": 100000.0, "hours_remaining": 24 * 7},
        )
        est = engine.compute(spec, fr)
        assert est.p_yes > 0.80


# ===========================================================================
# Crypto Pipeline
# ===========================================================================


class TestCryptoPipelineRunCycle:
    """CryptoPipeline integration tests (all deps mocked)."""

    def _make_pipeline(self, assignments=None, snapshot=None) -> Any:
        from src.config.policy import load_policy
        from src.execution.paper_executor import PaperExecutor
        from src.portfolio.risk_manager import RiskManager
        from src.sources.exchange_router import ExchangeRouter
        from src.trading.crypto_pipeline import CryptoPipeline

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

        exchange_router = AsyncMock(spec=ExchangeRouter)
        exchange_router.name = "exchange_router"

        policy = load_policy("policy.yaml")
        engine = CryptoEngine()

        crypto_policy = policy.for_category("crypto")
        executor = PaperExecutor(policy=crypto_policy)
        risk_manager = RiskManager(policy=crypto_policy)

        pipeline = CryptoPipeline(
            repo=repo, exchange_router=exchange_router,
            engine=engine, executor=executor,
            risk_manager=risk_manager, policy=policy,
        )
        return pipeline, repo, exchange_router

    async def test_no_markets(self) -> None:
        pipeline, repo, router = self._make_pipeline()
        summary = await pipeline.run_cycle()
        assert summary["markets_found"] == 0

    async def test_skip_no_spec(self) -> None:
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = None

        pipeline, repo, router = self._make_pipeline(assignments=[assignment])
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1

    async def test_skip_source_error(self) -> None:
        assignment = MagicMock()
        assignment.market_id = "mkt-1"
        assignment.contract_spec_json = {
            "category": "crypto",
            "asset": "BTC",
            "threshold": 100000.0,
            "comparison": "above",
            "exchange": "coinbase",
        }

        pipeline, repo, router = self._make_pipeline(assignments=[assignment])
        router.fetch.return_value = FetchResult(
            source_name="coinbase", source_key="BTC-USD",
            ts_source=datetime.now(timezone.utc),
            raw_json={}, normalized_json={}, error="api_error",
        )
        summary = await pipeline.run_cycle()
        assert summary["markets_skipped"] == 1


# ===========================================================================
# Scheduler Crypto Cycle
# ===========================================================================


class TestSchedulerCryptoCycle:
    """StructuredTradingEngine crypto cycle tests."""

    def _policy(self) -> Any:
        from src.config.policy import Policy
        return Policy()

    async def test_no_pipeline_returns_empty(self) -> None:
        from src.app.scheduler import StructuredTradingEngine

        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(),
            policy=self._policy(), crypto_pipeline=None,
        )
        result = await engine.run_crypto_cycle()
        assert result == {}

    async def test_delegates_to_pipeline(self) -> None:
        from src.app.scheduler import StructuredTradingEngine

        mock_pipeline = AsyncMock()
        mock_pipeline.run_cycle.return_value = {"markets_found": 3}

        engine = StructuredTradingEngine(
            repo=AsyncMock(), gamma_client=AsyncMock(),
            policy=self._policy(), crypto_pipeline=mock_pipeline,
        )
        result = await engine.run_crypto_cycle()
        assert result == {"markets_found": 3}
        mock_pipeline.run_cycle.assert_called_once()

"""Application entrypoint — wires components and starts the scheduler.

Configures logging, loads policy, bootstraps the database, and launches
APScheduler with periodic jobs for market ingestion and classification.
"""

from __future__ import annotations

import asyncio
import logging

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.app.scheduler import StructuredTradingEngine
from src.config.policy import load_policy
from src.evaluation.kill_switch import KillSwitch, KillSwitchConfig
from src.config.settings import get_settings
from src.db.models import Base
from src.db.repository import Repository
from src.db.session import get_engine, get_session_factory
from src.engines.crypto import CryptoEngine
from src.engines.earnings import EarningsEngine
from src.engines.macro import MacroEngine
from src.engines.weather import WeatherEngine
from src.execution.paper_executor import PaperExecutor
from src.polymarket.gamma_client import GammaClient
from src.portfolio.risk_manager import RiskManager
from src.sources.awc import AWCAdapter
from src.sources.bls import BLSAdapter
from src.sources.edgar import EDGARAdapter
from src.sources.exchange_router import ExchangeRouter
from src.sources.fred import FREDAdapter
from src.sources.nws import NWSAdapter
from src.trading.crypto_pipeline import CryptoPipeline
from src.trading.earnings_pipeline import EarningsPipeline
from src.trading.macro_pipeline import MacroPipeline
from src.trading.weather_pipeline import WeatherPipeline

logger = structlog.get_logger(__name__)


def configure_logging(level: str) -> None:
    """Configure structlog for console output."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s")
    logging.getLogger().setLevel(numeric_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
    )


async def run_migrations(engine) -> None:  # type: ignore[no-untyped-def]
    """Create all tables using SQLAlchemy metadata."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def main() -> None:
    """Wire all components together and run the trading loop."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info(
        "structured_trader_starting", database=settings.database_url[:30] + "..."
    )

    # Load policy.
    policy = load_policy(settings.policy_path)
    logger.info("policy_loaded", path=settings.policy_path)

    # Setup DB.
    engine = get_engine(settings.database_url)
    session_factory = get_session_factory(engine)
    await run_migrations(engine)

    # Setup components.
    repo = Repository(session_factory)
    gamma_client = GammaClient(base_url=settings.gamma_api_base_url)

    # Weather pipeline.
    weather_policy = policy.for_category("weather")
    engine_params = policy.categories.get("weather", None)
    forecast_horizon = 168.0
    if engine_params and engine_params.engine_params:
        forecast_horizon = float(
            engine_params.engine_params.get("forecast_horizon_hours", 168.0)
        )

    nws_adapter = NWSAdapter(user_agent=settings.nws_user_agent)
    awc_adapter = AWCAdapter()
    weather_engine = WeatherEngine(forecast_horizon_hours=forecast_horizon)
    paper_executor = PaperExecutor(policy=weather_policy)
    risk_manager = RiskManager(policy=weather_policy)

    weather_pipeline = WeatherPipeline(
        repo=repo,
        nws=nws_adapter,
        awc=awc_adapter,
        engine=weather_engine,
        executor=paper_executor,
        risk_manager=risk_manager,
        policy=policy,
    )

    # Macro pipeline.
    macro_policy = policy.for_category("macro")
    macro_engine_params = policy.categories.get("macro", None)
    lookback_months = 24
    if macro_engine_params and macro_engine_params.engine_params:
        lookback_months = int(
            macro_engine_params.engine_params.get("lookback_months", 24)
        )

    bls_adapter = BLSAdapter(api_key=settings.bls_api_key)
    fred_adapter = FREDAdapter(
        api_key=settings.fred_api_key,
        lookback_months=lookback_months,
    )
    macro_engine = MacroEngine(lookback_months=lookback_months)
    macro_executor = PaperExecutor(policy=macro_policy)
    macro_risk_manager = RiskManager(policy=macro_policy)

    macro_pipeline = MacroPipeline(
        repo=repo,
        bls=bls_adapter,
        fred=fred_adapter,
        engine=macro_engine,
        executor=macro_executor,
        risk_manager=macro_risk_manager,
        policy=policy,
    )

    # Crypto pipeline.
    crypto_policy = policy.for_category("crypto")
    crypto_engine_params = policy.categories.get("crypto", None)
    default_vol = 3.0
    near_res_hours = 1.0
    if crypto_engine_params and crypto_engine_params.engine_params:
        default_vol = float(
            crypto_engine_params.engine_params.get("default_volatility_daily_pct", 3.0)
        )
        near_res_hours = float(
            crypto_engine_params.engine_params.get("near_resolution_hours", 1.0)
        )

    exchange_router = ExchangeRouter()
    crypto_engine = CryptoEngine(
        default_volatility_daily_pct=default_vol,
        near_resolution_hours=near_res_hours,
    )
    crypto_executor = PaperExecutor(policy=crypto_policy)
    crypto_risk_manager = RiskManager(policy=crypto_policy)

    crypto_pipeline = CryptoPipeline(
        repo=repo,
        exchange_router=exchange_router,
        engine=crypto_engine,
        executor=crypto_executor,
        risk_manager=crypto_risk_manager,
        policy=policy,
    )

    # Earnings pipeline.
    earnings_policy = policy.for_category("earnings")
    earnings_engine_params = policy.categories.get("earnings", None)
    pre_filing_conf = 0.30
    if earnings_engine_params and earnings_engine_params.engine_params:
        pre_filing_conf = float(
            earnings_engine_params.engine_params.get("pre_filing_confidence", 0.30)
        )

    edgar_adapter = EDGARAdapter()
    earnings_engine = EarningsEngine(pre_filing_confidence=pre_filing_conf)
    earnings_executor = PaperExecutor(policy=earnings_policy)
    earnings_risk_manager = RiskManager(policy=earnings_policy)

    earnings_pipeline = EarningsPipeline(
        repo=repo,
        edgar=edgar_adapter,
        engine=earnings_engine,
        executor=earnings_executor,
        risk_manager=earnings_risk_manager,
        policy=policy,
    )

    # Kill switch (S6).
    kill_switch = KillSwitch(config=KillSwitchConfig())
    logger.info("kill_switch_configured")

    trading_engine = StructuredTradingEngine(
        repo=repo,
        gamma_client=gamma_client,
        policy=policy,
        weather_pipeline=weather_pipeline,
        macro_pipeline=macro_pipeline,
        crypto_pipeline=crypto_pipeline,
        earnings_pipeline=earnings_pipeline,
        kill_switch=kill_switch,
    )

    # Setup scheduler.
    scheduler = AsyncIOScheduler()

    # Market ingestion every 15 minutes.
    scheduler.add_job(
        trading_engine.ingest_markets,
        "interval",
        minutes=15,
        id="ingest_markets",
        name="Market Ingestion",
        max_instances=1,
        coalesce=True,
    )

    # Market classification every 15 minutes (after ingestion).
    scheduler.add_job(
        trading_engine.classify_markets,
        "interval",
        minutes=15,
        id="classify_markets",
        name="Market Classification",
        max_instances=1,
        coalesce=True,
    )

    # Weather cycle every 2 minutes.
    scheduler.add_job(
        trading_engine.run_weather_cycle,
        "interval",
        minutes=2,
        id="weather_cycle",
        name="Weather Cycle",
        max_instances=1,
        coalesce=True,
    )

    # Macro cycle every 5 minutes.
    scheduler.add_job(
        trading_engine.run_macro_cycle,
        "interval",
        minutes=5,
        id="macro_cycle",
        name="Macro Cycle",
        max_instances=1,
        coalesce=True,
    )

    # Crypto cycle every 1 minute.
    scheduler.add_job(
        trading_engine.run_crypto_cycle,
        "interval",
        minutes=1,
        id="crypto_cycle",
        name="Crypto Cycle",
        max_instances=1,
        coalesce=True,
    )

    # Earnings cycle every 10 minutes.
    scheduler.add_job(
        trading_engine.run_earnings_cycle,
        "interval",
        minutes=10,
        id="earnings_cycle",
        name="Earnings Cycle",
        max_instances=1,
        coalesce=True,
    )

    # Position review every 5 minutes.
    scheduler.add_job(
        trading_engine.review_open_positions,
        "interval",
        minutes=5,
        id="position_review",
        name="Position Review",
        max_instances=1,
        coalesce=True,
    )

    # Daily PnL aggregation at 00:10 UTC (S6).
    scheduler.add_job(
        trading_engine.aggregate_daily_pnl,
        "cron",
        hour=0,
        minute=10,
        id="aggregate_daily_pnl",
        name="Daily PnL Aggregation",
        max_instances=1,
        coalesce=True,
    )

    # Calibration cycle at 00:30 UTC (S6).
    scheduler.add_job(
        trading_engine.run_calibration_cycle,
        "cron",
        hour=0,
        minute=30,
        id="calibration_cycle",
        name="Calibration Cycle",
        max_instances=1,
        coalesce=True,
    )

    # Daily reset at midnight UTC.
    scheduler.add_job(
        trading_engine.reset_daily,
        "cron",
        hour=0,
        minute=0,
        id="daily_reset",
        name="Daily Reset",
        max_instances=1,
        coalesce=True,
    )

    # Run initial ingestion on startup.
    logger.info("running_initial_cycle")
    try:
        await trading_engine.ingest_markets()
    except Exception:
        logger.exception("initial_cycle_error")

    scheduler.start()
    logger.info(
        "scheduler_started", jobs=[j.id for j in scheduler.get_jobs()]
    )

    # Keep running.
    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("shutting_down")
        scheduler.shutdown()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

"""Application entrypoint — wires components and starts the scheduler.

Configures logging, loads policy, bootstraps the database, and launches
APScheduler with periodic jobs for market ingestion, candidate scanning,
position review, and daily resets.
"""

from __future__ import annotations

import asyncio
import logging

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.aggregation.aggregator import Aggregator
from src.app.scheduler import TradingEngine
from src.evidence.embedder import EvidenceEmbedder
from src.evidence.fulltext import FullTextFetcher
from src.config.budget import BudgetTracker, DailyBudget
from src.config.policy import load_policy
from src.config.settings import get_settings
from src.db.models import Base
from src.db.repository import Repository
from src.db.session import get_engine, get_session_factory
from src.evidence.rss_ingestor import RSSIngestor
from src.evidence.xai_search import XAISearchClient
from src.llm.anthropic_client import AnthropicLLMClient
from src.llm.base import BaseLLMClient
from src.llm.gemini_client import GeminiLLMClient
from src.llm.openai_client import OpenAILLMClient
from src.llm.panel import PanelOrchestrator
from src.llm.schemas import DEFAULT_PANEL, ESCALATION_AGENTS
from src.llm.xai_client import XAILLMClient
from src.packets.builder import PacketBuilder
from src.polymarket.clob_client import CLOBClient
from src.polymarket.gamma_client import GammaClient
from src.signals.collector import SignalCollector
from src.signals.trends import GoogleTrendsClient
from src.signals.triage import TriageScorer
from src.signals.wikipedia import WikipediaPageviewClient

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
    """Create all tables using SQLAlchemy metadata.

    For M1 we use ``create_all`` as a simple bootstrap.  Full Alembic
    migration will be used in production.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def main() -> None:
    """Wire all components together and run the trading loop."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info(
        "agent_trader_starting", database=settings.database_url[:30] + "..."
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
    clob_client = CLOBClient(base_url=settings.clob_api_base_url)
    rss_ingestor = RSSIngestor(settings=settings)
    packet_builder = PacketBuilder(policy=policy)

    # Setup LLM panel (M3).
    budget_tracker = BudgetTracker(DailyBudget())
    llm_clients: dict[str, BaseLLMClient] = {}

    if settings.openai_api_key:
        llm_clients["openai"] = OpenAILLMClient(api_key=settings.openai_api_key)
    if settings.anthropic_api_key:
        llm_clients["anthropic"] = AnthropicLLMClient(api_key=settings.anthropic_api_key)
    if settings.google_api_key:
        llm_clients["gemini"] = GeminiLLMClient(api_key=settings.google_api_key)
    if settings.xai_api_key:
        llm_clients["xai"] = XAILLMClient(
            api_key=settings.xai_api_key, base_url=settings.xai_base_url
        )

    # Setup xAI search client (M4).
    xai_search_client = None
    if settings.xai_api_key:
        xai_search_client = XAISearchClient(
            api_key=settings.xai_api_key,
            base_url=settings.xai_base_url,
        )
        logger.info("xai_search_client_configured")

    panel_orchestrator = None
    aggregator = None
    if llm_clients:
        panel_orchestrator = PanelOrchestrator(
            clients=llm_clients,
            budget_tracker=budget_tracker,
            policy=policy,
            charter_path=settings.charter_path,
            default_panel=DEFAULT_PANEL,
            escalation_agents=ESCALATION_AGENTS,
        )
        aggregator = Aggregator(policy=policy)
        logger.info("llm_panel_configured", providers=list(llm_clients.keys()))
    else:
        logger.warning("llm_panel_disabled", reason="no_api_keys_configured")

    # Setup evidence embedder for semantic matching.
    embedder = None
    if settings.openai_api_key:
        embedder = EvidenceEmbedder(api_key=settings.openai_api_key)
        logger.info("evidence_embedder_configured")

    # Setup full-text fetcher for enriching short RSS excerpts.
    fulltext_fetcher = FullTextFetcher()

    # Setup signal components.
    trends_client = None
    if settings.google_trends_enabled:
        trends_client = GoogleTrendsClient(
            trailing_days=settings.google_trends_trailing_days,
            spike_threshold=policy.triage_trends_spike_threshold,
        )
        logger.info("google_trends_client_configured")

    wiki_client = None
    if settings.wikipedia_enabled:
        wiki_client = WikipediaPageviewClient(
            trailing_days=settings.wikipedia_trailing_days,
            spike_threshold=policy.triage_wiki_spike_threshold,
        )
        logger.info("wikipedia_client_configured")

    signal_collector = SignalCollector(
        repo=repo,
        policy=policy,
        trends_client=trends_client,
        wiki_client=wiki_client,
    )
    triage_scorer = TriageScorer(
        policy=policy,
        trends_enabled=settings.google_trends_enabled,
        wiki_enabled=settings.wikipedia_enabled,
    )
    logger.info("signal_components_configured")

    trading_engine = TradingEngine(
        repo=repo,
        gamma_client=gamma_client,
        policy=policy,
        rss_ingestor=rss_ingestor,
        packet_builder=packet_builder,
        panel_orchestrator=panel_orchestrator,
        aggregator=aggregator,
        xai_search_client=xai_search_client,
        embedder=embedder,
        fulltext_fetcher=fulltext_fetcher,
        signal_collector=signal_collector,
        triage_scorer=triage_scorer,
        clob_client=clob_client,
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

    # Evidence ingestion every 20 minutes.
    scheduler.add_job(
        trading_engine.ingest_evidence,
        "interval",
        minutes=20,
        id="ingest_evidence",
        name="Evidence Ingestion",
        max_instances=1,
        coalesce=True,
    )

    # Signal collection every 30 minutes (before candidate scan).
    scheduler.add_job(
        trading_engine.collect_signals,
        "interval",
        minutes=30,
        id="collect_signals",
        name="Signal Collection",
        max_instances=1,
        coalesce=True,
    )

    # Candidate scan every 30 minutes.
    scheduler.add_job(
        trading_engine.run_candidate_scan,
        "interval",
        minutes=30,
        id="candidate_scan",
        name="Candidate Scan",
        max_instances=1,
        coalesce=True,
    )

    # Open position review every 15 minutes.
    scheduler.add_job(
        trading_engine.review_open_positions,
        "interval",
        minutes=15,
        id="position_review",
        name="Position Review",
        max_instances=1,
        coalesce=True,
    )

    # Daily model scoring at 00:30 UTC (after resolutions settle).
    scheduler.add_job(
        trading_engine.run_daily_scoring,
        "cron",
        hour=0,
        minute=30,
        id="daily_scoring",
        name="Daily Model Scoring",
        max_instances=1,
        coalesce=True,
    )

    # Daily reset at midnight UTC.
    def daily_reset() -> None:
        trading_engine.reset_daily()
        budget_tracker.reset_daily()
        if xai_search_client:
            xai_search_client.reset_daily()

    scheduler.add_job(
        daily_reset,
        "cron",
        hour=0,
        minute=0,
        id="daily_reset",
        name="Daily Reset",
        max_instances=1,
        coalesce=True,
    )

    # Run initial ingestion and scan on startup.
    logger.info("running_initial_cycle")
    try:
        await trading_engine.ingest_markets()
        await trading_engine.ingest_evidence()
        await trading_engine.collect_signals()
        await trading_engine.run_candidate_scan()
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

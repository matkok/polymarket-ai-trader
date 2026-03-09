"""Tests for src.app.main — entrypoint wiring and configuration."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import structlog

from src.app.main import configure_logging, main, run_migrations


# ---- configure_logging ------------------------------------------------------


class TestConfigureLogging:
    """Logging configuration tests."""

    def test_sets_log_level_info(self) -> None:
        """INFO level is applied correctly."""
        configure_logging("INFO")
        root = logging.getLogger()
        assert root.level <= logging.INFO

    def test_sets_log_level_debug(self) -> None:
        """DEBUG level is applied correctly."""
        configure_logging("DEBUG")
        root = logging.getLogger()
        assert root.level <= logging.DEBUG

    def test_case_insensitive(self) -> None:
        """Log level string is case-insensitive."""
        configure_logging("info")
        root = logging.getLogger()
        assert root.level <= logging.INFO

    def test_invalid_level_falls_back_to_info(self) -> None:
        """An unrecognised level string falls back to INFO."""
        configure_logging("NONEXISTENT")
        # getattr(logging, "NONEXISTENT", logging.INFO) -> INFO
        root = logging.getLogger()
        assert root.level <= logging.INFO


# ---- run_migrations ---------------------------------------------------------


class TestRunMigrations:
    """Database migration (create_all) tests."""

    async def test_calls_create_all(self) -> None:
        """run_migrations calls Base.metadata.create_all via run_sync."""
        mock_conn = AsyncMock()

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        # engine.begin() is NOT awaited, so engine must be MagicMock.
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_cm

        await run_migrations(mock_engine)

        mock_conn.run_sync.assert_called_once()


# ---- Scheduler job configuration -------------------------------------------


class TestSchedulerJobConfig:
    """Verify scheduler jobs have max_instances=1 and coalesce=True."""

    async def test_all_jobs_have_max_instances_and_coalesce(self) -> None:
        """Every add_job call includes max_instances=1 and coalesce=True."""
        mock_scheduler = MagicMock()

        with (
            patch("src.app.main.get_settings") as mock_settings,
            patch("src.app.main.load_policy"),
            patch("src.app.main.get_engine") as mock_get_engine,
            patch("src.app.main.get_session_factory"),
            patch("src.app.main.run_migrations", new_callable=AsyncMock),
            patch("src.app.main.Repository"),
            patch("src.app.main.GammaClient"),
            patch("src.app.main.RSSIngestor"),
            patch("src.app.main.PacketBuilder"),
            patch("src.app.main.BudgetTracker"),
            patch("src.app.main.DailyBudget"),
            patch("src.app.main.FullTextFetcher"),
            patch("src.app.main.SignalCollector"),
            patch("src.app.main.TriageScorer"),
            patch("src.app.main.TradingEngine") as mock_engine_cls,
            patch("src.app.main.AsyncIOScheduler", return_value=mock_scheduler),
            patch("src.app.main.asyncio.sleep", side_effect=KeyboardInterrupt),
        ):
            settings = MagicMock()
            settings.log_level = "INFO"
            settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
            settings.policy_path = "policy.yaml"
            settings.gamma_api_base_url = "https://example.com"
            settings.openai_api_key = None
            settings.anthropic_api_key = None
            settings.google_api_key = None
            settings.xai_api_key = None
            settings.google_trends_enabled = False
            settings.wikipedia_enabled = False
            settings.charter_path = "charter.md"
            mock_settings.return_value = settings

            mock_engine_instance = MagicMock()
            mock_engine_instance.ingest_markets = AsyncMock()
            mock_engine_instance.ingest_evidence = AsyncMock()
            mock_engine_instance.collect_signals = AsyncMock()
            mock_engine_instance.run_candidate_scan = AsyncMock()
            mock_engine_cls.return_value = mock_engine_instance

            mock_db_engine = MagicMock()
            mock_get_engine.return_value = mock_db_engine
            mock_db_engine.dispose = AsyncMock()

            mock_scheduler.get_jobs.return_value = []
            mock_scheduler.start = MagicMock()
            mock_scheduler.shutdown = MagicMock()

            try:
                await main()
            except (KeyboardInterrupt, SystemExit):
                pass

            # Verify all add_job calls have max_instances=1 and coalesce=True.
            assert mock_scheduler.add_job.call_count == 7
            for c in mock_scheduler.add_job.call_args_list:
                assert c.kwargs.get("max_instances") == 1, (
                    f"Job {c.kwargs.get('id')} missing max_instances=1"
                )
                assert c.kwargs.get("coalesce") is True, (
                    f"Job {c.kwargs.get('id')} missing coalesce=True"
                )

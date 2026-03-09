# CLAUDE.md — Project conventions for agent-trader-structured

## Build and test

```bash
# Run tests (conda env)
conda run -n agent-trader-structured python -m pytest tests/ -v

# Run single test file
conda run -n agent-trader-structured python -m pytest tests/test_scheduler.py -v

# Docker build and run
docker compose up --build -d
docker compose logs -f app
docker compose down
```

## Project overview

Polymarket structured paper trading system. Source-driven, deterministic
probability models. Only trades markets mappable to clear settlement rules.

Key entrypoint: `src/app/main.py` — wires everything together and starts APScheduler.
Core orchestrator: `src/app/scheduler.py` — `StructuredTradingEngine` class drives ingestion.

## Code conventions

- `from __future__ import annotations` at the top of every file
- Type hints on all function signatures
- structlog for logging (module-level `logger = structlog.get_logger(__name__)`)
- Pydantic v2 models for validation (BaseModel, not dataclass for config)
- dataclasses for internal data structures
- SQLAlchemy 2.0 Mapped/mapped_column pattern for ORM models
- Async everywhere — all DB ops and HTTP calls are async
- **Conda environment**: `agent-trader-structured` (Python 3.12)

## Test conventions

- pytest with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- Test classes grouped by module with `# ---- Section ---` separators
- Use `MagicMock(spec=Model)` for SQLAlchemy models in tests
- Use `AsyncMock` for repository and client mocks
- Settings tests use `Settings(_env_file=None)` with `patch.dict(os.environ, {}, clear=True)`

## Database

- Postgres 16 via Docker (port 5433)
- `Base.metadata.create_all` for bootstrap (no Alembic migrations yet)
- Repository pattern: `src/db/repository.py`
- DB name: `agent_trader_structured`

## Policy

- `policy.yaml` — global + category-specific parameters
- `Policy` Pydantic model with SHA-256 version hash
- `Policy.for_category(cat)` returns effective policy with category overrides
- All monetary values in EUR (suffix `_eur`)

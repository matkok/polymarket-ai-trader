# Contributing to Agent-Trader

## Getting Started

1. Fork the repository and clone your fork
2. Copy `.env.example` to `.env` and fill in your API keys
3. Set up the conda environment:
   ```bash
   # LLM trader
   conda create -n agent-trader python=3.12 -y
   conda activate agent-trader
   pip install -e ".[dev]"

   # Structured trader
   conda create -n agent-trader-structured python=3.12 -y
   conda activate agent-trader-structured
   cd structured && pip install -e ".[dev]"
   ```
4. Run tests to verify your setup:
   ```bash
   conda run -n agent-trader python -m pytest tests/ -v
   conda run -n agent-trader-structured python -m pytest structured/tests/ -v
   ```

## Code Style

- `from __future__ import annotations` at the top of every Python file
- Type hints on all function signatures
- `structlog` for logging (`logger = structlog.get_logger(__name__)`)
- Pydantic v2 for config/settings models
- `dataclasses` for internal data structures
- SQLAlchemy 2.0 `Mapped`/`mapped_column` pattern
- Async everywhere (all DB and HTTP operations)
- All monetary values in EUR with `_eur` suffix

## Testing

- All tests must pass before submitting a PR
- Use `pytest` with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- Group tests in classes with `# ---- Section ---` separators
- Use `MagicMock(spec=Model)` for SQLAlchemy model mocks (never `Model.__new__()`)
- Use `AsyncMock` for repository and client mocks

## Pull Requests

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if behavior changes
- Ensure all existing tests pass

## Project Structure

- `src/` — LLM-based trader (general prediction markets)
- `structured/` — Deterministic structured trader (crypto, weather, macro, earnings)
- `tests/` — LLM trader tests
- `structured/tests/` — Structured trader tests
- `policy.yaml` / `structured/policy.yaml` — Risk and sizing parameters

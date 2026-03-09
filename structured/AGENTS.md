# AGENTS.md — AI Agent Guide for agent-trader-structured

Instructions for AI coding agents working on this codebase.

## Project Identity

Deterministic, source-driven paper trading system for Polymarket weather, macro,
crypto, and earnings markets. No LLM-in-the-loop — probability comes from CDF
models over authoritative data sources (NWS, BLS, FRED, Binance, EDGAR).

## Build & Test

```bash
# Conda environment
conda run -n agent-trader-structured python -m pytest tests/ -v

# Single file
conda run -n agent-trader-structured python -m pytest tests/test_weather_engine.py -v

# Docker
docker compose up --build -d
docker compose logs -f app
```

614 tests, all must pass. `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.

## Key Entrypoints

- `src/app/main.py` — wires all components, starts APScheduler (9 jobs)
- `src/app/scheduler.py` — `StructuredTradingEngine` orchestrator
- `src/contracts/registry.py` — `classify_markets_batch()` dispatches to parsers
- `src/trading/weather_pipeline.py` — weather trading loop (2 min cycle)
- `src/trading/macro_pipeline.py` — macro trading loop (5 min cycle)
- `src/trading/crypto_pipeline.py` — crypto trading loop (1 min cycle)
- `src/trading/earnings_pipeline.py` — earnings trading loop (10 min cycle)
- `src/portfolio/lifecycle.py` — 8-rule position lifecycle (deterministic + engine-driven)
- `src/evaluation/calibration.py` — Brier score evaluation (daily)
- `src/evaluation/kill_switch.py` — auto-disable on poor calibration

## Code Conventions (Must Follow)

- `from __future__ import annotations` at top of every `.py` file
- Type hints on all function signatures
- `structlog` for logging: `logger = structlog.get_logger(__name__)`
- `dataclasses` for internal data structures (not Pydantic)
- `Pydantic v2` for config/settings/policy models only
- SQLAlchemy 2.0 `Mapped`/`mapped_column` pattern
- Async everywhere — all DB and HTTP operations are async
- All monetary values in EUR with `_eur` suffix

## Test Conventions

- Group tests in classes with `# ---- Section ---` separators
- `MagicMock(spec=Model)` for SQLAlchemy models (never `Model.__new__()`)
- `AsyncMock` for repository/client mocks
- Settings tests: `Settings(_env_file=None)` with `patch.dict(os.environ, {}, clear=True)`

## Adding a New Category (e.g., "sports", "commodities")

1. Create `src/contracts/<category>.py` with `<Category>ContractSpec` and `<Category>Parser`
2. Register parser in `src/contracts/registry.py` (order matters — first match wins)
3. Create `src/sources/<source>.py` implementing `SourceAdapter` ABC
4. Create `src/engines/<category>.py` implementing `PricingEngine` ABC
5. Create `src/trading/<category>_pipeline.py` following `WeatherPipeline` pattern
6. Add category to `policy.yaml` under `categories:` with bankroll and engine_params
7. Wire in `src/app/main.py` and add scheduler job in `src/app/scheduler.py`
8. Add `run_<category>_cycle()` to `StructuredTradingEngine` with kill switch check
9. Add tests in `tests/test_<category>_engine.py`

## Contract Parser Pattern

Parsers follow a two-stage gate:

```python
class FooParser(ContractParser):
    def can_parse(self, question, rules_text) -> bool:
        """Fast keyword check — must use word-boundary regex for short words."""
        ...

    def parse(self, question, rules_text) -> ParseResult:
        """Full structural parse — returns ContractSpec or reject reason."""
        # 1. Check negative context (sports/geopolitics blocklists)
        # 2. Try regex patterns in priority order
        # 3. No match → ParseResult(matched=False, reject_reason="keyword_only_no_structure")
        ...
```

**Critical:** Never use substring matching (`"rain" in text`) for short keywords.
Always use `\b` word-boundary regex to avoid false positives (e.g., "Ukraine"
contains "rain", "window" contains "wind").

## Source Adapter Pattern

```python
class FooAdapter(SourceAdapter):
    async def fetch(self, spec: ContractSpec) -> FetchResult:
        """Fetch data from external API → FetchResult with values dict."""
        ...

    async def health_check(self) -> bool:
        ...
```

`FetchResult.to_observation_dict()` serializes for DB persistence.

## Pricing Engine Pattern

```python
class FooEngine(PricingEngine):
    def compute(self, spec: ContractSpec, observation: dict) -> PriceEstimate:
        """Source data → CDF → p_yes with confidence."""
        ...
```

All `p_yes` values clamped to `[0.01, 0.99]`. Confidence decays with lead time.

## Pipeline Pattern

See `src/trading/weather_pipeline.py` for the canonical loop:

```
get_markets_by_category → reconstruct spec → resolve dates → fetch source →
compute p_yes → compare to market → size → risk check → paper execute → persist
```

## Database

15 tables in Postgres 16 (port 5433). Key structured tables:

- `category_assignments` — parser results (parsed/rejected + spec JSON)
- `source_observations` — raw data from NWS/AWC/BLS/Binance/EDGAR
- `engine_prices` — computed p_yes + confidence + model details
- `calibration_stats` — Brier score per category/engine/date
- `category_pnl_daily` — daily realized + unrealized PnL per category

FK ordering: always upsert markets before inserting snapshots or assignments.

## Policy

`policy.yaml` has global defaults + category overrides. Access via:

```python
policy = load_policy("policy.yaml")
weather_policy = policy.for_category("weather")  # merged CategoryPolicy
```

Four categories: weather (10k EUR), macro (10k EUR), crypto (10k EUR),
earnings (10k EUR). Total bankroll: 40k EUR.

Category-specific `engine_params` (e.g., `forecast_horizon_hours`) are accessed
from `policy.categories["weather"].engine_params`.

## Position Lifecycle + Entry Guards (S7)

### Entry Guards
Every pipeline calls `repo.get_position(market_id)` before sizing. Skips if:
- Open position already exists for that market
- Closed position within `reentry_cooldown_hours` (6h default, per-category)

### Lifecycle Rules (8 total)
`PositionLifecycle` has two evaluation methods:

**`evaluate()` — deterministic safety (4 rules):**
1. No market snapshot → CLOSE
2. Approaching resolution → CLOSE
3. Mid moved against > 2× edge threshold → CLOSE (stop loss)
4. Unrealized PnL < -3% → REDUCE (partial stop)

**`evaluate_with_engine()` — adds engine-driven rules (4 more):**
5. Min hold time gate (30 min) → skip engine exits
6. Edge flip with cost buffer → CLOSE
7. Take profit (edge < band + cost) → CLOSE
8. Confidence collapse (< `min_confidence_hard`) → CLOSE

### Review Loop
- `review_open_positions()` every 5 min
- Per-category policy via `Policy.for_category()`
- Engine staleness check: `engine_stale_hours` (2h default)

## Common Pitfalls

- **Don't use `Model.__new__()`** for SQLAlchemy objects in tests — use `MagicMock(spec=Model)`
- **Don't forget negative filters** when adding weather keywords — sports teams
  have weather-word names (Carolina Hurricanes, Miami Heat, OKC Thunder)
- **Don't add generic fallbacks** to parsers — keyword-only matches without
  structural regex are false positives
- **Always test with Docker** after adding dependencies — `pyproject.toml` must
  match what's installed in conda
- **Circular imports** — use `TYPE_CHECKING` guard for pipeline imports in scheduler
- **Kill switch check** — every `run_<category>_cycle()` must check
  `self._kill_switch.is_enabled(category)` before running the pipeline

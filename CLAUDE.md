# CLAUDE.md — Project conventions for agent-trader

## Build and test

```bash
# Run tests (conda env)
conda run -n agent-trader python -m pytest tests/ -v

# Run single test file
conda run -n agent-trader python -m pytest tests/test_scheduler.py -v

# Docker build and run
docker compose up --build -d
docker compose logs -f app
docker compose down
```

## Project overview

Polymarket paper trading system. Milestones 1-6 complete (582 tests). Running 24/7 on production server. See SPECS.md for the full plan.

Key entrypoint: `src/app/main.py` — wires everything together and starts APScheduler.
Core orchestrator: `src/app/scheduler.py` — `TradingEngine` class drives ingestion, signals, scanning, review, scoring.

## Code conventions

- `from __future__ import annotations` at the top of every file
- Type hints on all function signatures
- structlog for logging (module-level `logger = structlog.get_logger(__name__)`)
- Pydantic v2 models for validation (BaseModel, not dataclass for config)
- dataclasses for internal data structures (SizingInput, CandidateScore, etc.)
- SQLAlchemy 2.0 Mapped/mapped_column pattern for ORM models
- Async everywhere — all DB ops and HTTP calls are async

## Test conventions

- pytest with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- Test classes grouped by module with `# ---- Section ---` separators
- Use `MagicMock(spec=Model)` for SQLAlchemy models in tests (never `Model.__new__()`)
- Use `AsyncMock` for repository and client mocks
- Use `MagicMock(spec=Position)` for Position objects in tests
- Settings tests use `Settings(_env_file=None)` with `patch.dict(os.environ, {}, clear=True)`

## Database

- Postgres 16 via Docker (service name: `postgres`)
- M1 uses `Base.metadata.create_all` for bootstrap (no Alembic migrations yet)
- Repository pattern: `src/db/repository.py` has all CRUD methods
- FK ordering matters: always upsert markets before inserting snapshots

## Policy

- `policy.yaml` — all risk/sizing parameters
- `Policy` Pydantic model with SHA-256 version hash for reproducibility
- All monetary values in EUR (suffix `_eur`)
- Bankroll: 10,000 EUR

## Important patterns

- Markets flow: Gamma API -> bulk_upsert_markets -> add_snapshot (FK order)
- Markets ingestion maps Gamma API `description` field to `rules_text` column
- Candidate scoring: filter (9 steps) -> score (5 weighted factors) -> sort -> truncate
- Rules text gate: markets with null/short rules_text (< 50 chars) are filtered pre-LLM (prevents ambiguity churn)
- Triage flow: CandidateSelector -> SignalCollector -> TriageScorer -> only should_panel=True -> LLM panel
- Signal sources: microstructure (always), evidence freshness (always), Google Trends (optional), Wikipedia (optional)
- Triage weights: odds_move 0.25, volume 0.15, evidence 0.20, trends 0.15, wiki 0.10, spread 0.15
- Guardrails: wide_spread (50% size), spread_widening (75% factor), social_only (block trade)
- Veto eligibility: only `rules_lawyer` and `escalation_anthropic` can contribute to rules-ambiguity veto quorum (VETO_ELIGIBLE_AGENTS)
- Veto weights: rules_lawyer=2.0 (can veto solo, quorum=2), all others=1.0
- Escalation dedup: when both rules_lawyer and escalation_anthropic vote, escalation's contribution is removed (same model family, correlated)
- Confidence gate: hard block (< 0.25 in aggregator), soft ramp (0.25–0.60 in sizing via effective_conf)
- Entry price filter: blocks entry_price > 0.90 unless massive edge override (edge >= 0.15 + full conf + low disagree)
- Position reentry cooldown: 6h after last order, requires same side + increasing signed edge + higher conf + lower disagree
- Position sizing: edge-based with effective_conf (not raw confidence), disagreement penalty, clamped to policy limits, adjusted by guardrails
- Risk checks: 5 independent constraints, all violations collected (not short-circuited)
- Paper fills: ask + slippage for buys, bid - slippage for sells, clamped [0.01, 0.99]
- Review triggers: odds_move, liquidity_drop, approaching_resolution, new_evidence, volume_surge, trends_spike, wiki_spike
- Budget tracker maps provider "gemini" to "google" cap via _PROVIDER_ALIASES
- Tiered model ladder: each provider has 2 tiers (best model → cheaper model as spend increases)
  - OpenAI: gpt-5-mini (to 2.00 EUR) → gpt-4.1-mini (to 4.00 EUR)
  - Anthropic: claude-sonnet-4 (to 2.50 EUR) → claude-haiku-4.5 (to 4.50 EUR)
  - Gemini: gemini-2.5-flash (to 1.00 EUR) → gemini-2.5-flash (to 3.00 EUR)
  - xAI: grok-3-fast (to 0.50 EUR) → grok-3-fast (to 2.50 EUR)
- `BudgetTracker.select_model(provider)` walks the ladder; returns None when all tiers exhausted
- MODEL_LADDER in schemas.py uses "google" key (not "gemini") to match _PROVIDER_ALIASES
- Escalation_anthropic uses Sonnet (not Opus) — auto-downgrades to Haiku via ladder
- Total daily cap: 14.00 EUR (sum of Tier B caps); actual spend expected ~6-8 EUR
- Gemini max_output_tokens=4096 (was 1024, caused JSON truncation)
- Paper executor: 0% fee for standard markets (2% for 15-min crypto) + 50bps slippage


# AGENTS.md — AI Agent Guide for agent-trader

Instructions for AI coding agents working on this codebase.

## Project Identity

Polymarket multi-LLM paper trading system. Scans ~29,000 active prediction
markets, collects real-world signals, runs a role-based LLM panel (OpenAI,
Anthropic, Google, xAI) for probability forecasting, and executes simulated
trades with deterministic risk management.

Currency: EUR. Bankroll: 10,000 EUR. Daily LLM budget: ~6-8 EUR (14 EUR hard cap).

## Build & Test

```bash
# Conda environment
conda run -n agent-trader python -m pytest tests/ -v

# Single file
conda run -n agent-trader python -m pytest tests/test_scheduler.py -v

# Docker
docker compose up --build -d
docker compose logs -f app
```

591 tests, all must pass. `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.

## Key Entrypoints

- `src/app/main.py` — wires all components, starts APScheduler (7 jobs)
- `src/app/scheduler.py` — `TradingEngine` orchestrator (1,694 lines, 20 methods)
- `src/app/candidate_selector.py` — deterministic market ranking (9-step filter + 5-factor score)
- `src/llm/panel.py` — multi-model panel orchestrator
- `src/aggregation/aggregator.py` — deterministic consensus + vetoes
- `src/signals/triage.py` — deterministic triage scorer + guardrails
- `src/evaluation/scorer.py` — daily model scoring orchestrator

## Scheduler Jobs

| Job | Interval | Description |
|-----|----------|-------------|
| `ingest_markets` | 15 min | Fetch ~29k markets from Gamma API (batched upserts) |
| `ingest_evidence` | 20 min | RSS feeds + full-text enrichment |
| `collect_signals` | 30 min | Microstructure, trends, wiki signals |
| `candidate_scan` | 30 min | Selection → triage → panel → execution |
| `position_review` | 15 min | Lifecycle management for open positions |
| `daily_scoring` | 00:30 UTC | Model evaluation on resolved markets |
| `daily_reset` | 00:00 UTC | Reset daily tracking + budget |

## Code Conventions (Must Follow)

- `from __future__ import annotations` at top of every `.py` file
- Type hints on all function signatures
- `structlog` for logging: `logger = structlog.get_logger(__name__)`
- `Pydantic v2` for config/settings/policy models and API schemas
- `dataclasses` for internal data structures (SizingInput, CandidateScore, etc.)
- SQLAlchemy 2.0 `Mapped`/`mapped_column` pattern
- Async everywhere — all DB and HTTP operations are async
- All monetary values in EUR with `_eur` suffix

## Test Conventions

- Group tests in classes with `# ---- Section ---` separators
- `MagicMock(spec=Model)` for SQLAlchemy models (never `Model.__new__()`)
- `MagicMock(spec=Position)` for Position objects
- `AsyncMock` for repository/client mocks
- Settings tests: `Settings(_env_file=None)` with `patch.dict(os.environ, {}, clear=True)`

## Data Flow

```
Gamma API → Market Ingestion (bulk upsert, dedup) → DB
RSS Feeds → Evidence Ingestion → DB
                                  ↓
                        Candidate Selector (9-step filter + 5-factor score)
                                  ↓
                        Signal Collector (microstructure, trends, wiki)
                                  ↓
                        Triage Scorer (weighted score + guardrails)
                                  ↓  only should_panel=True
                        Panel Cooldown Filter (skip if paneled < 8h ago)
                                  ↓
                        Packet Builder + Evidence Linker
                                  ↓
                        LLM Panel (4 agents, tiered model ladder) → Aggregator
                                  ↓
                        Position Sizing (+ guardrail adjustments)
                                  ↓
                        Risk Manager (5 constraints)
                                  ↓
                        Paper Executor → DB (orders, fills, positions)
```

## LLM Panel Agents

**Default panel (every cycle):**

| Agent ID | Provider | Role |
|----------|----------|------|
| rules_lawyer | Anthropic | Rules ambiguity (veto-eligible, weight 2.0) |
| probabilist | OpenAI | Calibrated probability estimation |
| sanity_checker | Google | Contrarian red-teaming |
| x_signals | xAI | Social sentiment (conditional) |

**Escalation agents (triggered by conditions):**

| Agent ID | Provider | Trigger |
|----------|----------|---------|
| escalation_anthropic | Anthropic | Rules ambiguity quorum |
| escalation_openai | OpenAI | High disagreement or fast odds move |
| escalation_google | Google | High stakes positions |

## Key Patterns

### Panel Cooldown
Markets skip re-paneling for `panel_cooldown_hours` (8h) after last LLM
evaluation. Uses `repo.get_recently_paneled_market_ids()` querying
`model_runs` table. Ensures broad exploration of ~29k markets.

### Triage Gate
CandidateSelector picks top N → SignalCollector computes signals →
TriageScorer applies 6 weighted components + 3 guardrails → only
`should_panel=True` candidates proceed to LLM panel.

### Veto System
- Only `rules_lawyer` (weight 2.0) and `escalation_anthropic` are veto-eligible
- De-duplication: if both vote, escalation's contribution is removed (correlated)
- `rules_lawyer` can veto solo (weight 2.0 >= quorum 2)

### Confidence Gate
- Hard block: confidence < 0.25 → trade forbidden
- Soft ramp: 0.25-0.60 → effective_conf scales linearly 0→1
- Full confidence: >= 0.60 → effective_conf = 1.0

### Tiered Model Ladder
Each provider has 2 tiers. When Tier A budget is hit, auto-downgrade to
cheaper Tier B model. Only skip agent when all tiers exhausted.
`BudgetTracker.select_model(provider)` walks the ladder.

### Bulk Operations
- `bulk_upsert_markets(batch_size=500)` — batches to stay under PG's 32,767 bind-param limit
- `bulk_add_snapshots(batch_size=500)` — batched snapshot inserts
- Market deduplication before upsert (Gamma API returns duplicates across pages)

## Database

14 tables in Postgres 16 (port 5432):

**M1 (7):** markets, market_snapshots, decisions, orders, fills, positions, resolutions
**M2-M5 (5):** evidence_items, packets, model_runs, aggregations, model_scores_daily
**M6 (1):** signal_snapshots
**Online (1):** online_scores

FK ordering: always upsert markets before inserting snapshots.

## Policy

`policy.yaml` has all risk/sizing parameters. Key values:

- Bankroll: 10,000 EUR
- Edge threshold: 0.08 (8%)
- Panel cooldown: 8 hours
- Max candidates per cycle: 10 (yaml), 15 (code default)
- Max panel markets per day: 10
- Position reentry cooldown: 6 hours
- Daily LLM budget: 14 EUR hard cap

`Policy` model with SHA-256 version hash. Stored with every decision.

## Budget

Provider "gemini" maps to "google" cap via `_PROVIDER_ALIASES`.
`MODEL_LADDER` in `schemas.py` uses "google" key (not "gemini").

| Provider | Tier A | Cap A | Tier B | Cap B |
|----------|--------|-------|--------|-------|
| OpenAI | gpt-5-mini | 2.00 | gpt-4.1-mini | 4.00 |
| Anthropic | claude-sonnet-4 | 2.50 | claude-haiku-4.5 | 4.50 |
| Google | gemini-2.5-flash | 1.00 | gemini-2.5-flash | 3.00 |
| xAI | grok-3-fast | 0.50 | grok-3-fast | 2.50 |

## Common Pitfalls

- **Don't use `Model.__new__()`** for SQLAlchemy objects in tests — use `MagicMock(spec=Model)`
- **FK ordering** — always upsert markets before inserting snapshots (foreign key constraint)
- **Circular imports** — use `TYPE_CHECKING` guard for heavy imports in scheduler
- **Budget aliases** — provider "gemini" maps to "google" in budget tracking
- **Veto eligibility** — only `rules_lawyer` and `escalation_anthropic` can veto; don't add other agents
- **Dedup escalation** — when both rules_lawyer and escalation_anthropic vote, remove escalation's contribution
- **Gemini max_output_tokens** — must be 4096 (was 1024, caused JSON truncation)
- **Rules text gate** — markets with null/short rules_text (< 50 chars) are filtered pre-LLM

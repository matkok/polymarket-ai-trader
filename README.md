# Agent-Trader

Polymarket paper trading system with two independent trading strategies:

- **LLM Trader** (`src/`) — Multi-LLM panel (OpenAI, Anthropic, Google, xAI) for probability forecasting on general prediction markets. Collects real-world signals (RSS evidence, Google Trends, Wikipedia pageviews), runs role-based LLM agents for analysis, and trades with deterministic risk management.

- **Structured Trader** (`structured/`) — Source-driven, deterministic probability models for specific market categories (crypto, weather, macro, earnings). No LLMs — uses public data APIs (Coinbase, NWS, BLS, FRED) and statistical models to price contracts.

Both traders share the same execution engine (paper fills with slippage simulation), risk management framework, and database schema pattern, but differ in how they generate probability estimates.

Currency: EUR. Bankroll: 10,000 EUR per trader. Daily LLM budget (LLM trader only): ~6-8 EUR/day (14 EUR hard cap with tiered model ladder).

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Deterministic paper system (no LLM) | Done |
| M2 | Evidence ingestion and packet building | Done |
| M3 | Full LLM panel (OpenAI, Anthropic, Google, xAI) | Done |
| M4 | Open position review loop with triggers | Done |
| M5 | Model scoring and leaderboards | Done |
| M6 | Signal collection and triage scoring | Done |

591 tests passing. Running 24/7 on production server.

### Capabilities

- Live market ingestion from Polymarket Gamma API (~29,000 active markets)
- RSS evidence ingestion with full-text enrichment and semantic linking
- Proactive xAI search for sparse-evidence candidates
- Signal collection: market microstructure (odds moves, volume ratios, spread), evidence freshness, Google Trends spikes, Wikipedia pageview spikes
- Deterministic triage scoring with guardrails (wide spread, spread widening, social-only evidence blocking)
- Role-based LLM panel (4 agents: rules lawyer, probabilist, sanity checker, X signals) with priority-based escalation and tiered model ladder (auto-downgrade to cheaper models as budget is consumed)
- Pre-entry rules_text quality gate: blocks markets with missing/weak resolution rules before LLM spend
- Two-layer aggregation: probability vote (60% weighted mean + 40% median) and rules ambiguity hard veto with eligible-agent quorum
- Veto eligibility restricted to rules_lawyer (weight 2.0, can veto solo) and escalation_anthropic, with de-duplication for correlated votes
- Tiered confidence gate: hard block (< 0.25) and soft ramp (0.25–0.60) replacing raw confidence in sizing
- Entry price filter: blocks trades at extreme probabilities (> 0.90) unless massive edge override
- Position reentry cooldown: 6-hour stability window requiring same side, increasing edge, higher confidence, lower disagreement
- Panel cooldown: markets skip re-paneling for 8 hours after last LLM evaluation, ensuring broad market exploration across cycles
- Candidate selection with 9-step filtering (including rules_text gate) and 5 weighted scoring factors
- Edge-based position sizing with effective confidence, disagreement penalties, and triage guardrail adjustments
- Portfolio risk manager (5 independent constraint checks)
- Position lifecycle management (hold/reduce/close) with panel re-analysis triggers
- Signal-based review triggers (volume surge, trends spike, wiki spike)
- Daily model scoring with Brier scores, calibration, PnL attribution, and veto value
- Paper executor with slippage and fee simulation (0% fee for standard markets, 2% for 15-min crypto)
- APScheduler loop (ingestion 15min, evidence 20min, signals 30min, scan 30min, review 15min, scoring daily, reset midnight)
- Full structured logging via structlog

## Local development

```bash
# Clone and start locally
cp .env.example .env    # edit API keys
docker compose up --build -d
docker compose logs -f app
docker compose down

# Create conda environment
conda create -n agent-trader python=3.12 -y
conda activate agent-trader

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run the app locally (requires Postgres at localhost:5432)
python -m src.app.main
```

## Project structure

```
agent-trader/
  SPECS.md              # Full project specification (all milestones)
  charter.md            # Trading constitution (injected into LLM prompts)
  policy.yaml           # Deterministic risk/sizing/triage parameters
  docker-compose.yml    # Postgres + app
  Dockerfile
  pyproject.toml
  alembic.ini
  src/
    config/
      settings.py       # Pydantic Settings from env vars / .env
      policy.py         # Policy model, YAML loader, version hashing
      budget.py         # Daily LLM budget tracker
    db/
      models.py         # SQLAlchemy ORM (14 tables)
      session.py        # Async engine and session factory
      repository.py     # CRUD helpers with upserts
      migrations/       # Alembic setup
    polymarket/
      schemas.py        # Pydantic models for Gamma/CLOB responses
      gamma_client.py   # Async market discovery client
      clob_client.py    # Async price/orderbook client
      ws_watcher.py     # WebSocket price subscriber
    app/
      candidate_selector.py  # Deterministic market ranking
      scheduler.py           # TradingEngine orchestrator
      main.py                # Entrypoint, wiring, APScheduler
    evidence/
      schemas.py        # Evidence dataclasses
      rss_ingestor.py   # RSS feed ingestion
      xai_search.py     # xAI Grok live search
      linker.py         # Semantic evidence-to-market linking
      embedder.py       # OpenAI embeddings for similarity
      fulltext.py       # Full-text enrichment for short excerpts
      quality.py        # Evidence quality scoring
    packets/
      builder.py        # Evidence packet builder
      schemas.py        # Packet Pydantic models
    llm/
      base.py           # Base LLM client interface
      openai_client.py  # OpenAI provider
      anthropic_client.py # Anthropic provider
      gemini_client.py  # Google Gemini provider
      xai_client.py     # xAI Grok provider
      panel.py          # Multi-model panel orchestrator
      prompt_builder.py # Prompt construction
      schemas.py        # ModelProposal, PanelResult
    aggregation/
      aggregator.py     # Deterministic consensus + vetoes
    signals/
      schemas.py        # Signal dataclasses (7 types)
      microstructure.py # Odds moves, volume ratios, spreads
      trends.py         # Google Trends spike detection
      wikipedia.py      # Wikipedia pageview spike detection
      collector.py      # Signal collection orchestrator
      triage.py         # Deterministic triage scorer + guardrails
    portfolio/
      sizing.py         # Edge-based position sizing
      risk_manager.py   # 5 constraint checks
      lifecycle.py      # Hold/reduce/close decisions
    execution/
      paper_executor.py # Simulated fills with slippage
      fills.py          # PnL calculations
    evaluation/
      metrics.py        # Brier score, log loss, calibration
      attribution.py    # PnL attribution (support, dissent, veto, sizing error)
      scorer.py         # Daily model scoring orchestrator
      reports.py        # Daily PnL/exposure reports
  tests/                # 591 tests
    test_config.py
    test_polymarket.py
    test_candidate_selector.py
    test_portfolio.py
    test_execution.py
    test_scheduler.py
    test_signals.py
    test_main.py
    test_reports.py
    test_llm.py
    test_aggregation.py
    test_evidence.py
    test_xai_search.py
    test_packets.py
    test_budget.py
    test_attribution.py
    test_scorer.py
    test_metrics.py
    test_online_scorer.py
    test_connectivity.py
```

## Architecture

```
Gamma API --> Market Ingestion --> DB (markets, snapshots)
                                     |
RSS Feeds --> Evidence Ingestion --> DB (evidence_items)
                                     |
                                     v
                              Candidate Selector (filter + score)
                                     |
                                     v
                         Signal Collector (microstructure, trends, wiki)
                                     |
                                     v
                         Triage Scorer (weighted score + guardrails)
                                     |  only candidates with should_panel=True
                                     v
                         Packet Builder + Evidence Linker
                                     |
                                     v
                         LLM Panel (4 providers, tiered model ladder) --> Aggregator
                                     |
                                     v
                              Position Sizing (+ guardrail adjustments)
                                     |
                                     v
                              Risk Manager (5 constraints)
                                     |
                                     v
                              Paper Executor --> DB (orders, fills, positions)
                                     |
                                     v
                              Position Review (lifecycle + signal triggers)
                                     |
                                     v
                              Daily Model Scoring + Reports
```

Without LLM API keys, the panel is bypassed and a synthetic edge from candidate scores is used. Without signal components, triage is skipped (backward compatible).

## Key dependencies

- Python 3.12+
- PostgreSQL 16 (via Docker)
- SQLAlchemy 2.0 (async) + asyncpg
- Alembic (migrations)
- Pydantic v2 + pydantic-settings
- httpx (async HTTP)
- websockets (CLOB price feeds)
- APScheduler (job scheduling)
- structlog (structured logging)

## Configuration

All config is via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Postgres connection string | `postgresql+asyncpg://...localhost:5432/agent_trader` |
| `GAMMA_API_BASE_URL` | Polymarket Gamma API | `https://gamma-api.polymarket.com` |
| `CLOB_API_BASE_URL` | Polymarket CLOB API | `https://clob.polymarket.com` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `OPENAI_API_KEY` | OpenAI key | empty |
| `ANTHROPIC_API_KEY` | Anthropic key | empty |
| `GOOGLE_API_KEY` | Google key | empty |
| `XAI_API_KEY` | xAI key | empty |
| `GOOGLE_TRENDS_ENABLED` | Enable Google Trends signals | `false` |
| `GOOGLE_TRENDS_TRAILING_DAYS` | Trends trailing window | `30` |
| `WIKIPEDIA_ENABLED` | Enable Wikipedia pageview signals | `false` |
| `WIKIPEDIA_TRAILING_DAYS` | Wikipedia trailing window | `30` |

Risk and triage parameters are in `policy.yaml`. The trading constitution is in `charter.md`.

## Known issues

- **Gemini parse failures**: Gemini Flash sometimes produces malformed JSON (missing commas, truncated strings). Non-critical — the panel works with OpenAI + Anthropic + xAI when Gemini fails. Retry logic handles transient failures.
- **Anthropic markdown fences**: Claude Haiku occasionally wraps JSON in ```json``` fences. The parser strips these, but sometimes the response is also truncated. Retry + escalation handles this.

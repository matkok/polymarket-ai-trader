# Polymarket Multi LLM Trading Panel (Paper Trading) Project Spec

## 1. Purpose

Build a paper trading system for Polymarket that:
1. Scans markets and detects candidate opportunities using deterministic filters.
2. Builds an evidence packet per candidate market from reputable sources and Grok live search for X signals.
3. Runs a panel of LLMs (OpenAI, Claude, Gemini, Grok) to independently forecast probability and propose trade intent using a strict JSON schema.
4. Aggregates model outputs deterministically, applies risk and portfolio policy deterministically, and executes simulated trades.
5. Logs every input, output, decision, and fill so we can score each model and the final policy over weeks and later re weight or disable models.
6. Enforces daily budget caps per provider and hard risk constraints.

Non goals:
* Live trading is not in scope for v1.
* No browser automation. All ingestion is via APIs, RSS, or xAI live search.

## 2. High level architecture

Services (can start as one process, later split):
1. Market Ingestion
2. Market Watcher
3. Evidence Ingestor
4. Packet Builder
5. LLM Panel Orchestrator
6. Aggregator
7. Portfolio and Risk Manager
8. Paper Executor
9. Evaluator and Reporting
10. Dashboard (optional)

Core loops:
1. Candidate loop: find new markets to consider.
2. Open position loop: review open trades and decide hold, reduce, close, or add.

Decision principle:
* Models propose. Code decides and executes.

## 3. Polymarket interfaces

Use Polymarket global APIs:
* Gamma API for market discovery and metadata.
* CLOB REST and WebSocket for prices and order books.

Paper trading only requires read access to Gamma and CLOB.

## 4. Trading Charter and Policy

Two files that are versioned in git and stored with every decision.

### 4.1 Charter (charter.md)
This is the constitution injected into every LLM call (cached when available).

Charter must include:
1. Evidence discipline
   * Cite only URLs in the provided packet.
   * Label information as fact, sentiment, or speculation.
2. Rules discipline
   * If resolution is ambiguous, raise ambiguity score and recommend no trade.
3. Risk discipline
   * No martingale.
   * Size scales with edge and confidence and liquidity and shrinks with disagreement.
4. Execution discipline
   * LLM never instructs direct order placement. Only structured proposal output.
5. Exit discipline
   * Provide clear exit triggers tied to odds change, evidence invalidation, and time to resolution.

### 4.2 Policy (policy.yaml)
Deterministic portfolio and risk limits. Example fields:
* bankroll_eur: 10000
* cash_reserve_target_frac
* max_total_exposure_frac
* max_exposure_per_market_frac
* max_open_positions
* max_daily_loss_frac
* min_liquidity_eur
* edge_threshold (0.08)
* min_confidence_hard (0.25)
* min_confidence_full (0.60)
* max_entry_price (0.90)
* position_reentry_cooldown_hours (6)
* disagreement_block_threshold
* disagreement_size_penalty_start
* odds_move_recheck_threshold
* new_evidence_recheck_window_minutes
* min_rules_text_length (50)
* panel_cooldown_hours (8)
* max_candidates_per_cycle (15 default, 10 in yaml)
* max_panel_markets_per_day

Policy must be stored with a policy_version hash.

## 5. Data flow

1. Market Ingestion polls Gamma for all ~29,000 active markets (paginated, 300 pages × 100), deduplicates, and upserts to DB in batches of 500.
2. Candidate Selector ranks markets using deterministic rules:
   * liquidity and volume thresholds
   * rules_text quality gate (min 50 chars, blocks null/weak rules pre-LLM)
   * time to resolution constraints
   * odds movement and volatility
   * novelty and category diversification
3. Market Watcher subscribes to CLOB WebSocket for the current shortlist.
4. Evidence Ingestor collects:
   * RSS and reputable sources
   * xAI Grok live search for X and web evidence when escalation is triggered
5. Packet Builder composes Packet JSON for each candidate market.
6. LLM Panel Orchestrator calls all models on the same packet and returns strict JSON.
7. Aggregator computes consensus and vetoes deterministically.
8. Portfolio and Risk Manager determines action and size deterministically.
9. Paper Executor simulates fills using bid ask and slippage.
10. Evaluator scores models and policy daily and weekly.

## 6. Evidence ingestion and packet building

### 6.1 Evidence sources
Start conservative:
* RSS feeds from reputable outlets.
* Official sources relevant to market categories (government, company IR, sports leagues, courts).
Optional escalation:
* Grok live search to retrieve X and web sources for specific markets.

### 6.2 Evidence storage
Store raw fetched content and extracted text.
Deduplicate by hash of normalized text and URL.

### 6.3 Packet constraints
Packet is the only information LLMs can use.
Packet must be bounded to control cost and ensure reproducibility.

Packet max items:
* max_evidence_items_per_packet: 10 (configurable)
Packet max text:
* each evidence item includes a short excerpt plus link to full text in DB.

Packet must include:
* market question
* market rules and resolution criteria
* current odds and implied probability
* bid ask spread and liquidity snapshot
* evidence list with URLs, timestamps, excerpts
* open positions summary for that market (if any)

## 7. LLM panel protocol

Models used in v1:
* OpenAI model (example GPT 5 mini)
* Anthropic model (example Claude Haiku class)
* Gemini model (example Gemini Flash class)
* xAI Grok (Grok 3 fast), plus live search escalation as needed

### 7.1 Strict JSON output schema (ModelProposal)
Every model must output valid JSON matching this schema. Invalid output is retried once, then skipped.

Metadata fields (injected by orchestrator, never set by LLM):
* model_id: string (agent_id of the proposing agent)
* run_id: string (unique UUID per call)
* market_id: string
* ts_utc: ISO string

Probability vote (soft vote):
* p_true: float in [0,1]
* confidence: float in [0,1]
* direction: "BUY_YES" | "BUY_NO"

Ambiguity vote (hard vote — two separate dimensions):
* rules_ambiguity: float in [0,1] (veto-able via weighted quorum)
* evidence_ambiguity: float in [0,1] (confidence penalty only, never vetoes)

Sizing and horizon:
* recommended_max_exposure_frac: float in [0,1]
* hold_horizon_hours: float (>= 0)

Analysis:
* thesis: string
* key_risks: array of short strings
* evidence: array of objects:
  * url: string (must be in packet)
  * claim: string
  * strength: float in [0,1]
* exit_triggers: array of short strings
* notes: string (short)

Hard rules:
* No URLs outside the packet.
* If rules_ambiguity is high, the aggregator may veto the trade via weighted quorum.

### 7.2 Two round option
Round 1: independent proposals.
Round 2: critique and revise using only:
* other models p_true, confidence, ambiguity_score, and top thesis bullets.

Round 2 is optional in v1. Keep it off until v1 is stable.

### 7.3 Role-Based Agent Panel

The panel uses role-based agents instead of per-provider tiers. Each agent has a specialized role with a customized system prompt preamble.

**Default panel (4 agents, run every cycle):**

| Agent ID | Provider | Model | Role | Always On |
|----------|----------|-------|------|-----------|
| rules_lawyer | Anthropic | claude-sonnet-4-20250514 | rules | Yes |
| probabilist | OpenAI | gpt-5-mini | probability | Yes |
| sanity_checker | Google | gemini-2.5-flash | sanity | Yes |
| x_signals | xAI | grok-3-fast | signals | No (conditional) |

**Escalation agents (called only when escalation is triggered):**

| Agent ID | Provider | Model | Role |
|----------|----------|-------|------|
| escalation_openai | OpenAI | gpt-5.2 | probability |
| escalation_anthropic | Anthropic | claude-sonnet-4-20250514 | rules |
| escalation_google | Google | gemini-2.5-pro | arbiter |

**Role preambles** (injected into system prompt):
* **rules** — Rules Lawyer: focuses on resolution criteria ambiguity
* **probability** — Market Probabilist: calibrated probability estimation
* **sanity** — Sanity Checker: contrarian perspective, red-teams the thesis
* **signals** — X Signals Analyst: social sentiment and narrative tracking
* **arbiter** — Arbiter: called for disputes, weighs all evidence independently

**Conditional triggering (x_signals agent):**
* Odds move in last 6h >= odds_move_recheck_threshold
* No credible evidence in last 6h
* Position is in top-3 by exposure

Default panel handles ~90% of calls. Escalation agents are gated by rules in section 7.4.

### 7.4 Escalation Rules

Escalation uses priority-based single-agent selection. First matching condition wins:

1. **Rules ambiguity quorum** (veto_score >= policy.veto_quorum):
   * Escalation agent: escalation_anthropic (Claude Sonnet 4, rules role; auto-downgrades to Haiku via model ladder)
   * Trigger: RULES_AMBIGUITY

2. **Partial rules ambiguity** (0 < veto_score < policy.veto_quorum):
   * Escalation agent: escalation_anthropic
   * Trigger: RULES_AMBIGUITY (confidence downgraded during aggregation)

3. **High disagreement** (std dev of p_true >= 0.12):
   * Escalation agent: escalation_openai (GPT-5.2, probability role)
   * Trigger: DISAGREEMENT

4. **High stakes** (top-3 position exposure OR proposed size > 5% of bankroll):
   * Escalation agent: escalation_google (Gemini Pro, arbiter role)
   * Trigger: HIGH_STAKES

5. **Fast odds move** (odds_move >= policy.odds_move_recheck_threshold):
   * Escalation agent: escalation_openai
   * Trigger: FAST_ODDS_MOVE

Escalation proposals are added to the existing proposals and re-aggregated.

### 7.5 Cost Control Strategy

* **Prompt caching:** Use provider-native prompt caching (Anthropic cache_control, OpenAI cached completions, Gemini context caching) for the charter + policy preamble. Expected savings: 50-90% on input tokens for repeated calls.
* **Per-day market limits:** max_panel_markets_per_day (default 10) caps the number of markets sent through the full panel per day.
* **Gated Grok Live Search:** Live search counts against daily_sources_cap_xai_search (default 10). Only triggered for top candidates and open positions with large odds moves.
* **Token budgets:** Hard per-call limits (max_input_tokens_per_call, max_output_tokens_per_call) prevent runaway costs.
* **Tiered model ladder:** Each provider has a two-tier ladder. When Tier A cap is hit, calls switch to a cheaper Tier B model instead of being skipped entirely. Agents only skip when all tiers are exhausted.

## 8. Aggregation

Aggregator is deterministic.

Inputs:
* Market snapshot (p_mkt, bid ask, liquidity)
* ModelProposal list
* Policy parameters

Outputs:
* p_consensus (mean and median)
* disagreement (std dev or MAD)
* veto flags and reasons
* trade_allowed boolean
* recommended_action (BUY_YES, BUY_NO, HOLD, REDUCE, CLOSE, DO_NOTHING)
* recommended_size_eur (pre clamp)
* attribution summary (supporting, dissenting, vetoing models)

### 8.1 Layer 1: Probability vote (soft)
* weight per model = confidence * reliability_weight (reliability starts at 1.0)
* weighted_mean = sum(p_true * weight) / sum(weights)
* median_p = median of all p_true values
* p_consensus = 0.6 * weighted_mean + 0.4 * median_p
* confidence = weighted mean of all confidences
* disagreement = stdev of p_true values (0.0 if single proposal)

### 8.2 Layer 2: Rules ambiguity vote (hard veto)
* Only veto-eligible agents contribute: `rules_lawyer` and `escalation_anthropic` (VETO_ELIGIBLE_AGENTS). Other agents' rules_ambiguity is ignored for veto scoring.
* For each eligible proposal with rules_ambiguity >= ambiguity_veto_threshold:
  * veto_score += VETO_WEIGHTS.get(model_id, 1.0) — rules_lawyer gets 2.0x weight (can veto solo)
* De-duplication: if both rules_lawyer and escalation_anthropic vote, escalation's contribution is removed (same model family, correlated evidence)
* If veto_score >= veto_quorum: veto = True, trade_allowed = False
* If 0 < veto_score < quorum: no veto, but confidence *= 0.5

### 8.3 Evidence ambiguity (confidence penalty only)
* Count models with evidence_ambiguity >= ambiguity_veto_threshold
* If any: confidence *= max(0.5, 1.0 - 0.15 * count)
* Never vetoes trade.

### 8.4 Confidence gate
* Hard block: if weighted confidence < min_confidence_hard (0.25), trade_allowed = False.
* Applied after evidence ambiguity penalty, before other trade_allowed checks.

### 8.5 Disagreement handling
If disagreement >= disagreement_block_threshold then trade_allowed = false.
Else apply size penalty starting at disagreement_size_penalty_start.

### 8.6 Model classification
* Consensus direction: "BUY_YES" if p_consensus > p_market else "BUY_NO"
* Models aligned with consensus direction → supporting_models
* Models opposed → dissenting_models
* Models that triggered veto → vetoing_models

## 9. Portfolio and Risk Manager

Deterministic capital allocation and lifecycle control.

Inputs:
* current portfolio state (cash, open positions, exposure)
* aggregator output for candidates
* triggers for open positions (odds move, new evidence, nearing resolution)
* policy.yaml

Outputs:
* list of Actions:
  * OPEN
  * ADD
  * REDUCE
  * CLOSE
  * HOLD
  * DO_NOTHING
Each action includes:
* market_id
* side
* size_eur
* limit_price_ref (bid, ask, mid)
* reason_json
* policy_constraints_hit

### 9.1 Position sizing (v1)
Let:
* edge = p_consensus - p_mkt for YES
* edge = p_mkt - p_consensus for NO

**Entry price filter:**
* entry_price = p_market for BUY_YES, (1 - p_market) for BUY_NO
* If entry_price > max_entry_price (0.90), block unless massive edge override (edge >= 0.15 AND confidence >= min_confidence_full AND disagreement <= disagreement_size_penalty_start)

If abs(edge) < edge_threshold (0.08) then no trade.

**Effective confidence (replaces raw confidence in formula):**
* If confidence < min_confidence_full (0.60): effective_conf = (confidence - min_confidence_hard) / (min_confidence_full - min_confidence_hard), clamped [0, 1]
* If confidence >= min_confidence_full: effective_conf = 1.0

Base size:
* size = bankroll_eur * base_risk_frac
Scaled:
* size *= abs(edge) / edge_scale
* size *= effective_conf  (not raw confidence — single confidence factor, no double penalty)
* size *= (1 - disagreement_penalty)

Clamp:
* per market max exposure
* total max exposure
* daily loss stop

### 9.2 Position reentry cooldown
When re-entering a previously-traded market within position_reentry_cooldown_hours (6h):
* Requires same consensus side as prior aggregation (blocks direction flips)
* Requires signed edge > 0 and increasing vs prior (edge strengthening)
* Requires higher confidence and lower disagreement than prior
* Uses get_latest_order_ts() — only advances on actual fills, not review-loop PnL refresh
* Fresh entries to never-traded markets are unaffected
* Aggregation_json stores p_market and consensus_side for cross-time comparison

### 9.3 Open position review
Triggers:
* odds moved beyond odds_move_recheck_threshold
* new evidence within window
* time to resolution crosses thresholds
* liquidity drops below min

Exit rules:
* close if thesis invalidated by evidence or if edge flips
* take profit if market price moves to within small band of p_consensus and confidence drops
* reduce if disagreement increases or ambiguity increases

All exit decisions are deterministic using updated aggregator outputs.

## 10. Paper execution model

Simulated fills:
* entry at best ask for buys, best bid for sells
* configurable slippage model
* configurable fees model
* partial fills optional in v2

Store:
* orders
* fills
* position updates
* realized and unrealized pnl

## 11. Budget and rate limiting

### 11.1 Daily budget caps (EUR)
All budgets are in EUR. Hard cap is 14.00 EUR/day; actual spend expected ~6-8 EUR thanks to tiered model ladder.

Provider caps (Tier B / maximum):
* daily_eur_cap_openai: 4.00
* daily_eur_cap_anthropic: 4.50
* daily_eur_cap_google: 3.00
* daily_eur_cap_xai: 2.50
* daily_eur_cap_total: 14.00 (hard ceiling)

Note: Provider "gemini" maps to "google" budget cap via _PROVIDER_ALIASES.

**Tiered model ladder:** Instead of skipping agents when a budget cap is hit, each provider walks a two-tier ladder of progressively cheaper models:

| Provider | Tier A (best) | Cap A | Tier B (cheaper) | Cap B |
|----------|---------------|-------|------------------|-------|
| OpenAI | gpt-5-mini | 2.00 | gpt-4.1-mini | 4.00 |
| Anthropic | claude-sonnet-4-20250514 | 2.50 | claude-haiku-4-5-20251001 | 4.50 |
| Google | gemini-2.5-flash | 1.00 | gemini-2.5-flash | 3.00 |
| xAI | grok-3-fast | 0.50 | grok-3-fast | 2.50 |

Logic: `BudgetTracker.select_model(provider)` checks cumulative spend → returns Tier A model if under Cap A, Tier B if under Cap B, None if exhausted.

Budget enforcement:
* Before each model call, `select_model()` picks the appropriate tier model.
* If all tiers exhausted (select_model returns None), skip that agent and log `panel_budget_exhausted`.
* If tier switched, log `panel_model_tiered` with default and selected model.
* Cost is estimated for the *selected* model (not the agent's default).
* Escalation and arbiter calls consume the same provider budget — no separate pool.

### 11.2 Token limits per call
Set hard caps:
* max_input_tokens_per_call
* max_output_tokens_per_call

Packet pruning is deterministic:
* keep newest evidence
* keep most relevant evidence by keyword score
* keep at least one official source if present

### 11.3 xAI live search budget
Define:
* daily_sources_cap_xai_search
* max_sources_per_market

Only enable live search on:
* top K candidates per day
* open positions on large odds moves
* markets with insufficient evidence from RSS

## 12. Logging and reproducibility

Every run must be reproducible given stored artifacts:
* packet_json and packet_hash
* charter_version hash
* policy_version hash
* prompt_version id
* model_id and model parameters
* raw model response
* parsed proposal JSON
* aggregator output JSON
* decision JSON
* market snapshots at decision time

## 13. Database schema (Postgres, 14 tables)

### M1 tables (implemented)

1. markets
   * market_id PK (string)
   * question
   * rules_text nullable
   * category nullable
   * resolution_time_utc nullable
   * status (default "active")
   * created_ts_utc (server default now)
   * updated_ts_utc (server default now, onupdate now)
2. market_snapshots
   * snapshot_id PK (auto)
   * market_id FK -> markets
   * ts_utc
   * best_bid nullable
   * best_ask nullable
   * mid nullable
   * liquidity nullable
   * volume nullable
   * INDEX (market_id, ts_utc)
3. decisions
   * decision_id PK (auto)
   * market_id FK -> markets
   * ts_utc
   * action
   * size_eur
   * reason_json (JSON, nullable)
   * policy_version nullable
   * INDEX (ts_utc)
4. orders
   * order_id PK (auto)
   * decision_id FK -> decisions
   * market_id FK -> markets
   * side
   * size_eur
   * limit_price_ref nullable
   * status (default "pending")
   * created_ts_utc (server default now)
5. fills
   * fill_id PK (auto)
   * order_id FK -> orders
   * ts_utc
   * price
   * size_eur
   * fee_eur (default 0)
6. positions
   * position_id PK (auto)
   * market_id FK -> markets (unique)
   * side
   * size_eur
   * avg_entry_price
   * last_price nullable
   * unrealized_pnl (default 0)
   * realized_pnl (default 0)
   * status (default "open")
   * opened_ts_utc
   * last_update_ts_utc
7. resolutions
   * resolution_id PK (auto)
   * market_id FK -> markets (unique)
   * resolved_ts_utc
   * outcome (YES, NO)

### Tables for M2-M5 (implemented)

8. evidence_items
   * evidence_id PK
   * ts_utc
   * source_type (rss, xai_search, xai_social)
   * url
   * title
   * published_ts_utc nullable
   * raw_content_ref
   * extracted_text
   * content_hash (unique index)
9. packets
   * packet_id PK
   * market_id FK
   * ts_utc
   * packet_json
   * packet_hash
   * packet_version
10. model_runs
    * run_id PK
    * market_id FK
    * packet_id FK
    * ts_utc
    * model_id
    * tier
    * prompt_version
    * charter_version
    * policy_version
    * raw_response
    * parsed_json
    * parse_ok boolean
    * budget_skip boolean
    * estimated_cost_eur
11. aggregations
    * agg_id PK
    * market_id FK
    * ts_utc
    * aggregation_json
    * policy_version
12. model_scores_daily
    * score_id PK
    * model_id
    * score_date
    * markets_scored
    * brier_score
    * log_loss
    * calibration_json
    * pnl_attrib_json
    * veto_value_json
    * notes

### Tables for M6 (implemented)

13. signal_snapshots
    * signal_id PK
    * market_id FK -> markets
    * ts_utc
    * odds_move_1h, odds_move_6h, odds_move_24h (nullable floats)
    * volume_ratio_24h (nullable float)
    * spread_current, spread_widening (nullable floats)
    * evidence_count_6h, evidence_count_24h, credible_evidence_6h (nullable ints)
    * google_trends_spike, wikipedia_spike (nullable floats)
    * triage_score (nullable float)
    * triage_reasons (JSON, nullable)
    * INDEX (market_id, ts_utc)
    * INDEX (ts_utc)

14. online_scores
    * score_id PK
    * market_id FK -> markets
    * ts_utc
    * model_id
    * p_true
    * outcome nullable
    * brier nullable
    * INDEX (market_id, ts_utc)

### Indexes

* market_snapshots by (market_id, ts_utc)
* decisions by ts_utc
* model_runs by (market_id, ts_utc)
* evidence_items by content_hash (unique) and url
* aggregations by (market_id, ts_utc)
* model_scores_daily by score_date, unique (model_id, score_date)
* signal_snapshots by (market_id, ts_utc) and (ts_utc)

## 14. Evaluation metrics

Forecast metrics per model:
* Brier score on resolved markets.
* Log loss optional.
* Calibration by bins.

Trading metrics:
* daily pnl, drawdown, exposure
* pnl per category
* win rate
* average edge at entry vs realized outcome

Attribution metrics:
* support value: model supported profitable trades
* dissent value: model dissented on losing trades
* veto value: model flagged ambiguity and trade would have lost
* sizing error: model recommended higher exposure on losing trades

Panel dynamics:
* disagreement vs pnl correlation
* redundancy between models

## 15. Implementation milestones

Milestone 1: Deterministic paper system **[DONE]**
* Gamma ingestion
* CLOB price watcher
* Candidate selection
* Paper executor
* Portfolio risk limits
* Basic reporting
* Docker Compose (Postgres + app)

Milestone 2: Evidence and packet **[DONE]**
* RSS ingestion with full-text enrichment
* Semantic evidence-to-market linking (OpenAI embeddings + keyword fallback)
* Evidence quality scoring
* Packet builder with hashing and versioning

Milestone 3: Full panel **[DONE]**
* LLM panel calls for 4 providers (OpenAI, Anthropic, Google, xAI)
* 3-tier escalation (default, escalation, arbiter)
* Strict JSON validation and retries
* Deterministic aggregation with consensus, disagreement, vetoes
* Per-provider daily budget caps with tiered model ladder (auto-downgrade to cheaper models)

Milestone 4: Open position review loop **[DONE]**
* Trigger detection: odds move, liquidity drop, approaching resolution, new evidence
* Panel re-analysis with packet rebuilding
* xAI search on odds move (web + social)
* Lifecycle decisions: hold, reduce, close (with aggregation-aware evaluation)
* Take-profit, edge flip, disagreement increase, ambiguity veto detection

Milestone 5: Scoring and leaderboards **[DONE]**
* Daily model scoring on resolved markets
* Brier score, log loss, calibration by bins
* PnL attribution (support value, dissent value, sizing error)
* Veto value tracking

Milestone 6: Signal collection and triage **[DONE]**
* Market microstructure signals (odds moves 1h/6h/24h, volume ratio, spread, spread widening)
* Evidence freshness signals (count by time window, credible vs social)
* Google Trends spike detection (optional, via pytrends)
* Wikipedia pageview spike detection (optional, via Wikimedia REST API)
* Signal collector orchestrating all sources independently
* Deterministic triage scorer (6 weighted components, 3 guardrails)
* Triage gate between CandidateSelector and LLM panel
* Guardrails: wide_spread (50% size reduction), spread_widening (75% factor), social_only (trade block)
* Signal-based review triggers: volume_surge, trends_spike, wiki_spike
* SignalSnapshot DB table for audit trail
* Panel cooldown: markets skip re-paneling for 8 hours after last LLM evaluation (broad market exploration)
* Batched DB operations: bulk_upsert_markets and bulk_add_snapshots with batch_size=500 to handle ~29k markets
* Market deduplication before bulk upsert (Gamma API returns duplicates across pages)
* 591 tests passing

## 16. Repo layout (Python monorepo)

```
agent-trader/
  README.md
  SPECS.md
  CLAUDE.md               # Project conventions for Claude Code
  charter.md
  policy.yaml
  docker-compose.yml
  Dockerfile
  pyproject.toml
  alembic.ini
  .env / .env.example
  .gitignore
  .dockerignore
  src/
    config/
      settings.py          # Pydantic Settings from env
      policy.py            # Policy model + YAML loader + hash
      budget.py            # Daily budget tracker
    db/
      models.py            # SQLAlchemy ORM (14 tables)
      session.py           # Async engine + session factory
      repository.py        # CRUD helpers
      migrations/
    polymarket/
      schemas.py           # Pydantic models for API responses
      gamma_client.py      # Market discovery
      clob_client.py       # Prices + order books
      ws_watcher.py        # WebSocket price feed
    app/
      candidate_selector.py  # Deterministic ranking
      scheduler.py           # TradingEngine orchestrator
      main.py                # Entrypoint + APScheduler
    evidence/              # M2
      schemas.py           # Evidence dataclasses
      rss_ingestor.py      # RSS feed ingestion
      xai_search.py        # xAI Grok live search
      linker.py            # Semantic evidence-market linking
      embedder.py          # OpenAI embedding client
      fulltext.py          # Full-text enrichment
      quality.py           # Evidence quality scoring
    packets/               # M2
      builder.py           # Packet construction + hashing
      schemas.py           # Packet Pydantic models
    llm/                   # M3
      base.py              # Base LLM client interface
      openai_client.py     # OpenAI provider
      anthropic_client.py  # Anthropic provider
      gemini_client.py     # Google Gemini provider
      xai_client.py        # xAI Grok provider
      panel.py             # Multi-model panel orchestrator
      prompt_builder.py    # Prompt construction
      schemas.py           # ModelProposal, PanelResult, ModelTierConfig, MODEL_LADDER
    aggregation/           # M3
      aggregator.py        # Deterministic consensus + vetoes
    signals/               # M6
      schemas.py           # Signal dataclasses (7 types)
      microstructure.py    # Odds moves, volume ratios, spreads
      trends.py            # Google Trends spike detection
      wikipedia.py         # Wikipedia pageview spike detection
      collector.py         # Signal collection orchestrator
      triage.py            # Deterministic triage scorer + guardrails
    portfolio/
      sizing.py            # Edge-based position sizing
      risk_manager.py      # 5 constraint checks
      lifecycle.py         # Hold/reduce/close logic
    execution/
      paper_executor.py    # Simulated fills
      fills.py             # PnL calculations
    evaluation/            # M5
      metrics.py           # Brier score, log loss, calibration
      attribution.py       # PnL attribution (support, dissent, veto, sizing error)
      scorer.py            # Daily model scoring orchestrator
      reports.py           # Daily reports
  tests/                   # 591 tests
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

## 17. Operational notes

* Use UTC everywhere.
* Validate all JSON with Pydantic models.
* Never execute trades if any critical data is missing (rules, odds, liquidity).
* Use deterministic packet pruning and deterministic candidate ranking.
* Use daily budgets and stop panel calls when exceeded.
* Keep a clear audit trail for every decision.

End of spec.


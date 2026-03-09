# Structured Trader — Specification & Roadmap

Deterministic, source-driven paper trading system for Polymarket.
Uses authoritative structured data (NWS, BLS, FRED, Binance, EDGAR) and
deterministic probability models instead of LLM panels.  Only trades markets
that can be mapped to clear settlement rules with parseable contract specs.

---

## Architecture Overview

```
Gamma API  →  Market Ingestion  →  Contract Parsing  →  Category Router
                                                             │
           ┌─────────────────┬───────────────┬───────────────┘
           ▼                 ▼               ▼               ▼
    Weather Pipeline  Macro Pipeline  Crypto Pipeline  Earnings Pipeline
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
    │ NWS / AWC   │  │ BLS / FRED  │  │ Binance /   │  │ EDGAR /      │
    │ Source Fetch │  │ Source Fetch │  │ Coinbase /  │  │ Ticker       │
    └──────┬──────┘  └──────┬──────┘  │ Kraken      │  │ Resolver     │
           ▼                ▼         └──────┬──────┘  └──────┬───────┘
    Weather Engine   Macro Engine     Crypto Engine    Earnings Engine
    (CDF → p_yes)   (CDF → p_yes)   (CDF → p_yes)   (CDF → p_yes)
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
           └────────────┬────────────────┬────────────────┬────┘
                        ▼                ▼                ▼
                     Position Sizing + Risk
                                ▼
                        Paper Executor
                                ▼
                    Calibration + Kill Switch (S6)
```

Four independent trading pipelines, each with its own 10,000 EUR bankroll,
isolated risk limits, data sources, and pricing engine. Total bankroll:
40,000 EUR.

---

## Phase Plan

| Phase | Name | Status | Tests |
|-------|------|--------|-------|
| S0 | Project Bootstrap + Shared Core | **Done** | 166 |
| S1 | Contract Parsing + Market Router | **Done** | +39 = 205 |
| S2 | Weather Engine v1 | **Done** | +98 = 303 |
| S3 | Macro Engine v1 | **Done** | +73 = 376 |
| S4 | Crypto Engine v1 | **Done** | +96 = 472 |
| S5 | Earnings Engine v1 | **Done** | +68 = 540 |
| S6 | Calibration + Kill Switch | **Done** | +24 = 564 |
| S7 | Position Lifecycle + Entry Guards | **Done** | +50 = 614 |

---

## S0: Project Bootstrap + Shared Core ✓

**Goal:** Self-contained `structured/` subdirectory inside agent-trader
with DB, market ingestion, policy, portfolio, and execution — zero LLM
dependencies.

**Delivered:**
- 15 DB tables (7 core from parent + 8 new structured tables)
- Market ingestion via Gamma API (`StructuredTradingEngine.ingest_markets()`)
- Category-aware policy (`Policy.for_category()`, `CategoryPolicy`)
- Paper executor and PnL calculations (copied from parent)
- Position sizing and risk manager with category filtering
- Position lifecycle (hold / reduce / close)
- Docker Compose (Postgres 16 on port 5433, separate DB)
- 166 tests passing

**Key files:**
- `src/app/scheduler.py` — `StructuredTradingEngine` orchestrator
- `src/app/main.py` — entrypoint + APScheduler
- `src/db/models.py` — 15 SQLAlchemy tables
- `src/db/repository.py` — async CRUD helpers
- `src/config/policy.py` — Policy + CategoryPolicy + `for_category()`

---

## S1: Contract Parsing + Market Router ✓

**Goal:** Classify ingested markets into categories and extract structured
`ContractSpec` with all fields needed for pricing engines.

**Delivered:**
- `ContractParser` ABC with `can_parse()` / `parse()` interface
- `WeatherParser` — regex patterns for temperature (0.90), precipitation
  (0.85), hurricane (0.85), snow occurrence (0.80). Word-boundary matching
  (`\b`) for short keywords (rain, snow, wind, hurricane). No generic fallback —
  keyword-only matches without structural patterns are rejected.
- `MacroParser` — regex patterns for CPI/unemployment/indicator thresholds
  (0.90), nonfarm payrolls (0.90), Fed rate (0.85), GDP (0.85)
- `ParserRegistry` — ordered dispatch, first match wins
- `classify_markets_batch()` — async batch classification, persists
  `CategoryAssignment` rows (parsed or rejected)
- `StructuredTradingEngine.classify_markets()` wired to real classification
- Negative context filters: sports blocklist (NHL, Stanley Cup, Carolina
  Hurricanes, Miami Heat, etc.) and geopolitics blocklist (Ukraine, Russia,
  ceasefire, election) prevent false positives
- 65 tests (39 original + 26 precision regression tests)

**Key files:**
- `src/contracts/base.py` — ABC, ContractSpec, ParseResult
- `src/contracts/weather.py` — WeatherContractSpec, WeatherParser
- `src/contracts/macro.py` — MacroContractSpec, MacroParser
- `src/contracts/registry.py` — ParserRegistry, classify_markets_batch()

**Data flow:**
```
get_unparsed_markets() → for each market:
  registry.classify(question, rules_text)
    → WeatherParser.can_parse() → WeatherParser.parse() → WeatherContractSpec
    → MacroParser.can_parse()  → MacroParser.parse()  → MacroContractSpec
    → CryptoParser.can_parse() → CryptoParser.parse() → CryptoContractSpec
    → EarningsParser.can_parse() → EarningsParser.parse() → EarningsContractSpec
    → no match → ParseResult(matched=False, reject_reason="no_parser_matched")
  → upsert_category_assignment(parse_status="parsed"|"rejected")
```

---

## S2: Weather Engine v1 ✓

**Goal:** Build the first full trading pipeline for weather markets —
from data fetching through probability computation to trade execution.

**Delivered:**

### Source Adapters
- **`SourceAdapter` ABC** (`src/sources/base.py`): Base class with `fetch()`
  and `health_check()`. `FetchResult` dataclass with `ok` property and
  `to_observation_dict()` for DB persistence.
- **`NWSAdapter`** (`src/sources/nws.py`): Grid-point hourly forecasts from
  api.weather.gov. Geocodes location → `/points/{lat},{lon}` → grid →
  `/gridpoints/{office}/{x},{y}/forecast/hourly`. Caches grid lookups.
  Retry on 500/503 (max 3). Quality score decays with lead time:
  `exp(-lead_hours / (2 × forecast_horizon_hours))`. User-Agent from settings.
- **`AWCAdapter`** (`src/sources/awc.py`): METAR observations from Aviation
  Weather Center (aviationweather.gov). Fetches actual temperature, precipitation,
  wind, visibility for station IDs near contract location. Used near resolution
  windows to validate forecasts against reality.
- **`geocoder`** (`src/sources/geocoder.py`): Static dict of ~55 US cities →
  (lat, lon). Fallback: Census Bureau geocoder API via httpx. Module-level cache.

### Date Resolution
- **`date_resolver`** (`src/trading/date_resolver.py`): Converts
  `date_description` → `(start, end)` datetime range. Handles named dates
  (Christmas, Independence Day), quarters (Q1-Q4), month+year, hurricane
  season, and generic dateutil fallback.

### Weather Probability Engine
- **`PricingEngine` ABC** (`src/engines/base.py`): `compute(spec, observation)`
  → `PriceEstimate` (p_yes, confidence, source_confidence, model_details).
- **`WeatherEngine`** (`src/engines/weather.py`): Dispatches by metric:
  - **Temperature:** Gaussian CDF. `p_yes = 1 - norm.cdf(threshold, loc=forecast_max, scale=spread)` where `spread = 2.0°F + 0.02 × lead_hours`.
  - **Precipitation:** Zero-inflated. `p_yes = P(rain) × (1 - gamma.cdf(threshold, a=2.0, scale=0.5))`. Confidence capped 0.70.
  - **Snow occurrence:** Joint probability: `P(temp < 33°F) × P(precip > 0)`. Confidence capped 0.75.
  - **Snowfall amount:** Cold probability × precip probability × Gamma amount distribution. Confidence capped 0.70.
  - **Hurricane:** Seasonal base rates (cat3+ = 0.45, cat5 = 0.05, any = 0.70). Confidence 0.30 (real NHC model deferred).
- All p_yes clamped to [0.01, 0.99].
- Confidence decay: `exp(-lead_hours / (2 × forecast_horizon_hours))`.

### Pipeline Integration
- **`WeatherPipeline`** (`src/trading/weather_pipeline.py`): Full trading loop:
  1. `repo.get_markets_by_category("weather")` → assignments
  2. Reconstruct `WeatherContractSpec` from stored `contract_spec_json`
  3. `resolve_date_range()` → skip if past resolution
  4. `nws.fetch(spec)` → `FetchResult` → `repo.add_source_observation()`
  5. `engine.compute(spec, observation)` → `PriceEstimate` → `repo.add_engine_price()`
  6. Near-resolution override (< 24h): AWC METAR replaces forecast, confidence → 0.95
  7. `compute_size()` → edge-based sizing with confidence ramp
  8. `risk_manager.check_new_trade_category()` → 5 constraints
  9. `executor.execute()` → persist Decision, Order, Fill, Position
  10. Return summary dict
- Scheduler: 2-minute interval, max_instances=1, coalesce=True.

### Tests: 72 (engine) + 26 (parser precision regression) = 98 new

---

## S3: Macro Engine v1 ✓

**Goal:** Build the economic/macro data trading pipeline — analogous to
S2 but for macroeconomic indicators.

**Delivered:**

### Source Adapters
- **`BLSAdapter`** (`src/sources/bls.py`): CPI, nonfarm payrolls, unemployment
  rate, PPI, PCE from api.bls.gov
- **`FREDAdapter`** (`src/sources/fred.py`): GDP, interest rates, historical
  economic series from api.stlouisfed.org
- **`ReleaseCalendar`** (`src/sources/release_calendar.py`): Tracks economic
  data release dates (BLS schedule, FOMC calendar), near-event high-frequency
  polling

### Macro Probability Engine
- **`MacroEngine`** (`src/engines/macro.py`): Historical changes → fitted
  distribution (normal/t-distribution) → CDF → p_yes
- Separate models per indicator:
  - **CPI/PPI/PCE:** Historical month-over-month change distribution
  - **Unemployment:** Historical change distribution + current trend
  - **GDP:** Quarterly growth rate distribution
  - **Nonfarm payrolls:** Historical deviation from consensus
  - **Fed rate:** Fed funds futures implied probabilities + dot plot signals

### Pipeline Integration
- **`MacroPipeline`** (`src/trading/macro_pipeline.py`): Full macro trading
  loop following the same pattern as weather
- `MacroContractSpec.bls_series_id` / `fred_series_id` resolved from
  indicator name
- Scheduler: 5-minute interval

### Tests: +73 new

---

## S4: Crypto Engine v1 ✓

**Goal:** Build the cryptocurrency price trading pipeline for BTC, ETH,
SOL and other major tokens.

**Delivered:**

### Source Adapters
- **`BinanceAdapter`** (`src/sources/binance.py`): Real-time crypto prices
- **`CoinbaseAdapter`** (`src/sources/coinbase.py`): Real-time crypto prices
- **`KrakenAdapter`** (`src/sources/kraken.py`): Real-time crypto prices
- **`ExchangeRouter`** (`src/sources/exchange_router.py`): Routes to best
  exchange per asset, handles failover

### Contract Parser
- **`CryptoParser`** (`src/contracts/crypto.py`): Three pattern types:
  - **Price threshold** (0.90): "Will BTC exceed $100,000 by June 30?"
  - **Trading above/below** (0.85): "Will ETH be trading above $5,000?"
  - **All-time high** (0.85): "Will Bitcoin hit a new ATH by March 2026?"
- Supports BTC, ETH, SOL, DOGE, ADA, XRP
- Negative context filters: DAO governance, NFT, blockchain tech, mining,
  gas fees, ethnic false positives

### Crypto Probability Engine
- **`CryptoEngine`** (`src/engines/crypto.py`): Volatility-based CDF pricing
  with exchange-specific confidence
- Default daily volatility: 3.0%

### Pipeline Integration
- **`CryptoPipeline`** (`src/trading/crypto_pipeline.py`): Full crypto
  trading loop, 1-minute cycle interval (fastest pipeline)

### Tests: +96 new

---

## S5: Earnings Engine v1 ✓

**Goal:** Build the corporate earnings trading pipeline for EPS, revenue,
and SEC filing markets.

**Delivered:**

### Source Adapters
- **`EDGARAdapter`** (`src/sources/edgar.py`): SEC EDGAR filings (10-K, 10-Q,
  8-K) via efts.sec.gov
- **`TickerResolver`** (`src/sources/ticker_resolver.py`): Company name →
  ticker symbol mapping

### Contract Parser
- **`EarningsParser`** (`src/contracts/earnings.py`): Three pattern types:
  - **Earnings threshold** (0.90): "Will Apple's EPS exceed $1.50 in Q1 2026?"
  - **Filing existence** (0.85): "Will Apple file its 10-K by March 2026?"
  - **Ticker-based** (0.85): "Will $AAPL EPS beat estimates in Q1 2026?"
- Negative context filters: crypto earnings, staking earnings, mining earnings

### Earnings Probability Engine
- **`EarningsEngine`** (`src/engines/earnings.py`): Historical EPS/revenue
  distribution → CDF → p_yes
- Pre-filing confidence: 0.30

### Pipeline Integration
- **`EarningsPipeline`** (`src/trading/earnings_pipeline.py`): Full earnings
  trading loop, 10-minute cycle interval

### Tests: +68 new

---

## S6: Calibration + Kill Switch ✓

**Goal:** Add performance measurement, automatic quality controls, and
daily PnL tracking.

**Delivered:**

### Calibration
- **Brier score** per category, per engine version, daily
- **Log loss** as secondary metric
- Stored in `calibration_stats` table
- Runs at 00:30 UTC daily

### Kill Switch
- **`KillSwitch`** (`src/evaluation/kill_switch.py`): Auto-disables a category
  when Brier score exceeds threshold over rolling N days
- Per-category enable/disable tracked in memory
- Fed by calibration results after each evaluation
- Reset daily at midnight UTC

### Category PnL Tracking
- Daily realized + unrealized PnL per category
- Trades opened / closed counts
- Stored in `category_pnl_daily` table
- Aggregated at 00:10 UTC daily

### Replay Engine
- **`replay.py`** (`src/evaluation/replay.py`): Historical replay for
  regression testing engine versions

### Tests: +24 new

---

## S7: Position Lifecycle + Entry Guards ✓

**Goal:** Add structured position management with entry guards to prevent
duplicate positions, reentry cooldown, per-category horizon filters, and
an 8-rule engine-driven lifecycle evaluator.

**Delivered:**

### Entry Guards
- **Duplicate position block:** Each pipeline calls `repo.get_position(market_id)`
  before sizing — skips markets with existing open positions
- **Reentry cooldown:** After closing a position, blocks re-entry for
  `reentry_cooldown_hours` (6h default). Per-category override via
  `CategoryPolicy.reentry_cooldown_hours`
- **Per-category horizon filter:** `max_hours_to_resolution` per category
  in `policy.yaml` (e.g., weather: 168h, crypto: 720h)

### Enhanced Lifecycle (8 rules)
- **`PositionLifecycle.evaluate()`** — 4 deterministic safety rules:
  no snapshot → CLOSE, approaching resolution → CLOSE, mid moved against > 2×
  edge threshold → CLOSE, unrealized PnL < -3% → REDUCE
- **`PositionLifecycle.evaluate_with_engine()`** — 4 additional engine-driven rules:
  min hold time gate (30 min), edge flip with cost buffer → CLOSE,
  take profit (edge < band + cost buffer) → CLOSE,
  confidence collapse (< `min_confidence_hard`) → CLOSE

### Position Review Loop
- `StructuredTradingEngine.review_open_positions()` runs every 5 minutes
- Per-position: fetch engine price + snapshot → check staleness → evaluate
- Engine staleness: only uses engine-driven rules if price < `engine_stale_hours` old
- Category-specific policy applied via `Policy.for_category()`
- Executes CLOSE and REDUCE decisions through `PaperExecutor` with realized PnL tracking

### Policy Additions
- `min_hold_minutes` (default 30): min hold time before engine exits
- `exit_flip_threshold` (default 0.04): edge flip threshold
- `take_profit_band` (default 0.02): take profit edge band
- `reentry_cooldown_hours` (default 6): per-category reentry cooldown
- `engine_stale_hours` (default 2): max age for engine prices

### Key files:
- `src/portfolio/lifecycle.py` — `PositionLifecycle` with `evaluate()` and `evaluate_with_engine()`
- `src/app/scheduler.py` — `review_open_positions()`, `_review_position()`, `_execute_close()`, `_execute_reduce()`
- `src/trading/weather_pipeline.py` — entry guard + reentry cooldown (same pattern in all 4 pipelines)
- `src/config/policy.py` — `CategoryPolicy` with new lifecycle fields

### Tests: +50 new (entry guards, lifecycle, review loop)

---

## Design Principles

1. **Source-driven, not LLM-driven:** Only trade markets with verifiable
   settlement criteria and authoritative data sources.
2. **Deterministic:** Given the same source data and policy, the system
   always makes the same decision. No randomness, no LLM variance.
3. **Category isolation:** Each category has its own bankroll, risk limits,
   pricing engine, and data sources. A bad weather model cannot blow up
   crypto positions.
4. **Parse or skip:** If a market cannot be mapped to a strict
   `ContractSpec`, it is rejected and never traded.
5. **Confidence decay:** Probability estimates carry confidence that
   degrades with forecast lead time and source staleness.
6. **Calibration-gated:** Categories can be automatically disabled when
   calibration degrades beyond thresholds.

---

## Database Tables (15)

### Core (7, from parent)
markets, market_snapshots, decisions, orders, fills, positions, resolutions

### Structured (8, new)
category_assignments, source_observations, engine_prices,
category_portfolios, category_pnl_daily, calibration_stats,
backtest_runs, market_resolutions

---

## Policy Summary

- Global bankroll: 40,000 EUR
- Weather bankroll: 10,000 EUR (edge threshold: 6%)
- Macro bankroll: 10,000 EUR (edge threshold: 8%)
- Crypto bankroll: 10,000 EUR (edge threshold: 8%)
- Earnings bankroll: 10,000 EUR (edge threshold: 10%)
- Category-specific overrides via `policy.yaml` → `Policy.for_category()`

# Agent-Trader Structured

Deterministic, source-driven paper trading system for Polymarket. Uses
authoritative structured data sources (NWS, BLS, FRED, Binance, EDGAR) and
deterministic probability models (CDF-based) instead of LLM panels. Only trades
markets that can be mapped to clear settlement rules with parseable contract
specs.

Lives as a self-contained subdirectory inside `agent-trader` with its own DB,
policy, and Docker Compose stack.

## Status

| Phase | Name | Status | Tests |
|-------|------|--------|-------|
| S0 | Project Bootstrap + Shared Core | Done | 166 |
| S1 | Contract Parsing + Market Router | Done | +39 = 205 |
| S2 | Weather Engine v1 | Done | +98 = 303 |
| S3 | Macro Engine v1 | Done | +73 = 376 |
| S4 | Crypto Engine v1 | Done | +96 = 472 |
| S5 | Earnings Engine v1 | Done | +68 = 540 |
| S6 | Calibration + Kill Switch | Done | +24 = 564 |
| S7 | Position Lifecycle + Entry Guards | Done | +50 = 614 |

## Quick Start

```bash
conda create -n agent-trader-structured python=3.12 -y
conda run -n agent-trader-structured pip install -e ".[dev]"
conda run -n agent-trader-structured python -m pytest tests/ -v
```

## Docker

```bash
docker compose up --build -d
docker compose logs -f app
docker compose down
```

The app starts APScheduler with ten jobs:

| Job | Interval | Description |
|-----|----------|-------------|
| `ingest_markets` | 15 min | Fetch all ~29k active markets from Gamma API |
| `classify_markets` | 15 min | Parse contracts into weather/macro/crypto/earnings |
| `weather_cycle` | 2 min | NWS fetch → CDF pricing → sizing → paper trade |
| `macro_cycle` | 5 min | BLS/FRED fetch → CDF pricing → sizing → paper trade |
| `crypto_cycle` | 1 min | Exchange price fetch → CDF pricing → sizing → paper trade |
| `earnings_cycle` | 10 min | EDGAR fetch → CDF pricing → sizing → paper trade |
| `aggregate_daily_pnl` | 00:10 UTC | Daily realized + unrealized PnL per category |
| `calibration_cycle` | 00:30 UTC | Brier score + kill switch evaluation |
| `position_review` | 5 min | Lifecycle review for all open positions |
| `daily_reset` | 00:00 UTC | Reset daily tracking + kill switch |

## Architecture

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

Each category has its own 10,000 EUR bankroll slice and isolated risk limits
via `Policy.for_category()`.

## Project Structure

```
structured/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── policy.yaml                    # Risk/sizing parameters per category
├── src/
│   ├── app/
│   │   ├── main.py                # Entrypoint — wires components + APScheduler
│   │   └── scheduler.py           # StructuredTradingEngine orchestrator
│   ├── config/
│   │   ├── settings.py            # Pydantic Settings (env vars)
│   │   ├── policy.py              # Policy + CategoryPolicy models
│   │   └── categories.py          # Category enum
│   ├── contracts/
│   │   ├── base.py                # ContractParser ABC, ContractSpec, ParseResult
│   │   ├── weather.py             # WeatherParser + word-boundary filters
│   │   ├── macro.py               # MacroParser (CPI, GDP, unemployment, etc.)
│   │   ├── crypto.py              # CryptoParser (BTC/ETH/SOL price thresholds, ATH)
│   │   ├── earnings.py            # EarningsParser (EPS, revenue, SEC filings)
│   │   └── registry.py            # ParserRegistry, classify_markets_batch()
│   ├── sources/
│   │   ├── base.py                # SourceAdapter ABC + FetchResult
│   │   ├── geocoder.py            # Location → (lat, lon) + Census fallback
│   │   ├── nws.py                 # NWS API adapter (forecasts)
│   │   ├── awc.py                 # Aviation Weather Center (METAR observations)
│   │   ├── bls.py                 # BLS API client (CPI, payrolls, unemployment)
│   │   ├── fred.py                # FRED API client (GDP, interest rates)
│   │   ├── release_calendar.py    # Economic release date tracker
│   │   ├── binance.py             # Binance exchange adapter
│   │   ├── coinbase.py            # Coinbase exchange adapter
│   │   ├── kraken.py              # Kraken exchange adapter
│   │   ├── exchange_router.py     # Routes to best exchange per asset
│   │   ├── edgar.py               # SEC EDGAR filings adapter
│   │   └── ticker_resolver.py     # Company name → ticker symbol
│   ├── engines/
│   │   ├── base.py                # PricingEngine ABC + PriceEstimate
│   │   ├── weather.py             # WeatherEngine (Gaussian/Gamma/joint CDF)
│   │   ├── macro.py               # MacroEngine (historical distribution CDF)
│   │   ├── crypto.py              # CryptoEngine (volatility-based CDF)
│   │   └── earnings.py            # EarningsEngine (historical EPS/revenue CDF)
│   ├── trading/
│   │   ├── date_resolver.py       # date_description → (start, end) datetime
│   │   ├── weather_pipeline.py    # Full weather trading loop orchestrator
│   │   ├── macro_pipeline.py      # Full macro trading loop orchestrator
│   │   ├── crypto_pipeline.py     # Full crypto trading loop orchestrator
│   │   └── earnings_pipeline.py   # Full earnings trading loop orchestrator
│   ├── portfolio/
│   │   ├── sizing.py              # Edge-based position sizing
│   │   ├── risk_manager.py        # 5 independent risk constraints
│   │   └── lifecycle.py           # Position hold/reduce/close
│   ├── execution/
│   │   ├── paper_executor.py      # Paper trade executor
│   │   └── fills.py               # Fill price calculation
│   ├── polymarket/
│   │   ├── gamma_client.py        # Gamma API client (market data)
│   │   ├── clob_client.py         # CLOB client (orderbook)
│   │   ├── ws_watcher.py          # WebSocket price watcher
│   │   └── schemas.py             # Pydantic schemas
│   ├── db/
│   │   ├── models.py              # 15 SQLAlchemy tables
│   │   ├── repository.py          # Async CRUD helpers
│   │   └── session.py             # Engine + session factory
│   └── evaluation/
│       ├── calibration.py         # Brier score, log loss, calibration curves
│       ├── kill_switch.py         # Auto-disable on poor performance
│       └── replay.py              # Historical replay engine
└── tests/
    ├── test_config.py
    ├── test_contracts.py          # 65 tests (incl. 26 precision regression)
    ├── test_db.py
    ├── test_execution.py
    ├── test_polymarket.py
    ├── test_portfolio.py
    ├── test_scheduler.py
    ├── test_weather_engine.py     # 72 tests (S2 engine + pipeline)
    ├── test_macro_engine.py       # S3 macro engine + pipeline
    ├── test_crypto_engine.py      # S4 crypto engine + pipeline
    ├── test_earnings_engine.py    # S5 earnings engine + pipeline
    ├── test_calibration.py        # S6 calibration + kill switch
    └── ...
```

## Weather Pipeline (S2)

The weather pipeline runs every 2 minutes and processes all markets classified
as `weather`:

1. **Fetch assignments** — `repo.get_markets_by_category("weather")`
2. **Reconstruct spec** — rebuild `WeatherContractSpec` from stored JSON
3. **Resolve dates** — `date_description` → `(start, end)` datetime range
4. **Fetch forecast** — NWS hourly grid-point forecasts via `NWSAdapter`
5. **Compute p_yes** — CDF-based probability per metric type
6. **Observation override** — within 24h of resolution, AWC METAR replaces forecast
7. **Size position** — edge-based sizing with confidence ramp
8. **Risk check** — 5 independent constraints (exposure, drawdown, etc.)
9. **Paper execute** — simulate fill with slippage
10. **Persist** — Decision, Order, Fill, Position to DB

### Probability Models

| Metric | Model | Confidence Cap |
|--------|-------|----------------|
| Temperature | Gaussian CDF, spread = 2°F + 0.02 × lead_hours | — |
| Precipitation | Zero-inflated: P(rain) × (1 - Gamma CDF) | 0.70 |
| Snow occurrence | Joint: P(temp < 33°F) × P(precip > 0) | 0.75 |
| Snowfall amount | Cold prob × precip prob × Gamma amount | 0.70 |
| Hurricane | Seasonal base rates (cat3+ = 0.45, any = 0.70) | 0.30 |

### Classifier Precision

The weather parser uses word-boundary regex (`\b`) and negative context filters
to avoid false positives:

- **Sports blocklist** — NHL, NBA, NFL, team names (Carolina Hurricanes, Miami Heat)
- **Geopolitics blocklist** — Ukraine, Russia, ceasefire, election
- **No generic fallback** — keyword-only matches without structural patterns are rejected

## Macro Pipeline (S3)

The macro pipeline runs every 5 minutes for markets classified as `macro`:

- **Sources:** BLS API (CPI, payrolls, unemployment) and FRED API (GDP, interest rates)
- **Release calendar:** Tracks economic data release dates for near-event polling
- **Engine:** Historical changes → fitted distribution → CDF → p_yes
- **Indicators:** CPI/PPI/PCE, unemployment, GDP, nonfarm payrolls, Fed rate

## Crypto Pipeline (S4)

The crypto pipeline runs every 1 minute for markets classified as `crypto`:

- **Sources:** Binance, Coinbase, Kraken via exchange router (best price per asset)
- **Assets:** BTC, ETH, SOL, DOGE, ADA, XRP
- **Patterns:** Price thresholds ("Will BTC exceed $100k?"), trading above/below, all-time high
- **Engine:** Volatility-based CDF pricing with exchange-specific confidence
- **ATH tracking:** Hardcoded reference prices for ATH markets

## Earnings Pipeline (S5)

The earnings pipeline runs every 10 minutes for markets classified as `earnings`:

- **Sources:** SEC EDGAR for filings, ticker resolver for company → symbol mapping
- **Patterns:** EPS/revenue thresholds, SEC filing existence (10-K, 10-Q, 8-K), ticker-based
- **Engine:** Historical EPS/revenue distribution → CDF → p_yes
- **Metrics:** EPS, revenue, guidance, filing deadlines

## Calibration + Kill Switch (S6)

Daily evaluation of prediction quality and automatic safety controls:

- **Brier score** per category, per engine version, daily
- **Kill switch:** Auto-disables a category when Brier score exceeds threshold
- **Daily PnL aggregation:** Realized + unrealized per category
- **Daily reset:** Clears daily tracking and kill switch state at midnight UTC

## Position Lifecycle + Entry Guards (S7)

Enhanced position management with 8-rule lifecycle and entry guards:

### Entry Guards
- **Duplicate position block** — each pipeline calls `repo.get_position()` before
  sizing; skips if an open position already exists for that market
- **Reentry cooldown** — after closing a position, blocks re-entry for
  `reentry_cooldown_hours` (6h default, per-category configurable)
- **Horizon filter** — per-category `max_hours_to_resolution` blocks markets
  that resolve too far in the future

### Position Lifecycle (8 rules)
The lifecycle evaluator runs in two tiers:

**Deterministic safety rules (always run):**
1. Close if no market snapshot available
2. Close if approaching resolution (`hours_to_resolution < min_hours_to_resolution`)
3. Close if mid moved against position by > 2x edge threshold (stop loss)
4. Reduce if unrealized PnL < -3% of position size (partial stop)

**Engine-driven rules (when fresh engine price available):**
5. Min hold time gate — skip engine exits on positions younger than `min_hold_minutes` (30 min)
6. Edge flip — close if engine's signed edge flips against position beyond `exit_flip_threshold` + cost buffer
7. Take profit — close if remaining edge won't cover exit costs (`spread/2 + slippage + fee`)
8. Confidence collapse — close if `engine_confidence < min_confidence_hard`

### Review Loop
- `review_open_positions()` runs every 5 minutes via APScheduler
- Fetches latest engine price and snapshot per position
- Applies category-specific policy via `Policy.for_category()`
- Engine staleness check: only uses engine rules if price is < `engine_stale_hours` old

## Policy

Configuration lives in `policy.yaml`:

```yaml
bankroll_eur: 40000.0              # Global bankroll
categories:
  weather:
    bankroll_eur: 10000.0          # Isolated weather bankroll
    max_open_positions: 10
    edge_threshold: 0.06
    engine_params:
      forecast_horizon_hours: 168
  macro:
    bankroll_eur: 10000.0
    max_open_positions: 10
    edge_threshold: 0.08
  crypto:
    bankroll_eur: 10000.0
    max_open_positions: 10
    edge_threshold: 0.08
  earnings:
    bankroll_eur: 10000.0
    max_open_positions: 10
    edge_threshold: 0.10
```

`Policy.for_category("weather")` returns a merged `CategoryPolicy` with
category-specific overrides applied on top of global defaults.

## Database

Postgres 16, port 5433 (separate from parent project on 5432). 15 tables:

**Core (7):** markets, market_snapshots, decisions, orders, fills, positions, resolutions

**Structured (8):** category_assignments, source_observations, engine_prices,
category_portfolios, category_pnl_daily, calibration_stats, backtest_runs,
market_resolutions

## Data Sources

| Source | API | Auth | Used For |
|--------|-----|------|----------|
| NWS | api.weather.gov | User-Agent only | Hourly forecasts |
| AWC | aviationweather.gov/api/data | None | METAR observations |
| Census | geocoding.geo.census.gov | None | Location → lat/lon fallback |
| BLS | api.bls.gov | API key | CPI, payrolls, unemployment |
| FRED | api.stlouisfed.org | API key | GDP, interest rates |
| Binance | api.binance.com | None | Crypto prices |
| Coinbase | api.coinbase.com | None | Crypto prices |
| Kraken | api.kraken.com | None | Crypto prices |
| EDGAR | efts.sec.gov | User-Agent only | SEC filings |
| Gamma | gamma-api.polymarket.com | None | Market data |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...localhost:5433/...` | Postgres connection |
| `GAMMA_API_BASE_URL` | `https://gamma-api.polymarket.com` | Gamma API |
| `NWS_USER_AGENT` | `agent-trader-structured` | NWS API User-Agent |
| `LOG_LEVEL` | `INFO` | Logging level |
| `POLICY_PATH` | `policy.yaml` | Policy file path |

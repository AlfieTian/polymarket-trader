# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_bayesian_engine.py -v

# Start the trader (reads config.yaml; dry_run=false means live orders)
python scripts/run_trader.py

# Run backtest
python scripts/backtest.py

# Check LLM calibration
python scripts/calibration_check.py

# One-time setup: approve USDC.e allowance on-chain (needed before live trading)
python scripts/approve_allowance.py
python scripts/onchain_approve.py

# Manually redeem resolved positions
python scripts/auto_redeem.py
```

## Credentials

All secrets are loaded from `.env` at the project root (never committed). Required variables:
- `POLYMARKET_PRIVATE_KEY` — Polygon wallet private key
- `POLYMARKET_WALLET_ADDRESS` — wallet address
- `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE` — CLOB API credentials
- `OPENAI_API_KEY` — used by `NewsFeed` for LLM market analysis (currently o4-mini)

## Architecture

The main trading loop runs in `scripts/run_trader.py` (`Trader.run_cycle`):

```
PolymarketClient.get_markets()
  → NewsFeed.get_llm_signals()  (parallel, o4-mini LLM per market)
    → BayesianEngine.batch_update()  (log-space sequential updates)
      → EdgeDetector.scan_markets()  (EV = p̂ - market_price, threshold min_edge)
        → KellySizer.calculate()  (fractional Kelly, portfolio cap)
          → RiskManager.validate_trade()  (daily loss, concentration limits)
            → CLOBExecutor.place_order()  (BUY via py-clob-client)
              → PositionManager.add_position()  (persisted to logs/positions_state.json)
```

Exits run **before** the LLM fetch (fast path: price-based stops/TPs) and **after** (edge-reversal exits). Auto-redemption of resolved markets runs every 20 cycles via `Redeemer`.

### Key modules

| File | Role |
|------|------|
| `src/signals/bayesian_engine.py` | `BayesianEngine` + `BeliefState` — log-space Bayesian updates per market |
| `src/signals/lmsr_pricer.py` | LMSR reference pricing; `EdgeDetector` uses it for EV calc |
| `src/strategy/edge_detector.py` | Scans markets for EV edge; filters by `min_edge` |
| `src/strategy/kelly_sizer.py` | Fractional Kelly sizing with per-market and portfolio caps |
| `src/strategy/performance_tracker.py` | Tracks closed trades; drives adaptive parameter updates |
| `src/data/polymarket_client.py` | REST market discovery + sports/esports ban list |
| `src/data/news_feed.py` | Fetches news, calls LLM, converts sentiment to `Signal` objects |
| `src/execution/clob_executor.py` | `CLOBExecutor` — wraps `py-clob-client`; dry-run aware |
| `src/execution/position_manager.py` | Persistent position state in `logs/positions_state.json` |
| `src/execution/redeemer.py` | On-chain redemption of resolved CTF positions |
| `src/risk/risk_manager.py` | Daily loss, market concentration, per-market loss limits |

### State persistence

Open positions survive restarts via `logs/positions_state.json`. Two cooldown files prevent re-entry after closes:
- `logs/force_close_cooldown.json` — 24h ban after failed exit retries
- `logs/exit_cooldown.json` — 4h ban after any close

On startup, `Trader._reconcile_startup_state()` cancels stale remote orders and reconciles local positions against on-chain token balances.

### Configuration (`config.yaml`)

Key tunables:
- `bayesian.prior_weight` — how strongly the market price anchors the prior (0.95 = near-full trust in market)
- `bayesian.min_confidence` — minimum average LLM confidence to open a position
- `strategy.min_edge` — minimum EV threshold (5% default)
- `strategy.kelly_fraction` — sizing multiplier (0.25 = quarter Kelly)
- `position.profit_target_pct` / `stop_loss_pct` — exit thresholds
- `trading.ban_sports_esports` — filter in `polymarket_client.py`

### Dry-run vs live

`config.yaml` `polymarket.dry_run: false` enables live trading. `CLOBExecutor` and `PolymarketClient` both check this flag independently. Polymarket minimum order is **5 shares**; the executor enforces this minimum automatically.

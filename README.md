# Polymarket Trading System

Quantitative prediction-market trading framework built on Bayesian signal processing and CLOB execution.

## Architecture

```
Data Ingestion → Bayesian Signal Engine → Edge Detection → Fractional Kelly Sizing → Risk Management → CLOB Execution
```

### Modules

| Module | Description |
|------|------|
| `src/signals/lmsr_pricer.py` | LMSR reference pricing + inefficiency detection |
| `src/signals/bayesian_engine.py` | Sequential Bayesian updates (log-space) |
| `src/strategy/edge_detector.py` | EV = p̂ - p edge filtering |
| `src/strategy/kelly_sizer.py` | 0.25x fractional Kelly sizing |
| `src/data/polymarket_client.py` | Polymarket REST + WebSocket |
| `src/data/news_feed.py` | News signal source → Bayesian signals |
| `src/execution/clob_executor.py` | CLOB limit/market order execution |
| `src/execution/order_manager.py` | Order lifecycle management |
| `src/risk/risk_manager.py` | Exposure / drawdown / concentration risk controls |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start trading (dry-run mode by default)
python scripts/run_trader.py

# Run backtest
python scripts/backtest.py
```

## Configuration

Edit `config.yaml`:

```yaml
polymarket:
  api_key: "your-api-key"
  private_key: "your-wallet-private-key"
  dry_run: true  # Set to false to enable live trading

strategy:
  min_edge: 0.03        # Minimum EV threshold
  kelly_fraction: 0.25  # Kelly multiplier (0.25 = quarter Kelly)
```

## Core Formulas

**LMSR Cost Function:**
$$C(\mathbf{q}) = b \cdot \ln\left(\sum_{i=1}^{n} e^{q_i/b}\right)$$

**Bayesian Update (log-space):**
$$\log P(H|\mathbf{D}) = \log P(H) + \sum_{k=1}^{t} \log P(D_k|H) - \log Z$$

**Fractional Kelly:**
$$f^* = \frac{\hat{p} - p}{1 - p} \times \text{kelly\_fraction}$$

## Risk Disclaimer

- Default `dry_run: true` — no real trades will be executed
- Prediction markets carry liquidity risk and model risk
- This system is for research and educational purposes only

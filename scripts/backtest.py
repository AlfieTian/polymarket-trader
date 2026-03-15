#!/usr/bin/env python3
"""
Simple Backtesting Framework

Simulates the trading strategy on historical price data.
"""

import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.bayesian_engine import BayesianEngine, Signal, SignalType
from src.strategy.edge_detector import EdgeDetector
from src.strategy.kelly_sizer import KellySizer

console = Console()


def generate_synthetic_market(
    n_steps: int = 100,
    true_prob: float = 0.65,
    noise: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic market data for backtesting.

    Args:
        n_steps: Number of time steps
        true_prob: True underlying probability
        noise: Market noise level
        seed: Random seed

    Returns:
        List of {step, market_price, signal_likelihood_yes, signal_likelihood_no}
    """
    rng = np.random.default_rng(seed)

    # Market price random walk around true prob with noise
    market_prices = [0.5]  # start at 50/50
    for _ in range(n_steps - 1):
        drift = 0.001 * (true_prob - market_prices[-1])  # mean reversion
        noise_val = rng.normal(0, noise * 0.1)
        new_price = np.clip(market_prices[-1] + drift + noise_val, 0.01, 0.99)
        market_prices.append(float(new_price))

    data = []
    for i, mp in enumerate(market_prices):
        # Generate noisy signals that reflect true probability
        ll_yes = true_prob + rng.normal(0, noise * 0.5)
        ll_no = (1 - true_prob) + rng.normal(0, noise * 0.5)
        ll_yes = float(np.clip(ll_yes, 0.01, 0.99))
        ll_no = float(np.clip(ll_no, 0.01, 0.99))

        data.append({
            "step": i,
            "market_price": mp,
            "signal_ll_yes": ll_yes,
            "signal_ll_no": ll_no,
        })

    return data


def run_backtest(
    true_prob: float = 0.65,
    n_steps: int = 100,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03,
    initial_bankroll: float = 1000.0,
):
    """Run a simple backtest."""
    console.print(f"\n[bold]📊 Backtest: true_prob={true_prob}, n_steps={n_steps}[/bold]")
    console.print(f"   Kelly fraction: {kelly_fraction}x, Min edge: {min_edge}")
    console.print(f"   Initial bankroll: ${initial_bankroll:.2f}\n")

    data = generate_synthetic_market(n_steps=n_steps, true_prob=true_prob)

    engine = BayesianEngine(prior_weight=0.5, min_observations=1)
    detector = EdgeDetector(min_edge=min_edge)
    sizer = KellySizer(
        kelly_fraction=kelly_fraction,
        max_position_usdc=initial_bankroll * 0.1,
        max_portfolio_usdc=initial_bankroll,
    )

    market_id = "backtest-market"
    engine.init_belief(market_id, data[0]["market_price"])

    bankroll = initial_bankroll
    n_trades = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    max_bankroll = bankroll
    max_drawdown = 0.0

    for i, d in enumerate(data):
        # Update beliefs
        signal = Signal(
            signal_type=SignalType.NEWS_SENTIMENT,
            likelihood_yes=d["signal_ll_yes"],
            likelihood_no=d["signal_ll_no"],
            confidence=0.6,
        )
        engine.update(market_id, signal)
        belief = engine.get_belief(market_id)

        # Detect edge
        opp = detector.detect(
            market_id=market_id,
            market_question="Backtest market",
            p_hat=belief.p_hat,
            market_price_yes=d["market_price"],
        )

        if opp is None:
            continue

        # Size position
        pos = sizer.calculate(
            market_id=market_id,
            side=opp.side,
            p_hat=opp.p_hat,
            market_price=opp.market_price,
            bankroll=bankroll,
        )

        if pos.position_usdc <= 0:
            continue

        n_trades += 1

        # Simulate outcome (simplified: resolve based on true probability)
        resolved_yes = np.random.random() < true_prob

        if opp.side == "YES":
            pnl = pos.position_usdc * (1 / opp.market_price - 1) if resolved_yes else -pos.position_usdc
        else:
            pnl = pos.position_usdc * (1 / (1 - opp.market_price) - 1) if not resolved_yes else -pos.position_usdc

        bankroll += pnl
        total_pnl += pnl

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        max_bankroll = max(max_bankroll, bankroll)
        drawdown = (max_bankroll - bankroll) / max_bankroll
        max_drawdown = max(max_drawdown, drawdown)

    # Results
    table = Table(title="📈 Backtest Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("True Probability", f"{true_prob:.2%}")
    table.add_row("Steps", str(n_steps))
    table.add_row("Trades", str(n_trades))
    table.add_row("Win Rate", f"{wins / n_trades:.1%}" if n_trades > 0 else "N/A")
    table.add_row("Total PnL", f"${total_pnl:+,.2f}")
    table.add_row("Final Bankroll", f"${bankroll:,.2f}")
    table.add_row("Return", f"{(bankroll / initial_bankroll - 1):+.1%}")
    table.add_row("Max Drawdown", f"{max_drawdown:.1%}")
    table.add_row("Sharpe (approx)", f"{total_pnl / (max_drawdown * initial_bankroll + 1):.2f}")

    console.print(table)


if __name__ == "__main__":
    # Run multiple scenarios
    run_backtest(true_prob=0.65, n_steps=200, kelly_fraction=0.25)
    run_backtest(true_prob=0.80, n_steps=200, kelly_fraction=0.25)
    run_backtest(true_prob=0.50, n_steps=200, kelly_fraction=0.25)  # no edge scenario
    run_backtest(true_prob=0.65, n_steps=200, kelly_fraction=1.0)   # full Kelly (dangerous!)

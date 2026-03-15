#!/usr/bin/env python3
"""
Bayesian Prior Calibration Analysis

Analyzes the trader's predictions vs market prices to check:
1. Systematic bias (are we consistently high or low?)
2. Edge distribution (how big are our detected edges?)
3. LLM signal impact (does LLM move p̂ meaningfully?)
4. p̂ stability over time (are signals bringing new information?)

Usage:
    python scripts/calibration_check.py [log_file]
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "logs/trader_live.log"


def parse_log(log_path: str) -> dict:
    """Parse trader log for calibration data."""
    data = {
        "edges": [],  # (market_id, side, p_hat, market_price, edge)
        "llm_signals": [],  # (market_id, p_hat_llm, confidence)
        "beliefs_init": [],  # (market_id, market_price, prior)
        "cycles": 0,
        "markets_per_cycle": [],
    }

    # Rich logger wraps lines — join all lines first, then parse
    # Read full file as single string
    with open(log_path, encoding="utf-8", errors="replace") as f:
        full_text = f.read()

    # Collapse multi-line log entries (lines starting with spaces are continuations)
    lines_joined = re.sub(r'\n\s{2,}', ' ', full_text)

    edge_pattern = re.compile(
        r"Edge detected: (\S+) (YES|NO) .{1,10}?=(\d+\.\d+).*?market=(\d+\.\d+).*?edge=\+(\d+\.\d+)"
    )
    llm_pattern = re.compile(
        r"LLM for (\S+): .{1,10}?=(\d+\.\d+).*?\(confidence=(\d+\.\d+)\)"
    )
    belief_pattern = re.compile(
        r"Initialized belief for (\S+):\s*market_price=(\d+\.\d+),\s*prior=(\d+\.\d+)"
    )
    cycle_pattern = re.compile(r"Cycle in (\d+\.\d+)s")
    tradeable_pattern = re.compile(r"Tradeable.*: (\d+)/(\d+)")

    for m in edge_pattern.finditer(lines_joined):
        data["edges"].append({
            "market_id": m.group(1),
            "side": m.group(2),
            "p_hat": float(m.group(3)),
            "market_price": float(m.group(4)),
            "edge": float(m.group(5)),
        })

    for m in llm_pattern.finditer(lines_joined):
        data["llm_signals"].append({
            "market_id": m.group(1),
            "p_hat_llm": float(m.group(2)),
            "confidence": float(m.group(3)),
        })

    for m in belief_pattern.finditer(lines_joined):
        data["beliefs_init"].append({
            "market_id": m.group(1),
            "market_price": float(m.group(2)),
            "prior": float(m.group(3)),
        })

    data["cycles"] = len(cycle_pattern.findall(lines_joined))
    for m in tradeable_pattern.finditer(lines_joined):
        data["markets_per_cycle"].append(int(m.group(1)))

    return data


def analyze(data: dict):
    """Run calibration analysis."""

    console.print("\n[bold]📊 Bayesian Prior Calibration Report[/bold]\n")

    # ─── Overview ─────────────────────────────────
    table = Table(title="Overview", show_header=False)
    table.add_row("Total cycles", str(data["cycles"]))
    table.add_row("Edge detections", str(len(data["edges"])))
    table.add_row("LLM signals", str(len(data["llm_signals"])))
    table.add_row("Markets initialized", str(len(data["beliefs_init"])))
    if data["markets_per_cycle"]:
        table.add_row("Avg tradeable/cycle", f"{np.mean(data['markets_per_cycle']):.1f}")
    console.print(table)

    # ─── Prior Bias Analysis ──────────────────────
    if data["beliefs_init"]:
        console.print("\n[bold]1. Prior Bias (init)[/bold]")
        inits = data["beliefs_init"]
        biases = [d["prior"] - d["market_price"] for d in inits]
        table = Table(title="Prior vs Market Price at Init")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Mean bias (prior - market)", f"{np.mean(biases):+.4f}")
        table.add_row("Median bias", f"{np.median(biases):+.4f}")
        table.add_row("Std dev", f"{np.std(biases):.4f}")
        table.add_row("Always positive?", "YES ⚠️" if all(b > 0 for b in biases) else "No ✅")
        console.print(table)

        if np.mean(biases) > 0.05:
            console.print("[yellow]⚠️  Prior systematically HIGHER than market — prior_weight < 1.0 pulls toward 0.5[/yellow]")
        elif np.mean(biases) < -0.05:
            console.print("[yellow]⚠️  Prior systematically LOWER than market[/yellow]")
        else:
            console.print("[green]✅ Prior bias is small[/green]")

    # ─── Edge Distribution ────────────────────────
    if data["edges"]:
        console.print("\n[bold]2. Edge Distribution[/bold]")
        edges = [d["edge"] for d in data["edges"]]
        table = Table(title="Detected Edge Statistics")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total edge detections", str(len(edges)))
        table.add_row("Mean edge", f"{np.mean(edges):.4f} ({np.mean(edges)*100:.1f}%)")
        table.add_row("Median edge", f"{np.median(edges):.4f}")
        table.add_row("Max edge", f"{np.max(edges):.4f}")
        table.add_row("Min edge", f"{np.min(edges):.4f}")

        # Edge by market
        by_market = defaultdict(list)
        for d in data["edges"]:
            by_market[d["market_id"]].append(d["edge"])

        table.add_row("Unique markets with edge", str(len(by_market)))
        console.print(table)

        # Per-market breakdown
        console.print("\n[bold]Edge by Market:[/bold]")
        market_table = Table()
        market_table.add_column("Market ID")
        market_table.add_column("Count")
        market_table.add_column("Avg Edge")
        market_table.add_column("Side")
        for mid, es in sorted(by_market.items(), key=lambda x: -len(x[1])):
            sides = set(d["side"] for d in data["edges"] if d["market_id"] == mid)
            market_table.add_row(
                mid,
                str(len(es)),
                f"{np.mean(es):.4f} ({np.mean(es)*100:.1f}%)",
                "/".join(sides),
            )
        console.print(market_table)

    # ─── LLM Signal Analysis ─────────────────────
    if data["llm_signals"]:
        console.print("\n[bold]3. LLM Signal Analysis[/bold]")

        by_market_llm = defaultdict(list)
        for d in data["llm_signals"]:
            by_market_llm[d["market_id"]].append(d)

        llm_table = Table(title="LLM Estimates by Market")
        llm_table.add_column("Market ID")
        llm_table.add_column("Signals")
        llm_table.add_column("Mean p̂")
        llm_table.add_column("Std p̂")
        llm_table.add_column("Mean Conf")

        for mid, signals in sorted(by_market_llm.items()):
            p_hats = [s["p_hat_llm"] for s in signals]
            confs = [s["confidence"] for s in signals]
            llm_table.add_row(
                mid,
                str(len(signals)),
                f"{np.mean(p_hats):.4f}",
                f"{np.std(p_hats):.4f}",
                f"{np.mean(confs):.2f}",
            )
        console.print(llm_table)

        # Check if LLM estimates are stable (low std) or noisy
        all_stds = []
        for signals in by_market_llm.values():
            if len(signals) > 1:
                all_stds.append(np.std([s["p_hat_llm"] for s in signals]))

        if all_stds:
            avg_std = np.mean(all_stds)
            if avg_std < 0.03:
                console.print(f"[yellow]⚠️  LLM estimates very stable (avg std={avg_std:.4f}) — may not be updating with new info[/yellow]")
            elif avg_std > 0.15:
                console.print(f"[yellow]⚠️  LLM estimates noisy (avg std={avg_std:.4f}) — high variance in predictions[/yellow]")
            else:
                console.print(f"[green]✅ LLM estimate variance reasonable (avg std={avg_std:.4f})[/green]")

    # ─── Recommendations ──────────────────────────
    console.print("\n[bold]4. Recommendations[/bold]")

    if data["beliefs_init"]:
        biases = [d["prior"] - d["market_price"] for d in data["beliefs_init"]]
        if abs(np.mean(biases)) > 0.05:
            console.print("  • [yellow]Increase prior_weight closer to 1.0 to reduce systematic bias[/yellow]")
            console.print(f"    Current bias: {np.mean(biases):+.4f} (prior_weight=0.7 pulls everything toward 0.5)")

    if data["edges"]:
        edges = [d["edge"] for d in data["edges"]]
        if np.mean(edges) > 0.10:
            console.print("  • [yellow]Avg edge >10% is suspiciously high — likely prior bias, not real alpha[/yellow]")
        if len(set(d["market_id"] for d in data["edges"])) < 3:
            console.print("  • [yellow]Very few unique markets generating edges — need more diverse markets[/yellow]")

    console.print("  • Track market resolutions to compute Brier Score (true calibration)")
    console.print("  • Compare LLM p̂ vs final outcome when markets resolve")
    console.print()


def main():
    log_path = Path(__file__).parent.parent / LOG_FILE
    if not log_path.exists():
        console.print(f"[red]Log file not found: {log_path}[/red]")
        sys.exit(1)

    data = parse_log(str(log_path))
    analyze(data)


if __name__ == "__main__":
    main()

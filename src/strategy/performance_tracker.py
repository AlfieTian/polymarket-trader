"""
Performance Tracker + Adaptive Strategy Adjuster

Records every closed trade, computes rolling metrics, and nudges
strategy parameters based on real P&L outcomes.

Adjustment rules (evaluated every EVAL_EVERY closed trades):
  - Win rate ≥ 60% AND avg_pnl > 0      → expand (kelly ×1.1, min_edge ×0.9)
  - Win rate ≤ 40% OR avg_pnl_per_trade < -0.30  → contract (kelly ×0.8, min_edge ×1.2)
  - Otherwise                             → hold

Parameters are clamped to safe ranges to avoid runaway adjustments.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

EVAL_EVERY = 5          # evaluate every N closed trades
ROLLING_WINDOW = 20     # use last N trades for metrics
HISTORY_FILE = Path(__file__).parent.parent.parent / "logs" / "trade_history.json"

# Safety clamps
KELLY_MIN, KELLY_MAX = 0.10, 0.50
EDGE_MIN,  EDGE_MAX  = 0.03, 0.15
POS_MIN,   POS_MAX   = 2.0,  15.0


@dataclass
class ClosedTrade:
    market_id: str
    side: str
    entry_price: float
    exit_price: float
    size_usdc: float
    realized_pnl: float       # USDC
    realized_pnl_pct: float   # fraction
    exit_reason: str
    closed_at: float = 0.0

    def __post_init__(self):
        if self.closed_at == 0.0:
            self.closed_at = time.time()

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > 0


class PerformanceTracker:
    def __init__(self):
        self._history: list[ClosedTrade] = []
        self._load()

    # ─── Persistence ──────────────────────────────────────────
    def _load(self):
        try:
            if HISTORY_FILE.exists():
                data = json.loads(HISTORY_FILE.read_text())
                self._history = [ClosedTrade(**t) for t in data]
                logger.info(f"📂 Loaded {len(self._history)} historical trades")
        except Exception as e:
            logger.warning(f"Could not load trade history: {e}")

    def _save(self):
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps([asdict(t) for t in self._history], indent=2)
            tmp_path = HISTORY_FILE.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
            # Atomic replace to avoid partial writes on crash
            os.replace(tmp_path, HISTORY_FILE)
        except Exception as e:
            logger.warning(f"Could not save trade history: {e}")

    # ─── Recording ────────────────────────────────────────────
    def record_close(self, trade: ClosedTrade):
        self._history.append(trade)
        self._save()
        result = "✅ WIN" if trade.is_win else "❌ LOSS"
        logger.info(
            f"📊 Trade closed [{result}] {trade.market_id} {trade.side} | "
            f"P&L: {trade.realized_pnl:+.2f} USDC ({trade.realized_pnl_pct:+.1%}) | "
            f"Reason: {trade.exit_reason}"
        )

    # ─── Metrics ──────────────────────────────────────────────
    def rolling_metrics(self, window: int = ROLLING_WINDOW) -> dict:
        recent = self._history[-window:] if len(self._history) >= window else self._history
        if not recent:
            return {}

        wins = sum(1 for t in recent if t.is_win)
        total_pnl = sum(t.realized_pnl for t in recent)
        pnls = [t.realized_pnl for t in recent]
        avg_pnl = total_pnl / len(recent)

        import statistics
        std = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
        sharpe = avg_pnl / std if std > 0 else 0.0

        return {
            "n_trades":    len(recent),
            "win_rate":    wins / len(recent),
            "total_pnl":   round(total_pnl, 3),
            "avg_pnl":     round(avg_pnl, 3),
            "sharpe":      round(sharpe, 2),
        }

    @property
    def total_closed(self) -> int:
        return len(self._history)

    def should_evaluate(self) -> bool:
        """True every EVAL_EVERY new trades."""
        return self.total_closed > 0 and self.total_closed % EVAL_EVERY == 0

    # ─── Adaptive Adjustment ──────────────────────────────────
    def suggest_adjustments(self, current: dict) -> dict | None:
        """
        Returns updated param dict if adjustment needed, else None.

        current = {
          "kelly_fraction": float,
          "min_edge": float,
          "max_position_usdc": float,
        }
        """
        m = self.rolling_metrics()
        if not m or m["n_trades"] < 3:
            return None

        win_rate = m["win_rate"]
        avg_pnl  = m["avg_pnl"]

        kelly   = current["kelly_fraction"]
        edge    = current["min_edge"]
        max_pos = current["max_position_usdc"]

        action = "hold"

        if win_rate >= 0.60 and avg_pnl > 0:
            # Performing well → expand carefully
            kelly   = min(kelly * 1.10, KELLY_MAX)
            edge    = max(edge  * 0.90, EDGE_MIN)
            max_pos = min(max_pos * 1.10, POS_MAX)
            action  = "expand"

        elif win_rate <= 0.40 or avg_pnl < -0.30:
            # Underperforming → pull back
            kelly   = max(kelly * 0.80, KELLY_MIN)
            edge    = min(edge  * 1.20, EDGE_MAX)
            max_pos = max(max_pos * 0.85, POS_MIN)
            action  = "contract"

        if action == "hold":
            logger.info(
                f"📈 Performance check: win={win_rate:.0%} avg_pnl={avg_pnl:+.2f} "
                f"Sharpe={m['sharpe']:.2f} → strategy UNCHANGED"
            )
            return None

        new_params = {
            "kelly_fraction":    round(kelly, 4),
            "min_edge":          round(edge, 4),
            "max_position_usdc": round(max_pos, 2),
        }
        logger.info(
            f"⚙️  Strategy AUTO-ADJUST [{action}] | "
            f"win={win_rate:.0%} avg_pnl={avg_pnl:+.2f} Sharpe={m['sharpe']:.2f}\n"
            f"   kelly: {current['kelly_fraction']:.3f}→{new_params['kelly_fraction']:.3f} | "
            f"min_edge: {current['min_edge']:.3f}→{new_params['min_edge']:.3f} | "
            f"max_pos: {current['max_position_usdc']:.1f}→{new_params['max_position_usdc']:.1f}"
        )
        return new_params

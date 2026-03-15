"""
Position Manager

Tracks open positions and decides when to exit:
- Profit target: exit when price moves in our favor by target_pct
- Stop loss: exit when loss exceeds stop_pct
- Time-based: exit N hours before resolution
- Edge reversal: exit if LLM reverses its view
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent.parent / "logs" / "positions_state.json"

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    EDGE_REVERSAL = "edge_reversal"
    PRE_RESOLUTION = "pre_resolution"
    MANUAL = "manual"


@dataclass
class Position:
    """An open position in a prediction market."""

    market_id: str
    condition_id: str
    token_id: str
    side: str          # "YES" or "NO"
    entry_price: float
    size: float        # shares
    size_usdc: float   # USDC invested
    p_hat_at_entry: float
    market_price_at_entry: float
    opened_at: float = field(default_factory=time.time)
    end_date: str = ""  # ISO timestamp of market resolution
    exit_retries: int = 0          # failed exit attempts so far
    exit_price_override: float = 0.0  # lowered price after retries (0 = use market price)
    force_close_failed: bool = False  # exit retries exhausted; manual intervention likely

    @property
    def current_value(self) -> float:
        """Current mark-to-market value (updated externally)."""
        return self._current_price * self.size if hasattr(self, "_current_price") else self.size_usdc

    def update_price(self, current_price: float):
        self._current_price = current_price

    @property
    def unrealized_pnl(self) -> float:
        if not hasattr(self, "_current_price"):
            return 0.0
        return (self._current_price - self.entry_price) * self.size

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (getattr(self, "_current_price", self.entry_price) - self.entry_price) / self.entry_price

    @property
    def age_hours(self) -> float:
        return (time.time() - self.opened_at) / 3600

    @property
    def hours_to_resolution(self) -> float:
        """Hours until market resolves. inf if no end_date."""
        if not self.end_date:
            return float("inf")
        try:
            from datetime import datetime, timezone
            end = datetime.fromisoformat(self.end_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = (end - now).total_seconds() / 3600
            return max(0, delta)
        except Exception:
            return float("inf")


@dataclass
class ExitSignal:
    position: Position
    reason: ExitReason
    current_price: float
    message: str


class PositionManager:
    """Manages open positions and generates exit signals.

    Exit conditions (checked in order):
    1. Stop loss: unrealized loss > stop_loss_pct
    2. Profit target: unrealized gain > profit_target_pct
    3. Edge reversal: current p_hat no longer supports position
    4. Pre-resolution: close N hours before market resolves
    """

    def __init__(
        self,
        profit_target_pct: float = 0.08,
        stop_loss_pct: float = 0.15,
        pre_resolution_hours: float = 2.0,
        min_edge_to_hold: float = 0.02,
        exit_max_retries: int = 5,
        exit_price_step: float = 0.01,
    ):
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.pre_resolution_hours = pre_resolution_hours
        self.min_edge_to_hold = min_edge_to_hold
        self.exit_max_retries = exit_max_retries
        self.exit_price_step = exit_price_step
        self._positions: dict[str, Position] = {}
        self._load_state()

    # ─── Persistence ──────────────────────────────────────────
    def _load_state(self):
        """Restore positions from disk after restart."""
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text())
                for d in data:
                    p = Position(**d)
                    self._positions[p.market_id] = p
                logger.info(f"📂 Restored {len(self._positions)} positions from disk")
        except Exception as e:
            logger.warning(f"Could not restore position state: {e}")

    def _save_state(self):
        """Persist current positions to disk."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            def _pos_dict(p):
                d = asdict(p)
                d.pop("_current_price", None)  # non-serializable runtime field
                return d
            payload = json.dumps([_pos_dict(p) for p in self._positions.values()], indent=2)
            tmp_path = STATE_FILE.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
            # Atomic replace to avoid partial writes on crash
            os.replace(tmp_path, STATE_FILE)
        except Exception as e:
            logger.warning(f"Could not save position state: {e}")

    def add_position(self, position: Position) -> None:
        """Record a new open position.

        If a position for this market_id already exists, accumulate into
        the existing entry (weighted-average price, summed size) instead of
        overwriting — prevents orphaned on-chain tokens when the bot buys
        the same market twice across cycles.
        """
        existing = self._positions.get(position.market_id)
        if existing is not None:
            # Accumulate: weighted-average entry price, sum sizes
            total_usdc = existing.size_usdc + position.size_usdc
            total_size = existing.size + position.size
            if total_size > 0:
                existing.entry_price = (
                    existing.entry_price * existing.size + position.entry_price * position.size
                ) / total_size
            existing.size = total_size
            existing.size_usdc = total_usdc
            # Keep the earlier opened_at and reset exit retry state
            existing.exit_retries = 0
            existing.exit_price_override = 0.0
            existing.force_close_failed = False
            self._save_state()
            logger.warning(
                f"📌 Accumulated into existing position: {position.market_id} {position.side} "
                f"now {existing.size:.2f} shares @ ${existing.entry_price:.3f} "
                f"(${existing.size_usdc:.2f} USDC total)"
            )
            return

        self._positions[position.market_id] = position
        self._save_state()
        logger.info(
            f"📌 New position: {position.market_id} {position.side} "
            f"{position.size:.2f} shares @ ${position.entry_price:.3f} "
            f"(${position.size_usdc:.2f} USDC)"
        )

    def close_position(self, market_id: str) -> Position | None:
        pos = self._positions.pop(market_id, None)
        if pos:
            self._save_state()
        return pos

    def get_position(self, market_id: str) -> Position | None:
        return self._positions.get(market_id)

    @property
    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    @property
    def total_exposure_usdc(self) -> float:
        return sum(p.size_usdc for p in self._positions.values())

    def check_exits(
        self,
        current_prices: dict[str, float],
        current_p_hats: dict[str, float],
    ) -> list[ExitSignal]:
        """Check all positions for exit conditions.

        Args:
            current_prices: {market_id → current YES price}
            current_p_hats: {market_id → current Bayesian estimate}

        Returns:
            List of ExitSignals for positions that should be closed
        """
        signals = []

        for market_id, pos in list(self._positions.items()):
            current_price = current_prices.get(market_id)
            if current_price is None:
                continue

            # Adjust price for NO positions (NO price = 1 - YES price)
            if pos.side == "NO":
                effective_price = 1.0 - current_price
            else:
                effective_price = current_price

            pos.update_price(effective_price)
            p_hat = current_p_hats.get(market_id)

            exit_signal = None

            # 0. Pre-resolution (HIGHEST priority — must exit before market resolves
            #    regardless of edge or P&L; past end_date also triggers this)
            if pos.hours_to_resolution <= self.pre_resolution_hours:
                exit_signal = ExitSignal(
                    position=pos,
                    reason=ExitReason.PRE_RESOLUTION,
                    current_price=effective_price,
                    message=(
                        f"Market resolves in {pos.hours_to_resolution:.1f}h "
                        f"(closing {self.pre_resolution_hours:.0f}h before resolution)"
                    ),
                )

            # 1. Stop loss
            elif pos.unrealized_pnl_pct <= -self.stop_loss_pct:
                exit_signal = ExitSignal(
                    position=pos,
                    reason=ExitReason.STOP_LOSS,
                    current_price=effective_price,
                    message=(
                        f"Stop loss triggered: {pos.unrealized_pnl_pct:.1%} loss "
                        f"(limit: -{self.stop_loss_pct:.0%})"
                    ),
                )

            # 2. Profit target
            elif pos.unrealized_pnl_pct >= self.profit_target_pct:
                exit_signal = ExitSignal(
                    position=pos,
                    reason=ExitReason.PROFIT_TARGET,
                    current_price=effective_price,
                    message=(
                        f"Profit target hit: +{pos.unrealized_pnl_pct:.1%} "
                        f"(target: +{self.profit_target_pct:.0%})"
                    ),
                )

            # 3. Edge reversal (our model no longer agrees with this position)
            elif p_hat is not None:
                if pos.side == "YES":
                    current_edge = p_hat - effective_price
                else:
                    current_edge = (1 - p_hat) - effective_price

                if current_edge < self.min_edge_to_hold:
                    exit_signal = ExitSignal(
                        position=pos,
                        reason=ExitReason.EDGE_REVERSAL,
                        current_price=effective_price,
                        message=(
                            f"Edge reversed: current edge {current_edge:.3f} "
                            f"< min_hold {self.min_edge_to_hold:.3f}"
                        ),
                    )

            if exit_signal:
                logger.info(
                    f"🚪 Exit signal [{exit_signal.reason.value}] {market_id} {pos.side}: "
                    f"{exit_signal.message}"
                )
                signals.append(exit_signal)

        return signals

    def summary(self) -> dict:
        """Portfolio summary."""
        return {
            "open_positions": len(self._positions),
            "total_exposure_usdc": self.total_exposure_usdc,
            "positions": [
                {
                    "market_id": p.market_id,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "size_usdc": p.size_usdc,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "age_hours": round(p.age_hours, 2),
                }
                for p in self._positions.values()
            ],
        }

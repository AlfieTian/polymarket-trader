"""
Fractional Kelly Position Sizing

Kelly criterion: f* = (p̂ - p) / (1 - p)   for YES bets
                 f* = (p - p̂) / p          for NO bets

Document note: "NEVER full Kelly on 5min markets!"
Default: 0.25x Kelly (quarter Kelly) for risk management.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Calculated position size."""

    market_id: str
    side: str
    kelly_full: float  # full Kelly fraction
    kelly_fraction_used: float  # multiplier applied
    kelly_adjusted: float  # kelly_full * kelly_fraction
    position_usdc: float  # final USDC amount
    bankroll: float  # available bankroll
    capped: bool = False  # True if capped by max limits
    reason: str = ""


class KellySizer:
    """Fractional Kelly position sizing with safety limits.

    Uses quarter Kelly by default (0.25x) to reduce variance
    at the cost of lower expected growth.

    Reference: QR-PM-2026-0041 §6 — "NEVER full Kelly on 5min markets!"
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_usdc: float = 100.0,
        max_portfolio_usdc: float = 1000.0,
    ):
        """
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            max_position_usdc: Max USDC per single market position
            max_portfolio_usdc: Max total USDC across all positions
        """
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")

        self.kelly_fraction = kelly_fraction
        self.max_position_usdc = max_position_usdc
        self.max_portfolio_usdc = max_portfolio_usdc
        self._current_exposure: dict[str, float] = {}

    @property
    def total_exposure(self) -> float:
        """Total current exposure across all markets."""
        return sum(self._current_exposure.values())

    @property
    def remaining_capacity(self) -> float:
        """Remaining USDC capacity before hitting portfolio limit."""
        return max(0, self.max_portfolio_usdc - self.total_exposure)

    def calculate(
        self,
        market_id: str,
        side: str,
        p_hat: float,
        market_price: float,
        bankroll: float,
    ) -> PositionSize:
        """Calculate position size using fractional Kelly.

        For YES bets: f* = (p̂ - p) / (1 - p)
        For NO bets:  f* = (p - p̂) / p

        Args:
            market_id: Market identifier
            side: "YES" or "NO"
            p_hat: Our estimated probability of YES
            market_price: Current market price for this side
            bankroll: Available bankroll in USDC

        Returns:
            PositionSize with calculated amounts
        """
        # Calculate full Kelly fraction
        if side == "YES":
            # f* = (p̂ - p) / (1 - p)
            if market_price >= 1.0:
                kelly_full = 0.0
            else:
                kelly_full = (p_hat - market_price) / (1.0 - market_price)
        elif side == "NO":
            # f* = ((1 - p̂) - (1 - p)) / (1 - (1 - p)) = (p - p̂) / p
            if market_price <= 0.0:
                kelly_full = 0.0
            else:
                kelly_full = (market_price - p_hat) / market_price
        else:
            raise ValueError(f"side must be 'YES' or 'NO', got '{side}'")

        # Negative Kelly = negative edge, don't trade
        if kelly_full <= 0:
            return PositionSize(
                market_id=market_id,
                side=side,
                kelly_full=kelly_full,
                kelly_fraction_used=self.kelly_fraction,
                kelly_adjusted=0.0,
                position_usdc=0.0,
                bankroll=bankroll,
                reason="negative_edge",
            )

        # Apply fractional Kelly
        kelly_adjusted = kelly_full * self.kelly_fraction
        position_usdc = kelly_adjusted * bankroll

        # Apply caps
        capped = False
        original = position_usdc

        # Cap 1: max per-market position
        if position_usdc > self.max_position_usdc:
            position_usdc = self.max_position_usdc
            capped = True

        # Cap 2: remaining portfolio capacity
        remaining = self.remaining_capacity
        if position_usdc > remaining:
            position_usdc = remaining
            capped = True

        # Cap 3: can't bet more than bankroll
        if position_usdc > bankroll:
            position_usdc = bankroll
            capped = True

        position_usdc = max(0, position_usdc)

        reason = ""
        if capped:
            reason = f"capped from ${original:.2f} to ${position_usdc:.2f}"

        result = PositionSize(
            market_id=market_id,
            side=side,
            kelly_full=kelly_full,
            kelly_fraction_used=self.kelly_fraction,
            kelly_adjusted=kelly_adjusted,
            position_usdc=position_usdc,
            bankroll=bankroll,
            capped=capped,
            reason=reason,
        )

        logger.info(
            f"Kelly sizing {market_id} {side}: "
            f"f*={kelly_full:.4f}, f*×{self.kelly_fraction}={kelly_adjusted:.4f}, "
            f"size=${position_usdc:.2f} USDC"
            f"{' (CAPPED)' if capped else ''}"
        )

        return result

    def record_position(self, market_id: str, usdc: float) -> None:
        """Record a new position for portfolio tracking."""
        self._current_exposure[market_id] = (
            self._current_exposure.get(market_id, 0) + usdc
        )

    def close_position(self, market_id: str) -> None:
        """Remove a market from exposure tracking."""
        self._current_exposure.pop(market_id, None)

    def reset_positions(self) -> None:
        """Clear exposure tracking so it can be rebuilt from persisted positions."""
        self._current_exposure.clear()

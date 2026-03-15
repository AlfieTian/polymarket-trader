"""
Risk Management Module

Pre-trade and portfolio-level risk controls:
- Max exposure per market
- Max portfolio exposure
- Daily loss limit (halt trading if exceeded)
- Concentration check
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check."""

    approved: bool
    reason: str = ""
    market_exposure: float = 0.0
    portfolio_exposure: float = 0.0
    daily_pnl: float = 0.0


class RiskManager:
    """Portfolio-level risk management.

    Enforces:
    1. Max exposure per market (USDC)
    2. Max total portfolio exposure (USDC)
    3. Max daily loss limit (halt trading)
    4. Concentration limit (no single market > X% of portfolio)
    """

    def __init__(
        self,
        max_position_usdc: float = 100.0,
        max_portfolio_usdc: float = 1000.0,
        max_daily_loss_usdc: float = 200.0,
        max_market_concentration: float = 0.3,
    ):
        self.max_position_usdc = max_position_usdc
        self.max_portfolio_usdc = max_portfolio_usdc
        self.max_daily_loss_usdc = max_daily_loss_usdc
        self.max_market_concentration = max_market_concentration

        self._positions: dict[str, float] = {}  # market_id → USDC exposure
        self._daily_pnl: float = 0.0
        self._daily_reset_date: str = ""
        self._halted: bool = False

    @property
    def total_exposure(self) -> float:
        return sum(self._positions.values())

    @property
    def is_halted(self) -> bool:
        return self._halted

    def _check_daily_reset(self) -> None:
        """Reset daily PnL at day boundary."""
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_pnl = 0.0
            self._daily_reset_date = today
            self._halted = False
            logger.info(f"Daily PnL reset for {today}")

    def record_pnl(self, amount: float) -> None:
        """Record realized PnL (positive = profit, negative = loss)."""
        self._check_daily_reset()
        self._daily_pnl += amount

        if self._daily_pnl <= -self.max_daily_loss_usdc:
            self._halted = True
            logger.warning(
                f"🚨 TRADING HALTED — Daily loss ${abs(self._daily_pnl):.2f} "
                f"exceeds limit ${self.max_daily_loss_usdc:.2f}"
            )

    def validate_trade(
        self,
        market_id: str,
        size_usdc: float,
        direction: str = "BUY",
    ) -> RiskCheckResult:
        """Pre-trade risk validation.

        Args:
            market_id: Market identifier
            size_usdc: Proposed trade size in USDC
            direction: "BUY" or "SELL"

        Returns:
            RiskCheckResult with approval status
        """
        self._check_daily_reset()

        # Check 1: Trading halt
        if self._halted:
            return RiskCheckResult(
                approved=False,
                reason=f"Trading halted — daily loss ${abs(self._daily_pnl):.2f} "
                f">= limit ${self.max_daily_loss_usdc:.2f}",
                daily_pnl=self._daily_pnl,
            )

        current_market_exposure = self._positions.get(market_id, 0.0)
        new_market_exposure = current_market_exposure + size_usdc
        new_portfolio_exposure = self.total_exposure + size_usdc

        # Check 2: Per-market exposure
        if new_market_exposure > self.max_position_usdc:
            return RiskCheckResult(
                approved=False,
                reason=f"Market exposure ${new_market_exposure:.2f} would exceed "
                f"limit ${self.max_position_usdc:.2f}",
                market_exposure=current_market_exposure,
                portfolio_exposure=self.total_exposure,
            )

        # Check 3: Portfolio exposure
        if new_portfolio_exposure > self.max_portfolio_usdc:
            return RiskCheckResult(
                approved=False,
                reason=f"Portfolio exposure ${new_portfolio_exposure:.2f} would exceed "
                f"limit ${self.max_portfolio_usdc:.2f}",
                market_exposure=current_market_exposure,
                portfolio_exposure=self.total_exposure,
            )

        # Check 4: Concentration
        if new_portfolio_exposure > 0:
            concentration = new_market_exposure / new_portfolio_exposure
            if concentration > self.max_market_concentration:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Market concentration {concentration:.1%} would exceed "
                    f"limit {self.max_market_concentration:.1%}",
                    market_exposure=current_market_exposure,
                    portfolio_exposure=self.total_exposure,
                )

        return RiskCheckResult(
            approved=True,
            market_exposure=new_market_exposure,
            portfolio_exposure=new_portfolio_exposure,
            daily_pnl=self._daily_pnl,
        )

    def record_position(self, market_id: str, size_usdc: float) -> None:
        """Record a new or increased position."""
        self._positions[market_id] = self._positions.get(market_id, 0.0) + size_usdc
        logger.info(
            f"Position recorded: {market_id} +${size_usdc:.2f} "
            f"(total market: ${self._positions[market_id]:.2f}, "
            f"portfolio: ${self.total_exposure:.2f})"
        )

    def close_position(self, market_id: str) -> None:
        """Remove a market from position tracking."""
        removed = self._positions.pop(market_id, 0)
        if removed:
            logger.info(f"Position closed: {market_id} (was ${removed:.2f})")

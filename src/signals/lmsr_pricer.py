"""
LMSR (Logarithmic Market Scoring Rule) Pricing Engine

Implements the Hanson LMSR for reference pricing and inefficiency detection.
Core formulas from QR-PM-2026-0041 §1-3.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LMSRState:
    """Current state of the LMSR market."""

    quantities: np.ndarray  # outcome quantities vector
    b: float  # liquidity parameter
    n_outcomes: int

    @property
    def prices(self) -> np.ndarray:
        """Current implied prices (softmax)."""
        return lmsr_prices(self.quantities, self.b)

    @property
    def cost(self) -> float:
        """Current cost function value."""
        return lmsr_cost(self.quantities, self.b)

    @property
    def max_loss(self) -> float:
        """Market maker maximum loss bound."""
        return self.b * np.log(self.n_outcomes)


def _log_sum_exp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp.

    Uses the max-shift trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


def lmsr_cost(q: np.ndarray, b: float) -> float:
    """LMSR cost function: C(q) = b * ln(sum(exp(qi/b)))

    Args:
        q: Outcome quantities vector
        b: Liquidity parameter (b > 0)

    Returns:
        Cost function value in USDC
    """
    if b <= 0:
        raise ValueError(f"Liquidity parameter b must be positive, got {b}")
    return b * _log_sum_exp(q / b)


def lmsr_prices(q: np.ndarray, b: float) -> np.ndarray:
    """LMSR price function (softmax): pi(q) = exp(qi/b) / sum(exp(qj/b))

    Prices always sum to 1 and lie in (0, 1).

    Args:
        q: Outcome quantities vector
        b: Liquidity parameter

    Returns:
        Price vector (probability-like)
    """
    if b <= 0:
        raise ValueError(f"Liquidity parameter b must be positive, got {b}")
    shifted = q / b - np.max(q / b)  # numerical stability
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted)


def trade_cost(q: np.ndarray, b: float, outcome_idx: int, delta: float) -> float:
    """Cost of buying delta shares of outcome_idx.

    Cost = C(q + delta*e_i) - C(q)

    Args:
        q: Current quantities vector
        b: Liquidity parameter
        outcome_idx: Which outcome to trade (0-indexed)
        delta: Number of shares (positive = buy, negative = sell)

    Returns:
        Trade cost in USDC
    """
    q_new = q.copy()
    q_new[outcome_idx] += delta
    return lmsr_cost(q_new, b) - lmsr_cost(q, b)


def max_market_maker_loss(b: float, n: int) -> float:
    """Maximum market maker loss: L_max = b * ln(n)

    Args:
        b: Liquidity parameter
        n: Number of outcomes

    Returns:
        Maximum possible loss in USDC
    """
    return b * np.log(n)


@dataclass
class InefficiencySignal:
    """Detected pricing inefficiency between LMSR and CLOB."""

    market_id: str
    outcome_idx: int
    lmsr_price: float
    clob_price: float
    spread: float  # lmsr_price - clob_price
    abs_spread: float

    @property
    def direction(self) -> str:
        """Suggested trade direction."""
        return "BUY" if self.spread > 0 else "SELL"


class LMSRPricer:
    """LMSR-based reference pricer for inefficiency detection."""

    def __init__(self, b: float = 100_000.0):
        self.b = b
        self._states: dict[str, LMSRState] = {}

    def init_market(self, market_id: str, n_outcomes: int = 2) -> LMSRState:
        """Initialize a market with equal prior probabilities."""
        q = np.zeros(n_outcomes)
        state = LMSRState(quantities=q, b=self.b, n_outcomes=n_outcomes)
        self._states[market_id] = state
        logger.info(
            f"Initialized LMSR market {market_id}: n={n_outcomes}, b={self.b}, "
            f"L_max=${state.max_loss:,.2f}"
        )
        return state

    def update_from_trade(
        self, market_id: str, outcome_idx: int, delta: float
    ) -> LMSRState:
        """Simulate a trade and update internal state."""
        state = self._states[market_id]
        cost = trade_cost(state.quantities, state.b, outcome_idx, delta)
        state.quantities[outcome_idx] += delta
        logger.debug(
            f"Market {market_id}: traded {delta:+.2f} on outcome {outcome_idx}, "
            f"cost=${cost:.4f}, new prices={state.prices}"
        )
        return state

    def detect_inefficiency(
        self,
        market_id: str,
        clob_prices: list[float],
        min_spread: float = 0.03,
    ) -> list[InefficiencySignal]:
        """Compare LMSR implied prices to CLOB market prices.

        Args:
            market_id: Market identifier
            clob_prices: Current CLOB prices for each outcome
            min_spread: Minimum absolute spread to flag as inefficiency

        Returns:
            List of detected inefficiency signals
        """
        if market_id not in self._states:
            self.init_market(market_id, n_outcomes=len(clob_prices))

        state = self._states[market_id]
        lmsr_p = state.prices
        signals = []

        for i, (lp, cp) in enumerate(zip(lmsr_p, clob_prices)):
            spread = lp - cp
            abs_spread = abs(spread)
            if abs_spread >= min_spread:
                sig = InefficiencySignal(
                    market_id=market_id,
                    outcome_idx=i,
                    lmsr_price=lp,
                    clob_price=cp,
                    spread=spread,
                    abs_spread=abs_spread,
                )
                signals.append(sig)
                logger.info(
                    f"Inefficiency detected: {market_id} outcome {i} — "
                    f"LMSR={lp:.4f} vs CLOB={cp:.4f}, spread={spread:+.4f} → {sig.direction}"
                )

        return signals

"""
Sequential Bayesian Signal Processing Engine

Real-time belief updating using log-space arithmetic for numerical stability.
Reference: QR-PM-2026-0041 §5 — "The trader who updates fastest and most accurately wins. Period."
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Types of signals that can update beliefs."""

    NEWS_SENTIMENT = "news_sentiment"
    HISTORICAL_BASE_RATE = "historical_base_rate"
    MARKET_MOMENTUM = "market_momentum"
    EXPERT_FORECAST = "expert_forecast"
    SOCIAL_SENTIMENT = "social_sentiment"


@dataclass
class Signal:
    """A single observation/signal for Bayesian updating."""

    signal_type: SignalType
    likelihood_yes: float  # P(D|H=YES) — how likely this data if outcome is YES
    likelihood_no: float  # P(D|H=NO) — how likely this data if outcome is NO
    confidence: float = 1.0  # 0-1 weight for this signal's reliability
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def log_likelihood_ratio(self) -> float:
        """Log likelihood ratio: log(P(D|YES)) - log(P(D|NO))"""
        return np.log(self.likelihood_yes + 1e-15) - np.log(self.likelihood_no + 1e-15)


@dataclass
class BeliefState:
    """Current belief state for a market."""

    market_id: str
    log_prior_yes: float  # log P(H=YES)
    log_prior_no: float  # log P(H=NO)
    log_posterior_yes: float  # log P(H=YES|D1,...,Dt)
    log_posterior_no: float  # log P(H=NO|D1,...,Dt)
    n_updates: int = 0
    update_history: list[dict] = field(default_factory=list)
    last_update_ms: float = 0.0  # latency of last update

    @property
    def p_yes(self) -> float:
        """Posterior probability of YES outcome."""
        # Softmax of [log_posterior_yes, log_posterior_no]
        max_log = max(self.log_posterior_yes, self.log_posterior_no)
        exp_yes = np.exp(self.log_posterior_yes - max_log)
        exp_no = np.exp(self.log_posterior_no - max_log)
        return float(exp_yes / (exp_yes + exp_no))

    @property
    def p_no(self) -> float:
        """Posterior probability of NO outcome."""
        return 1.0 - self.p_yes

    @property
    def p_hat(self) -> float:
        """Alias for p_yes — our estimated true probability."""
        return self.p_yes


class BayesianEngine:
    """Sequential Bayesian updater for prediction market beliefs.

    Uses log-space arithmetic to prevent numerical underflow when
    accumulating many small likelihoods:

        log P(H|D1,...,Dt) = log P(H) + sum(log P(Dk|H)) - log Z

    Latency target: <50ms per update cycle.
    """

    def __init__(self, prior_weight: float = 0.7, min_observations: int = 3):
        """
        Args:
            prior_weight: Weight given to market price as prior (0-1).
                          Higher = more anchored to market consensus.
            min_observations: Minimum signals before trading is allowed.
        """
        self.prior_weight = prior_weight
        self.min_observations = min_observations
        self._beliefs: dict[str, BeliefState] = {}

    def init_belief(self, market_id: str, market_price: float) -> BeliefState:
        """Initialize belief state from current market price.

        The market price serves as our prior. We can optionally blend
        with a uniform prior using prior_weight.

        Args:
            market_id: Market identifier
            market_price: Current YES price on Polymarket (0-1)

        Returns:
            Initialized belief state
        """
        # Blend market price with uniform prior
        p_prior = self.prior_weight * market_price + (1 - self.prior_weight) * 0.5
        p_prior = np.clip(p_prior, 0.001, 0.999)  # avoid log(0)

        log_yes = np.log(p_prior)
        log_no = np.log(1.0 - p_prior)

        belief = BeliefState(
            market_id=market_id,
            log_prior_yes=log_yes,
            log_prior_no=log_no,
            log_posterior_yes=log_yes,
            log_posterior_no=log_no,
        )
        self._beliefs[market_id] = belief
        logger.info(
            f"Initialized belief for {market_id}: "
            f"market_price={market_price:.4f}, prior={p_prior:.4f}"
        )
        return belief

    def update(self, market_id: str, signal: Signal) -> BeliefState:
        """Apply a single Bayesian update with a new signal.

        P(H|D) ∝ P(D|H) * P(H)

        In log space:
            log P(H|D) = log P(H) + log P(D|H)  (unnormalized)

        Normalization is applied via log-sum-exp.

        Args:
            market_id: Market identifier
            signal: New signal observation

        Returns:
            Updated belief state
        """
        if market_id not in self._beliefs:
            raise KeyError(
                f"No belief initialized for {market_id}. Call init_belief() first."
            )

        t_start = time.perf_counter_ns()
        belief = self._beliefs[market_id]

        # Weight likelihood by signal confidence
        weighted_ll_yes = signal.confidence * np.log(signal.likelihood_yes + 1e-15)
        weighted_ll_no = signal.confidence * np.log(signal.likelihood_no + 1e-15)

        # Bayesian update in log space (unnormalized)
        log_yes_new = belief.log_posterior_yes + weighted_ll_yes
        log_no_new = belief.log_posterior_no + weighted_ll_no

        # Normalize via log-sum-exp
        log_z = _log_sum_exp_pair(log_yes_new, log_no_new)
        belief.log_posterior_yes = log_yes_new - log_z
        belief.log_posterior_no = log_no_new - log_z

        belief.n_updates += 1
        elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
        belief.last_update_ms = elapsed_ms

        belief.update_history.append(
            {
                "signal_type": signal.signal_type.value,
                "llr": signal.log_likelihood_ratio,
                "confidence": signal.confidence,
                "p_yes_after": belief.p_yes,
                "elapsed_ms": elapsed_ms,
                "timestamp": signal.timestamp,
            }
        )

        logger.debug(
            f"Updated {market_id} with {signal.signal_type.value}: "
            f"p_yes={belief.p_yes:.4f} ({elapsed_ms:.2f}ms)"
        )

        return belief

    def batch_update(self, market_id: str, signals: list[Signal]) -> BeliefState:
        """Apply multiple signals sequentially.

        log P(H|D1,...,Dt) = log P(H) + sum_k(log P(Dk|H)) - log Z
        """
        for signal in signals:
            self.update(market_id, signal)
        return self._beliefs[market_id]

    def get_belief(self, market_id: str) -> BeliefState | None:
        """Get current belief state for a market."""
        return self._beliefs.get(market_id)

    def is_tradeable(self, market_id: str) -> bool:
        """Check if we have enough observations to trade."""
        belief = self._beliefs.get(market_id)
        if belief is None:
            return False
        return belief.n_updates >= self.min_observations

    def reset(self, market_id: str) -> None:
        """Reset beliefs for a market."""
        if market_id in self._beliefs:
            del self._beliefs[market_id]


def _log_sum_exp_pair(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    max_val = max(a, b)
    return max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val))

"""Tests for LMSR Pricer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.lmsr_pricer import (
    LMSRPricer,
    lmsr_cost,
    lmsr_prices,
    max_market_maker_loss,
    trade_cost,
)


class TestLMSRFunctions:
    def test_prices_sum_to_one(self):
        q = np.array([100.0, 200.0, 50.0])
        prices = lmsr_prices(q, b=1000)
        assert abs(np.sum(prices) - 1.0) < 1e-10

    def test_equal_quantities_equal_prices(self):
        q = np.array([0.0, 0.0])
        prices = lmsr_prices(q, b=100)
        assert abs(prices[0] - 0.5) < 1e-10
        assert abs(prices[1] - 0.5) < 1e-10

    def test_higher_quantity_higher_price(self):
        q = np.array([100.0, 0.0])
        prices = lmsr_prices(q, b=100)
        assert prices[0] > prices[1]

    def test_cost_increases_with_quantity(self):
        q1 = np.array([0.0, 0.0])
        q2 = np.array([10.0, 0.0])
        assert lmsr_cost(q2, b=100) > lmsr_cost(q1, b=100)

    def test_trade_cost_positive_for_buy(self):
        q = np.array([0.0, 0.0])
        cost = trade_cost(q, b=100, outcome_idx=0, delta=10)
        assert cost > 0

    def test_trade_cost_negative_for_sell(self):
        q = np.array([100.0, 0.0])
        cost = trade_cost(q, b=100, outcome_idx=0, delta=-10)
        assert cost < 0

    def test_max_loss_binary(self):
        loss = max_market_maker_loss(b=100000, n=2)
        assert abs(loss - 100000 * np.log(2)) < 0.01

    def test_numerical_stability_large_quantities(self):
        """Should not overflow with large quantities."""
        q = np.array([1e6, 1e6])
        prices = lmsr_prices(q, b=1000)
        assert abs(np.sum(prices) - 1.0) < 1e-10

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError):
            lmsr_cost(np.array([0.0, 0.0]), b=0)
        with pytest.raises(ValueError):
            lmsr_cost(np.array([0.0, 0.0]), b=-1)


class TestLMSRPricer:
    def test_init_market(self):
        pricer = LMSRPricer(b=1000)
        state = pricer.init_market("test", n_outcomes=2)
        assert len(state.prices) == 2
        assert abs(state.prices[0] - 0.5) < 1e-10

    def test_detect_inefficiency(self):
        pricer = LMSRPricer(b=1000)
        pricer.init_market("test", n_outcomes=2)

        # LMSR says 50/50, CLOB says 60/40 → inefficiency
        signals = pricer.detect_inefficiency("test", [0.6, 0.4], min_spread=0.05)
        assert len(signals) == 2

    def test_no_inefficiency_when_aligned(self):
        pricer = LMSRPricer(b=1000)
        pricer.init_market("test", n_outcomes=2)

        signals = pricer.detect_inefficiency("test", [0.5, 0.5], min_spread=0.01)
        assert len(signals) == 0

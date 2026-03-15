"""Tests for Kelly Position Sizer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.kelly_sizer import KellySizer


class TestKellySizer:
    def test_yes_positive_edge(self):
        sizer = KellySizer(kelly_fraction=1.0, max_position_usdc=10000)
        pos = sizer.calculate("m1", "YES", p_hat=0.7, market_price=0.5, bankroll=1000)
        # f* = (0.7 - 0.5) / (1 - 0.5) = 0.4
        assert abs(pos.kelly_full - 0.4) < 0.001
        assert pos.position_usdc > 0

    def test_no_positive_edge(self):
        sizer = KellySizer(kelly_fraction=1.0, max_position_usdc=10000)
        pos = sizer.calculate("m1", "NO", p_hat=0.3, market_price=0.5, bankroll=1000)
        # f* = (0.5 - 0.3) / 0.5 = 0.4
        assert abs(pos.kelly_full - 0.4) < 0.001
        assert pos.position_usdc > 0

    def test_negative_edge_no_trade(self):
        sizer = KellySizer(kelly_fraction=0.25)
        pos = sizer.calculate("m1", "YES", p_hat=0.4, market_price=0.6, bankroll=1000)
        assert pos.position_usdc == 0
        assert pos.reason == "negative_edge"

    def test_fractional_kelly(self):
        sizer = KellySizer(kelly_fraction=0.25, max_position_usdc=10000)
        pos = sizer.calculate("m1", "YES", p_hat=0.7, market_price=0.5, bankroll=1000)
        # full Kelly = 0.4, quarter Kelly = 0.1, position = $100
        assert abs(pos.kelly_adjusted - 0.1) < 0.001
        assert abs(pos.position_usdc - 100) < 0.1

    def test_max_position_cap(self):
        sizer = KellySizer(kelly_fraction=1.0, max_position_usdc=50)
        pos = sizer.calculate("m1", "YES", p_hat=0.9, market_price=0.5, bankroll=1000)
        assert pos.position_usdc == 50
        assert pos.capped

    def test_max_portfolio_cap(self):
        sizer = KellySizer(kelly_fraction=1.0, max_position_usdc=10000, max_portfolio_usdc=100)
        sizer.record_position("m0", 80)
        pos = sizer.calculate("m1", "YES", p_hat=0.9, market_price=0.5, bankroll=1000)
        assert pos.position_usdc <= 20  # only 20 remaining

    def test_equal_probability_no_trade(self):
        sizer = KellySizer(kelly_fraction=0.25)
        pos = sizer.calculate("m1", "YES", p_hat=0.5, market_price=0.5, bankroll=1000)
        assert pos.position_usdc == 0

    def test_extreme_edge(self):
        sizer = KellySizer(kelly_fraction=0.25, max_position_usdc=10000)
        pos = sizer.calculate("m1", "YES", p_hat=0.99, market_price=0.01, bankroll=1000)
        assert pos.kelly_full > 0.9
        assert pos.position_usdc > 0

    def test_bankroll_cap(self):
        sizer = KellySizer(kelly_fraction=1.0, max_position_usdc=10000, max_portfolio_usdc=10000)
        pos = sizer.calculate("m1", "YES", p_hat=0.99, market_price=0.01, bankroll=50)
        assert pos.position_usdc <= 50

    def test_invalid_kelly_fraction(self):
        with pytest.raises(ValueError):
            KellySizer(kelly_fraction=0)
        with pytest.raises(ValueError):
            KellySizer(kelly_fraction=-0.5)

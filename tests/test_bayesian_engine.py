"""Tests for Bayesian Engine."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.bayesian_engine import BayesianEngine, Signal, SignalType


class TestBayesianEngine:
    def test_init_belief_from_market_price(self):
        engine = BayesianEngine(prior_weight=1.0)
        belief = engine.init_belief("test", 0.7)
        assert abs(belief.p_yes - 0.7) < 0.001

    def test_init_belief_blended(self):
        engine = BayesianEngine(prior_weight=0.5)
        belief = engine.init_belief("test", 0.8)
        # prior = 0.5 * 0.8 + 0.5 * 0.5 = 0.65
        assert abs(belief.p_yes - 0.65) < 0.001

    def test_update_bullish_signal(self):
        engine = BayesianEngine(prior_weight=1.0)
        engine.init_belief("test", 0.5)

        signal = Signal(
            signal_type=SignalType.NEWS_SENTIMENT,
            likelihood_yes=0.9,
            likelihood_no=0.1,
        )
        belief = engine.update("test", signal)

        # After strong bullish signal, p_yes should increase
        assert belief.p_yes > 0.5

    def test_update_bearish_signal(self):
        engine = BayesianEngine(prior_weight=1.0)
        engine.init_belief("test", 0.5)

        signal = Signal(
            signal_type=SignalType.NEWS_SENTIMENT,
            likelihood_yes=0.1,
            likelihood_no=0.9,
        )
        belief = engine.update("test", signal)
        assert belief.p_yes < 0.5

    def test_sequential_updates_converge(self):
        engine = BayesianEngine(prior_weight=1.0, min_observations=1)
        engine.init_belief("test", 0.5)

        # Send many bullish signals — should converge toward high probability
        for _ in range(20):
            signal = Signal(
                signal_type=SignalType.NEWS_SENTIMENT,
                likelihood_yes=0.8,
                likelihood_no=0.2,
                confidence=0.5,
            )
            engine.update("test", signal)

        belief = engine.get_belief("test")
        assert belief.p_yes > 0.9

    def test_opposing_signals_moderate(self):
        engine = BayesianEngine(prior_weight=1.0)
        engine.init_belief("test", 0.5)

        # Bullish then bearish — should stay near 0.5
        engine.update(
            "test",
            Signal(signal_type=SignalType.NEWS_SENTIMENT, likelihood_yes=0.8, likelihood_no=0.2),
        )
        engine.update(
            "test",
            Signal(signal_type=SignalType.NEWS_SENTIMENT, likelihood_yes=0.2, likelihood_no=0.8),
        )

        belief = engine.get_belief("test")
        assert 0.4 < belief.p_yes < 0.6

    def test_probabilities_sum_to_one(self):
        engine = BayesianEngine()
        engine.init_belief("test", 0.3)

        for _ in range(10):
            signal = Signal(
                signal_type=SignalType.MARKET_MOMENTUM,
                likelihood_yes=0.7,
                likelihood_no=0.4,
                confidence=0.8,
            )
            engine.update("test", signal)

        belief = engine.get_belief("test")
        assert abs(belief.p_yes + belief.p_no - 1.0) < 1e-10

    def test_is_tradeable(self):
        engine = BayesianEngine(min_observations=3)
        engine.init_belief("test", 0.5)

        assert not engine.is_tradeable("test")

        for i in range(3):
            engine.update(
                "test",
                Signal(signal_type=SignalType.NEWS_SENTIMENT, likelihood_yes=0.6, likelihood_no=0.4),
            )

        assert engine.is_tradeable("test")

    def test_unknown_market_raises(self):
        engine = BayesianEngine()
        with pytest.raises(KeyError):
            engine.update(
                "nonexistent",
                Signal(signal_type=SignalType.NEWS_SENTIMENT, likelihood_yes=0.5, likelihood_no=0.5),
            )

    def test_latency_under_threshold(self):
        """Update should complete in <50ms."""
        engine = BayesianEngine()
        engine.init_belief("test", 0.5)

        signal = Signal(
            signal_type=SignalType.NEWS_SENTIMENT,
            likelihood_yes=0.7,
            likelihood_no=0.3,
        )
        belief = engine.update("test", signal)
        assert belief.last_update_ms < 50

"""Tests for Edge Detector."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.edge_detector import EdgeDetector


class TestEdgeDetector:
    def test_detect_yes_edge(self):
        detector = EdgeDetector(min_edge=0.03)
        opp = detector.detect(
            market_id="m1",
            market_question="Test?",
            p_hat=0.65,
            market_price_yes=0.55,
        )
        assert opp is not None
        assert opp.side == "YES"
        assert abs(opp.edge - 0.10) < 0.001

    def test_detect_no_edge(self):
        detector = EdgeDetector(min_edge=0.03)
        opp = detector.detect(
            market_id="m1",
            market_question="Test?",
            p_hat=0.35,
            market_price_yes=0.55,
        )
        assert opp is not None
        assert opp.side == "NO"

    def test_below_threshold(self):
        detector = EdgeDetector(min_edge=0.05)
        opp = detector.detect(
            market_id="m1",
            market_question="Test?",
            p_hat=0.52,
            market_price_yes=0.50,
        )
        assert opp is None

    def test_no_edge_equal_prices(self):
        detector = EdgeDetector(min_edge=0.01)
        opp = detector.detect(
            market_id="m1",
            market_question="Test?",
            p_hat=0.50,
            market_price_yes=0.50,
        )
        assert opp is None

    def test_scan_markets_ranked(self):
        detector = EdgeDetector(min_edge=0.03)
        markets = [
            {"market_id": "m1", "question": "Q1?", "p_hat": 0.60, "market_price_yes": 0.50},
            {"market_id": "m2", "question": "Q2?", "p_hat": 0.80, "market_price_yes": 0.50},
            {"market_id": "m3", "question": "Q3?", "p_hat": 0.51, "market_price_yes": 0.50},  # below threshold
        ]
        opps = detector.scan_markets(markets)
        assert len(opps) == 2
        assert opps[0].market_id == "m2"  # largest edge first
        assert opps[1].market_id == "m1"

    def test_volume_filter(self):
        detector = EdgeDetector(min_edge=0.03, min_volume_24h=1000)
        opp = detector.detect(
            market_id="m1",
            market_question="Test?",
            p_hat=0.70,
            market_price_yes=0.50,
            volume_24h=500,  # below minimum
        )
        assert opp is None

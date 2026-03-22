"""Tests for Trader pending-order reconciliation."""

import importlib
import time

from src.execution.clob_executor import OrderStatus

run_trader = importlib.import_module("scripts.run_trader")
Trader = run_trader.Trader


class _DummyPositions:
    def __init__(self):
        self.added = []

    def add_position(self, position):
        self.added.append(position)

    def get_position(self, market_id: str):
        return None


class _DummyExposure:
    def __init__(self):
        self.recorded = []

    def record_position(self, market_id: str, size_usdc: float):
        self.recorded.append((market_id, size_usdc))


class _DummyExecutor:
    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.cancelled = []

    def refresh_order_status(self, order):
        status, filled_size, filled_avg_price = self._statuses.pop(0)
        order.status = status
        order.filled_size = filled_size
        order.filled_avg_price = filled_avg_price
        return order

    def cancel_order(self, order_id: str):
        self.cancelled.append(order_id)
        return True

    def _onchain_token_balance(self, token_id: str):
        return 0.0


def test_check_pending_orders_only_records_incremental_partial_fills():
    trader = Trader.__new__(Trader)
    trader._pending_order_max_age_s = 10_000
    trader._pending_orders = [{
        "order_id": "o1",
        "market_id": "m1",
        "condition_id": "c1",
        "token_id": "t1",
        "side": "YES",
        "entry_price": 0.6,
        "size": 10.0,
        "p_hat": 0.62,
        "market_price": 0.6,
        "end_date": "",
        "created_at": time.time(),
        "recorded_filled_size": 0.0,
    }]
    trader.positions = _DummyPositions()
    trader.risk = _DummyExposure()
    trader.kelly = _DummyExposure()
    trader._save_pending_orders = lambda: None
    trader.executor = _DummyExecutor([
        (OrderStatus.PARTIALLY_FILLED, 4.0, 0.6),
        (OrderStatus.PARTIALLY_FILLED, 4.0, 0.6),
        (OrderStatus.PARTIALLY_FILLED, 7.0, 0.6),
    ])

    trader._check_pending_orders()
    trader._check_pending_orders()
    trader._check_pending_orders()

    assert [round(pos.size, 6) for pos in trader.positions.added] == [4.0, 3.0]
    assert trader._pending_orders[0]["size"] == 3.0
    assert trader._pending_orders[0]["recorded_filled_size"] == 7.0

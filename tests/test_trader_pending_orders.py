"""Tests for Trader pending-order reconciliation."""

import asyncio
import importlib
import time

import pytest

from src.execution.clob_executor import Order, OrderSide, OrderStatus
from src.execution.position_manager import Position

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
        self.set_calls = []
        self.closed = []
        self.pnl = []

    def record_position(self, market_id: str, size_usdc: float):
        self.recorded.append((market_id, size_usdc))

    def set_position(self, market_id: str, size_usdc: float):
        self.set_calls.append((market_id, size_usdc))

    def close_position(self, market_id: str):
        self.closed.append(market_id)

    def record_pnl(self, amount: float):
        self.pnl.append(amount)


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


class _DummyExitPositions:
    exit_max_retries = 5
    exit_price_step = 0.01

    def __init__(self):
        self.closed = []
        self.saved = 0

    def _save_state(self):
        self.saved += 1

    def close_position(self, market_id: str):
        self.closed.append(market_id)


class _DummyPerf:
    def __init__(self):
        self.closed = []

    def record_close(self, trade):
        self.closed.append(trade)

    def should_evaluate(self):
        return False


class _DummyExitExecutor:
    def __init__(self, status, filled_size, filled_avg_price, onchain_balance=10.0):
        self.status = status
        self.filled_size = filled_size
        self.filled_avg_price = filled_avg_price
        self.onchain_balance = onchain_balance
        self.synced = 0

    def _onchain_token_balance(self, token_id: str):
        return self.onchain_balance

    def fetch_market_meta(self, condition_id: str):
        return {"tick_size": "0.01"}

    def place_order(self, token_id: str, side, price: float, size: float, condition_id: str):
        return Order(
            order_id="exit-1",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=self.status,
            filled_size=self.filled_size,
            filled_avg_price=self.filled_avg_price,
        )

    def refresh_order_status(self, order):
        return order

    def _sync_clob_balance(self):
        self.synced += 1


def test_execute_exit_partial_fill_updates_remaining_exposure():
    trader = Trader.__new__(Trader)
    trader.positions = _DummyExitPositions()
    trader.risk = _DummyExposure()
    trader.kelly = _DummyExposure()
    trader.perf = _DummyPerf()
    trader.executor = _DummyExitExecutor(
        OrderStatus.PARTIALLY_FILLED, filled_size=4.0, filled_avg_price=0.72
    )
    trader._add_exit_cooldown = lambda market_id: None
    trader._strat_cfg = {}
    trader.edge_detector = object()

    pos = Position(
        market_id="m1",
        condition_id="c1",
        token_id="t1",
        side="YES",
        entry_price=0.50,
        size=10.0,
        size_usdc=5.0,
        p_hat_at_entry=0.55,
        market_price_at_entry=0.50,
    )

    asyncio.run(trader._execute_exit(pos, current_price=0.80, market_map={}, reason="profit_target"))

    assert pos.size == 6.0
    assert pos.size_usdc == 3.0
    assert trader.risk.set_calls == [("m1", 3.0)]
    assert trader.kelly.set_calls == [("m1", 3.0)]
    assert trader.risk.pnl == [pytest.approx(0.88)]
    assert trader.perf.closed == []


def test_execute_exit_records_actual_fill_price_in_trade_history():
    trader = Trader.__new__(Trader)
    trader.positions = _DummyExitPositions()
    trader.risk = _DummyExposure()
    trader.kelly = _DummyExposure()
    trader.perf = _DummyPerf()
    trader.executor = _DummyExitExecutor(
        OrderStatus.FILLED, filled_size=10.0, filled_avg_price=0.72
    )
    trader._add_exit_cooldown = lambda market_id: None
    trader._strat_cfg = {}
    trader.edge_detector = object()

    pos = Position(
        market_id="m1",
        condition_id="c1",
        token_id="t1",
        side="YES",
        entry_price=0.50,
        size=10.0,
        size_usdc=5.0,
        p_hat_at_entry=0.55,
        market_price_at_entry=0.50,
    )

    asyncio.run(trader._execute_exit(pos, current_price=0.80, market_map={}, reason="profit_target"))

    assert trader.positions.closed == ["m1"]
    assert trader.risk.closed == ["m1"]
    assert trader.kelly.closed == ["m1"]
    assert trader.perf.closed[0].exit_price == 0.72
    assert trader.perf.closed[0].size_usdc == 5.0

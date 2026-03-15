"""
Order Lifecycle Manager

Handles order timeouts, re-pricing, and position tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from src.execution.clob_executor import CLOBExecutor, Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class ManagedOrder:
    """An order with lifecycle metadata."""

    order: Order
    max_age_s: float = 300.0  # cancel after 5 min if unfilled
    created_at: float = field(default_factory=time.time)
    reprices: int = 0
    max_reprices: int = 3

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at

    @property
    def is_stale(self) -> bool:
        return self.age_s > self.max_age_s and self.order.is_active


class OrderManager:
    """Manages order lifecycle: timeouts, re-pricing, tracking."""

    def __init__(self, executor: CLOBExecutor, stale_timeout_s: float = 300.0):
        self.executor = executor
        self.stale_timeout_s = stale_timeout_s
        self._managed: dict[str, ManagedOrder] = {}

    async def submit_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
    ) -> ManagedOrder:
        """Submit and track a new limit order."""
        order = await self.executor.place_limit_order(token_id, side, price, size)
        managed = ManagedOrder(order=order, max_age_s=self.stale_timeout_s)
        self._managed[order.order_id] = managed
        return managed

    async def sweep_stale(self) -> list[str]:
        """Cancel all stale (timed out) orders.

        Returns:
            List of cancelled order IDs
        """
        cancelled = []
        for oid, managed in list(self._managed.items()):
            if managed.is_stale:
                success = await self.executor.cancel_order(oid)
                if success:
                    cancelled.append(oid)
                    logger.info(
                        f"Swept stale order {oid} (age={managed.age_s:.0f}s)"
                    )
        return cancelled

    async def reprice_order(
        self,
        order_id: str,
        new_price: float,
    ) -> ManagedOrder | None:
        """Cancel an order and replace at a new price.

        Returns:
            New ManagedOrder or None if failed
        """
        managed = self._managed.get(order_id)
        if not managed:
            return None

        if managed.reprices >= managed.max_reprices:
            logger.warning(f"Max reprices reached for {order_id}")
            return None

        old_order = managed.order
        await self.executor.cancel_order(order_id)

        new_managed = await self.submit_order(
            token_id=old_order.token_id,
            side=old_order.side,
            price=new_price,
            size=old_order.size - old_order.filled_size,
        )
        new_managed.reprices = managed.reprices + 1
        return new_managed

    def get_active_orders(self) -> list[ManagedOrder]:
        return [m for m in self._managed.values() if m.order.is_active]

    @property
    def active_count(self) -> int:
        return len(self.get_active_orders())

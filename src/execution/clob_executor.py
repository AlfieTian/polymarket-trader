"""
CLOB Order Execution Layer

Places and manages orders on Polymarket's Central Limit Order Book.
Supports dry-run mode for safe testing.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    """Represents a single order."""

    order_id: str
    token_id: str
    side: OrderSide
    order_type: OrderType
    price: float
    size: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    filled_avg_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    dry_run: bool = False

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def fill_pct(self) -> float:
        return (self.filled_size / self.size * 100) if self.size > 0 else 0


class CLOBExecutor:
    """Polymarket CLOB order execution engine.

    Handles order placement, cancellation, and lifecycle management.
    Dry-run mode logs orders without sending them to the exchange.
    """

    def __init__(
        self,
        clob_url: str = "https://clob.polymarket.com",
        api_key: str = "",
        private_key: str = "",
        dry_run: bool = True,
    ):
        self.clob_url = clob_url
        self.api_key = api_key
        self.private_key = private_key
        self.dry_run = dry_run
        self._orders: dict[str, Order] = {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["POLY_API_KEY"] = self.api_key
            self._client = httpx.AsyncClient(
                base_url=self.clob_url,
                headers=headers,
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    def estimate_slippage(
        self,
        orderbook_bids: list[dict],
        orderbook_asks: list[dict],
        side: OrderSide,
        size: float,
    ) -> dict:
        """Estimate execution slippage based on order book depth.

        Args:
            orderbook_bids: List of {"price": float, "size": float}
            orderbook_asks: List of {"price": float, "size": float}
            side: BUY or SELL
            size: Order size in shares

        Returns:
            Dict with estimated avg_price, slippage_bps, and feasible flag
        """
        book = orderbook_asks if side == OrderSide.BUY else orderbook_bids
        if not book:
            return {"avg_price": 0, "slippage_bps": 0, "feasible": False}

        remaining = size
        total_cost = 0.0
        mid_price = (
            (orderbook_bids[0]["price"] + orderbook_asks[0]["price"]) / 2
            if orderbook_bids and orderbook_asks
            else book[0]["price"]
        )

        for level in book:
            fill = min(remaining, level["size"])
            total_cost += fill * level["price"]
            remaining -= fill
            if remaining <= 0:
                break

        if remaining > 0:
            return {
                "avg_price": total_cost / (size - remaining) if size > remaining else 0,
                "slippage_bps": 0,
                "feasible": False,
                "unfilled": remaining,
            }

        avg_price = total_cost / size
        slippage_bps = abs(avg_price - mid_price) / mid_price * 10000

        return {
            "avg_price": avg_price,
            "slippage_bps": slippage_bps,
            "feasible": True,
        }

    async def place_limit_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
    ) -> Order:
        """Place a limit order on the CLOB.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price (0-1)
            size: Number of shares

        Returns:
            Order object with status
        """
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            order_id=order_id,
            token_id=token_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            size=size,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            order.status = OrderStatus.OPEN
            self._orders[order_id] = order
            logger.info(
                f"[DRY RUN] Limit {side.value} {size:.2f} shares of {token_id} "
                f"@ ${price:.4f} — order_id={order_id}"
            )
            return order

        # Live execution
        try:
            client = await self._get_client()
            payload = {
                "tokenID": token_id,
                "side": side.value,
                "price": str(price),
                "size": str(size),
                "type": "LIMIT",
            }
            resp = await client.post("/order", json=payload)
            resp.raise_for_status()
            result = resp.json()

            order.order_id = result.get("orderID", order_id)
            order.status = OrderStatus.OPEN
            self._orders[order.order_id] = order

            logger.info(
                f"Placed limit {side.value} {size:.2f} @ ${price:.4f} "
                f"— order_id={order.order_id}"
            )
        except Exception as e:
            order.status = OrderStatus.FAILED
            logger.error(f"Failed to place order: {e}")

        return order

    async def place_market_order(
        self,
        token_id: str,
        side: OrderSide,
        size: float,
    ) -> Order:
        """Place a market order (aggressive limit at best ask/bid).

        Args:
            token_id: Token to trade
            side: BUY or SELL
            size: Number of shares

        Returns:
            Order object
        """
        # Market orders = aggressive limit at 0.99 (buy) or 0.01 (sell)
        price = 0.99 if side == OrderSide.BUY else 0.01
        return await self.place_limit_order(token_id, side, price, size)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancelled successfully
        """
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Order {order_id} not found")
            return False

        if not order.is_active:
            logger.warning(f"Order {order_id} is not active (status={order.status})")
            return False

        if self.dry_run:
            order.status = OrderStatus.CANCELLED
            order.updated_at = time.time()
            logger.info(f"[DRY RUN] Cancelled order {order_id}")
            return True

        try:
            client = await self._get_client()
            resp = await client.delete(f"/order/{order_id}")
            resp.raise_for_status()
            order.status = OrderStatus.CANCELLED
            order.updated_at = time.time()
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_open_orders(self) -> list[Order]:
        """Get all currently active orders."""
        return [o for o in self._orders.values() if o.is_active]

    async def cancel_all(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        open_orders = self.get_open_orders()
        cancelled = 0
        for order in open_orders:
            if await self.cancel_order(order.order_id):
                cancelled += 1
        return cancelled

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

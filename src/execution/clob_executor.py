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


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    order_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    filled_avg_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    dry_run: bool = False

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def fill_pct(self) -> float:
        return (self.filled_size / self.size * 100) if self.size > 0 else 0


class CLOBExecutor:
    """Polymarket CLOB executor with full BUY/SELL support, tick_size, neg_risk."""

    def __init__(
        self,
        clob_url: str = "https://clob.polymarket.com",
        private_key: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        wallet_address: str = "",
        dry_run: bool = True,
    ):
        self.clob_url = clob_url
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.wallet_address = wallet_address
        self.dry_run = dry_run
        self._orders: dict[str, Order] = {}
        self._clob_client = None
        # Cache: condition_id → {tick_size, neg_risk}
        self._market_meta: dict[str, dict] = {}

    def _init_clob_client(self):
        if self._clob_client is not None:
            return
        if not self.private_key:
            return
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.api_passphrase,
            )
            self._clob_client = ClobClient(
                host=self.clob_url,
                key=self.private_key,
                chain_id=137,
                creds=creds,
                signature_type=0,
                funder=self.wallet_address,
            )
            logger.info("✅ CLOB client initialized")
        except Exception as e:
            logger.error(f"Failed to init CLOB client: {e}")

    def fetch_market_meta(self, condition_id: str) -> dict:
        """Fetch tick_size and neg_risk for a market (cached)."""
        if condition_id in self._market_meta:
            return self._market_meta[condition_id]

        self._init_clob_client()
        if not self._clob_client:
            return {"tick_size": "0.01", "neg_risk": False}

        try:
            meta = self._clob_client.get_market(condition_id)
            result = {
                "tick_size": str(meta.get("minimum_tick_size", "0.01")),
                "neg_risk": bool(meta.get("neg_risk", False)),
            }
            self._market_meta[condition_id] = result
            return result
        except Exception as e:
            logger.warning(f"Could not fetch market meta for {condition_id}: {e}, using defaults")
            return {"tick_size": "0.01", "neg_risk": False}

    def _snap_price(self, price: float, tick_size: str) -> float:
        """Snap price to valid tick size increment."""
        tick = float(tick_size)
        snapped = round(round(price / tick) * tick, 4)
        return max(tick, min(1 - tick, snapped))

    def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        condition_id: str = "",
    ) -> Order:
        """Place a limit order on the CLOB.

        Args:
            token_id: YES or NO token ID to trade
            side: BUY or SELL
            price: Limit price (0-1)
            size: Number of shares
            condition_id: Market condition ID (needed for tick_size/neg_risk)

        Returns:
            Order object
        """
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            order_id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            order.status = OrderStatus.OPEN
            self._orders[order_id] = order
            logger.info(
                f"[DRY RUN] {side.value} {size:.2f} shares @ ${price:.3f} "
                f"token={token_id[:12]}..."
            )
            return order

        self._init_clob_client()
        if not self._clob_client:
            order.status = OrderStatus.FAILED
            return order

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY, SELL

            # Get tick_size and neg_risk
            meta = self.fetch_market_meta(condition_id) if condition_id else {"tick_size": "0.01", "neg_risk": False}
            snapped_price = self._snap_price(price, meta["tick_size"])

            order_args = OrderArgs(
                token_id=token_id,
                price=snapped_price,
                size=round(size, 2),
                side=BUY if side == OrderSide.BUY else SELL,
            )
            options = PartialCreateOrderOptions(
                tick_size=meta["tick_size"],
                neg_risk=meta["neg_risk"],
            )

            result = self._clob_client.create_and_post_order(order_args, options)

            order.order_id = result.get("orderID", order_id)
            order.status = OrderStatus.OPEN
            self._orders[order.order_id] = order

            logger.info(
                f"✅ {side.value} {size:.2f} @ ${snapped_price:.3f} "
                f"— order_id={order.order_id}"
            )
        except Exception as e:
            order.status = OrderStatus.FAILED
            logger.error(f"❌ Order failed: {e}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        order = self._orders.get(order_id)
        if not order or not order.is_active:
            return False

        if self.dry_run:
            order.status = OrderStatus.CANCELLED
            logger.info(f"[DRY RUN] Cancelled {order_id}")
            return True

        self._init_clob_client()
        if not self._clob_client:
            return False

        try:
            self._clob_client.cancel(order_id)
            order.status = OrderStatus.CANCELLED
            logger.info(f"✅ Cancelled {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Cancel failed: {e}")
            return False

    def cancel_all(self) -> int:
        """Cancel all open orders."""
        if self.dry_run:
            count = sum(1 for o in self._orders.values() if o.is_active)
            for o in self._orders.values():
                if o.is_active:
                    o.status = OrderStatus.CANCELLED
            logger.info(f"[DRY RUN] Cancelled {count} orders")
            return count

        self._init_clob_client()
        if not self._clob_client:
            return 0

        try:
            self._clob_client.cancel_all()
            count = sum(1 for o in self._orders.values() if o.is_active)
            for o in self._orders.values():
                if o.is_active:
                    o.status = OrderStatus.CANCELLED
            return count
        except Exception as e:
            logger.error(f"❌ Cancel all failed: {e}")
            return 0

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_active]

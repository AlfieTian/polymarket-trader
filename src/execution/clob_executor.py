"""
CLOB Order Execution Layer

Places and manages orders on Polymarket's Central Limit Order Book.
Supports dry-run mode for safe testing.
"""

import asyncio
import logging
import math
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
        max_position_usdc: float = 100.0,
    ):
        self.clob_url = clob_url
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.wallet_address = wallet_address
        self.dry_run = dry_run
        self.max_position_usdc = max_position_usdc
        self._orders: dict[str, Order] = {}
        self._clob_client = None
        # Cache: condition_id → {tick_size, neg_risk}
        self._market_meta: dict[str, dict] = {}

    def _parse_amount(self, value) -> float:
        """Best-effort parse for balance/allowance numeric values."""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value)
            if isinstance(value, dict):
                # Common nested shapes
                for key in ("balance", "available", "total", "allowance", "approved"):
                    if key in value:
                        return self._parse_amount(value.get(key))
        except Exception:
            return 0.0
        return 0.0

    def get_balance_allowance(self, asset_type: str, token_id: str = "") -> dict:
        """Fetch balance/allowance from CLOB (best-effort).

        Uses update_balance_allowance (which syncs from chain AND returns the value)
        so we never read stale cache. Falls back to get_balance_allowance if needed.
        """
        if self.dry_run:
            return {"balance": float("inf"), "allowance": float("inf")}

        self._init_clob_client()
        if not self._clob_client:
            return {"balance": 0.0, "allowance": 0.0}

        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            if asset_type.upper() == "COLLATERAL":
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=0)
            else:
                params = BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=token_id, signature_type=0
                )

            # Prefer update_balance_allowance: syncs CLOB from chain and returns current value
            result = None
            if hasattr(self._clob_client, "update_balance_allowance"):
                try:
                    result = self._clob_client.update_balance_allowance(params=params)
                    if not result:  # empty string or None
                        result = None
                except Exception as ue:
                    logger.debug(f"update_balance_allowance failed, falling back: {ue}")

            if result is None:
                result = self._clob_client.get_balance_allowance(params=params)
            
            logger.debug(f"Balance/allowance raw result: {result}")

            if isinstance(result, dict):
                balance = self._parse_amount(result.get("balance"))
                allowance = self._parse_amount(result.get("allowance"))
            else:
                balance = self._parse_amount(result)
                allowance = self._parse_amount(result)

            return {"balance": balance, "allowance": allowance}
        except Exception as e:
            logger.warning(f"Balance/allowance fetch failed: {e}")
            return {"balance": 0.0, "allowance": 0.0}

    def _reset_web3(self):
        """Reset Web3 connection (call after SSL/connection errors)."""
        self._w3 = None
        self._w3_checked_at = 0.0

    def _ensure_web3(self):
        """Lazily initialize Web3 + contract instances for on-chain reads."""
        if hasattr(self, "_w3") and self._w3 is not None:
            # Verify connection is still alive at most once per 60s
            now = time.time()
            last_check = getattr(self, "_w3_checked_at", 0.0)
            if now - last_check > 60:
                try:
                    self._w3.eth.block_number  # lightweight liveness check
                    self._w3_checked_at = now
                except Exception:
                    logger.debug("Web3 connection lost, reconnecting...")
                    self._w3 = None
        if hasattr(self, "_w3") and self._w3 is not None:
            return
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware

        CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
        CTF_ABI = [{"inputs":[{"name":"account","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
        USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        USDC_E_ABI = [{"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
        # Primary RPC; fallback list in case of SSL/403 issues
        RPC_LIST = [
            "https://polygon-rpc.com",
            "https://polygon-bor-rpc.publicnode.com",
        ]
        self._w3 = None
        for rpc in RPC_LIST:
            try:
                candidate = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                candidate.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                if candidate.is_connected():
                    self._w3 = candidate
                    logger.debug(f"Web3 connected via {rpc}")
                    break
            except Exception as e:
                logger.debug(f"RPC {rpc} failed: {e}")
        if self._w3 is None:
            raise RuntimeError("All Polygon RPC endpoints unavailable")
        self._ctf_contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI
        )
        self._usdc_e_contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(USDC_E_ADDRESS), abi=USDC_E_ABI
        )

    def _onchain_usdc_balance(self, fail_closed: bool = True) -> float | None:
        """Read USDC.e balance directly from Polygon chain (source of truth).

        Returns balance in USDC (raw / 1e6).
        """
        try:
            from web3 import Web3
            self._ensure_web3()
            wallet = Web3.to_checksum_address(self.wallet_address)
            raw = self._usdc_e_contract.functions.balanceOf(wallet).call()
            balance = raw / 1e6
            logger.info(f"🔍 On-chain USDC.e balance: ${balance:.4f}")
            return balance
        except Exception as e:
            logger.warning(f"On-chain USDC.e balance check failed: {e}")
            self._reset_web3()  # Force reconnect next call
            return 0.0 if fail_closed else None

    def _onchain_token_balance(self, token_id: str) -> float:
        """Check on-chain ERC-1155 conditional token balance via CTF contract.

        Polymarket's CLOB balance-allowance API is unreliable for conditional tokens
        (returns stale/zero values). We read directly from chain for SELL checks.
        Returns balance in shares (raw / 1e6).
        """
        try:
            from web3 import Web3
            self._ensure_web3()

            wallet = Web3.to_checksum_address(self.wallet_address)
            raw = self._ctf_contract.functions.balanceOf(wallet, int(token_id)).call()
            shares = raw / 1e6
            logger.info(f"🔍 On-chain balance for {token_id[:16]}...: raw={raw}, shares={shares:.4f}")
            return shares
        except Exception as e:
            logger.warning(f"On-chain token balance check failed for {token_id[:12]}...: {e}")
            self._reset_web3()  # Force reconnect next call
            return float("inf")  # fail open so the CLOB handles rejection

    def has_sufficient_balance(
        self,
        side: OrderSide,
        price: float,
        size: float,
        token_id: str = "",
    ) -> bool:
        """Pre-check balance before placing an order.

        BUY: check USDC.e balance/allowance via CLOB (update endpoint for freshness).
        SELL: check conditional token balance on-chain via CTF contract (CLOB is unreliable here).
        """
        if self.dry_run:
            return True

        if side == OrderSide.BUY:
            needed = size * price
            # Use on-chain USDC.e balance as source of truth
            available = self._onchain_usdc_balance(fail_closed=True)
            if available + 1e-9 < needed:
                logger.info(
                    f"⏭️  Order skipped: on-chain USDC.e balance "
                    f"${available:.4f} < needed ${needed:.4f}"
                )
                return False
            return True

        # SELL: use on-chain CTF balance (CLOB conditional balance is unreliable)
        if not token_id:
            return True  # no token_id to check; let CLOB handle it

        available = self._onchain_token_balance(token_id)
        logger.info(f"🔍 SELL balance check: available={available:.4f}, needed={size:.4f}, token_id={token_id[:16]}...")
        if available + 1e-9 < size:
            logger.warning(
                f"⏭️  Order skipped: on-chain token balance "
                f"{available:.4f} < needed {size:.4f} shares"
            )
            return False
        return True

    def _cancel_existing_orders(self, token_id: str):
        """Cancel all open orders for a given token_id to free locked collateral."""
        if self.dry_run or not self._clob_client:
            return
        try:
            orders = self._clob_client.get_orders()
            if not isinstance(orders, list):
                return
            stale = [o for o in orders if o.get("asset_id") == token_id and o.get("status") == "LIVE"]
            if not stale:
                return
            stale_ids = [o["id"] for o in stale]
            logger.info(f"🗑️  Cancelling {len(stale_ids)} stale order(s) for token {token_id[:16]}...")
            try:
                self._clob_client.cancel_orders(stale_ids)
            except Exception:
                # Fallback: cancel one by one
                for oid in stale_ids:
                    try:
                        self._clob_client.cancel(oid)
                    except Exception as ce:
                        logger.warning(f"Failed to cancel order {oid[:16]}...: {ce}")
            logger.info(f"✅ Cancelled {len(stale_ids)} stale order(s)")
        except Exception as e:
            logger.warning(f"Cancel existing orders failed (non-fatal): {e}")

    def get_live_orders(self) -> list[dict]:
        """Fetch currently live/open orders directly from CLOB."""
        if self.dry_run:
            return []

        self._init_clob_client()
        if not self._clob_client:
            return []

        try:
            orders = self._clob_client.get_orders()
            if not isinstance(orders, list):
                return []
            live_statuses = {"LIVE", "OPEN", "PENDING", "PARTIALLY_FILLED"}
            result = []
            for order in orders:
                status = str(order.get("status") or order.get("state") or "").upper()
                if status in live_statuses:
                    result.append(order)
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch live CLOB orders: {e}")
            return []

    def cancel_all_live_orders(self) -> int:
        """Cancel all live remote orders to reconcile after crashes/restarts."""
        if self.dry_run:
            return 0

        live_orders = self.get_live_orders()
        if not live_orders:
            return 0

        order_ids = [str(o.get("id") or o.get("orderID") or "") for o in live_orders]
        order_ids = [oid for oid in order_ids if oid]
        if not order_ids:
            return 0
        count = len(order_ids)

        self._init_clob_client()
        if not self._clob_client:
            return 0

        try:
            if hasattr(self._clob_client, "cancel_orders"):
                self._clob_client.cancel_orders(order_ids)
            else:
                for oid in order_ids:
                    self._clob_client.cancel(oid)
            logger.warning(f"🧹 Cancelled {count} live CLOB order(s) during reconciliation")
            return count
        except Exception as e:
            logger.warning(f"Failed to cancel live CLOB orders during reconciliation: {e}")
            return 0

    def _init_clob_client(self):
        if self._clob_client is not None:
            return
        if not self.private_key:
            return
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            # Use create_or_derive to ensure L2 auth is available (env creds sometimes 401)
            _l1 = ClobClient(host=self.clob_url, key=self.private_key, chain_id=137)
            creds = _l1.create_or_derive_api_creds()

            self._clob_client = ClobClient(
                host=self.clob_url,
                key=self.private_key,
                chain_id=137,
                creds=creds,
                signature_type=0,
                funder=self.wallet_address,
            )
            logger.info("✅ CLOB client initialized (creds derived)")

            # Sync on-chain balance snapshot on startup
            self._sync_clob_balance()
        except Exception as e:
            logger.error(f"Failed to init CLOB client: {e}")

    def _sync_clob_balance(self):
        """Notify CLOB to refresh on-chain token/USDC balances, fixing 'not enough balance/allowance' errors."""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            from .position_manager import STATE_FILE
            import json
            # Refresh USDC
            self._clob_client.update_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=0)
            )
            # Refresh each conditional token in positions
            try:
                positions = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else []
            except Exception:
                positions = []
            for pos in positions:
                tid = pos.get("token_id", "")
                if tid:
                    self._clob_client.update_balance_allowance(
                        params=BalanceAllowanceParams(
                            asset_type=AssetType.CONDITIONAL,
                            token_id=tid,
                            signature_type=0,
                        )
                    )
                    logger.info(f"🔄 CLOB balance synced for token {tid[:16]}...")
        except Exception as e:
            logger.warning(f"CLOB balance sync failed (non-fatal): {e}")

    def fetch_market_meta(self, condition_id: str) -> dict:
        """Fetch tick_size and neg_risk for a market (cached)."""
        if condition_id in self._market_meta:
            return self._market_meta[condition_id]

        self._init_clob_client()
        if not self._clob_client:
            return {"tick_size": "0.01", "neg_risk": False, "min_order_size": 5}

        try:
            meta = self._clob_client.get_market(condition_id)
            result = {
                "tick_size": str(meta.get("minimum_tick_size", "0.01")),
                "neg_risk": bool(meta.get("neg_risk", False)),
                "min_order_size": float(meta.get("min_order_size", 5)),
            }
            self._market_meta[condition_id] = result
            return result
        except Exception as e:
            logger.warning(f"Could not fetch market meta for {condition_id}: {e}, using defaults")
            return {"tick_size": "0.01", "neg_risk": False, "min_order_size": 5}

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

            # Get tick_size, neg_risk, min_order_size
            meta = self.fetch_market_meta(condition_id) if condition_id else {"tick_size": "0.01", "neg_risk": False, "min_order_size": 5}
            snapped_price = self._snap_price(price, meta["tick_size"])

            # Enforce market minimum order size (shares)
            min_size = meta.get("min_order_size", 5)
            if round(size, 2) < min_size:
                min_cost = min_size * snapped_price
                if min_cost <= self.max_position_usdc:
                    # Bump up to minimum — still within per-position cap
                    logger.info(f"📐 Bumping {size:.2f}→{min_size} shares (min order, cost ${min_cost:.2f})")
                    size = float(min_size)
                    order.size = size  # reflect actual executed size
                else:
                    logger.warning(f"⏭️  Order skipped: min order ${min_cost:.2f} > cap ${self.max_position_usdc:.2f}")
                    order.status = OrderStatus.FAILED
                    return order

            # Enforce Polymarket minimum order amount ($1 USDC for BUY orders)
            # Even if shares >= min_size, cost = shares × price must be >= $1
            MIN_ORDER_USDC = 1.0
            if side == OrderSide.BUY:
                order_cost = size * snapped_price
                if order_cost < MIN_ORDER_USDC:
                    # Bump shares up so cost hits $1
                    size_needed = MIN_ORDER_USDC / snapped_price if snapped_price > 0 else size
                    size_needed = math.ceil(size_needed * 100) / 100  # round up to 2dp
                    bumped_cost = size_needed * snapped_price
                    if bumped_cost <= self.max_position_usdc:
                        logger.info(
                            f"📐 Bumping {size:.2f}→{size_needed} shares "
                            f"(min $1 USDC order, cost ${bumped_cost:.2f})"
                        )
                        size = size_needed
                        order.size = size
                    else:
                        logger.warning(
                            f"⏭️  Order skipped: min $1 cost ${bumped_cost:.2f} > cap ${self.max_position_usdc:.2f}"
                        )
                        order.status = OrderStatus.FAILED
                        return order

            # Cancel any existing open orders for this token before placing new one
            # Prevents duplicate orders from piling up and freezing collateral
            self._cancel_existing_orders(token_id)

            # Pre-flight balance/allowance check (avoid API rejection)
            if not self.has_sufficient_balance(side=side, price=snapped_price, size=size, token_id=token_id):
                order.status = OrderStatus.FAILED
                return order

            # For SELL orders: floor to 2dp to avoid exceeding CLOB balance
            # (on-chain balance may be e.g. 4.9999 shares, round() gives 5.0 > CLOB limit)
            # For BUY orders: round() is safe (we're spending USDC, not tokens)
            if side == OrderSide.SELL:
                safe_size = math.floor(size * 100) / 100
            else:
                safe_size = round(size, 2)
            order_args = OrderArgs(
                token_id=token_id,
                price=snapped_price,
                size=safe_size,
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

    def refresh_order_status(self, order: Order) -> Order:
        """Refresh order status from CLOB and update filled info."""
        if self.dry_run:
            order.status = OrderStatus.FILLED
            order.filled_size = order.size
            order.filled_avg_price = order.price
            self._orders[order.order_id] = order
            return order

        self._init_clob_client()
        if not self._clob_client:
            return order

        try:
            fetch_fn = (
                getattr(self._clob_client, "get_order", None)
                or getattr(self._clob_client, "get_order_by_id", None)
                or getattr(self._clob_client, "get_order_status", None)
            )
            if not fetch_fn:
                logger.warning("CLOB client missing order-status method")
                return order

            data = fetch_fn(order.order_id)
            if not isinstance(data, dict):
                return order

            status_raw = str(
                data.get("status")
                or data.get("state")
                or data.get("orderStatus")
                or ""
            ).upper()
            if "PART" in status_raw:
                order.status = OrderStatus.PARTIALLY_FILLED
            elif "FILL" in status_raw:
                order.status = OrderStatus.FILLED
            elif "OPEN" in status_raw or "LIVE" in status_raw:
                order.status = OrderStatus.OPEN
            elif "CANCEL" in status_raw:
                order.status = OrderStatus.CANCELLED
            elif "FAIL" in status_raw or "REJECT" in status_raw:
                order.status = OrderStatus.FAILED

            filled_size = (
                data.get("filled_size")
                or data.get("filledSize")
                or data.get("matchedSize")
                or data.get("sizeFilled")
                or 0
            )
            order.filled_size = self._parse_amount(filled_size)

            avg_price = (
                data.get("avg_fill_price")
                or data.get("avgFillPrice")
                or data.get("avgPrice")
                or data.get("price")
                or 0
            )
            order.filled_avg_price = self._parse_amount(avg_price)

            self._orders[order.order_id] = order
        except Exception as e:
            logger.warning(f"Order status refresh failed: {e}")

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
            count = self.cancel_all_live_orders()
            if count == 0:
                self._clob_client.cancel_all()
            for o in self._orders.values():
                if o.is_active:
                    o.status = OrderStatus.CANCELLED
            return count
        except Exception as e:
            logger.error(f"❌ Cancel all failed: {e}")
            return 0

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_active]

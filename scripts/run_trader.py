#!/usr/bin/env python3
"""
Polymarket Trader — Main Entry Point

Async loop:
  Fetch markets → Bayesian update (LLM news signals) → Detect edge →
  Kelly sizing → Risk check → Execute BUY → Track positions → Execute SELL on exit
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.data.news_feed import NewsFeed
from src.data.polymarket_client import PolymarketClient
from src.execution.clob_executor import CLOBExecutor, OrderSide, OrderStatus
from src.execution.position_manager import Position, PositionManager
from src.execution.redeemer import Redeemer
from src.risk.risk_manager import RiskManager
from src.signals.bayesian_engine import BayesianEngine
from src.strategy.edge_detector import EdgeDetector
from src.strategy.kelly_sizer import KellySizer
from src.strategy.performance_tracker import ClosedTrade, PerformanceTracker

console = Console()
logger = logging.getLogger("trader")


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(__file__).parent.parent / path
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    from logging.handlers import RotatingFileHandler
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "logs/trader.log")
    log_path = Path(__file__).parent.parent / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Rotate at 20 MB, keep 5 backups → max ~100 MB log storage
    file_handler = RotatingFileHandler(log_path, maxBytes=20 * 1024 * 1024, backupCount=5)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            file_handler,
        ],
    )


class Trader:
    def __init__(self, config: dict):
        self.config = config
        pm_cfg = config.get("polymarket", {})
        strat_cfg = config.get("strategy", {})
        bay_cfg = config.get("bayesian", {})
        risk_cfg = config.get("risk", {})
        pos_cfg = config.get("position", {})

        self.dry_run = pm_cfg.get("dry_run", True)

        self.client = PolymarketClient(dry_run=self.dry_run)

        self.executor = CLOBExecutor(
            clob_url="https://clob.polymarket.com",
            private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            api_key=os.getenv("POLYMARKET_API_KEY", ""),
            api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
            wallet_address=os.getenv("POLYMARKET_WALLET_ADDRESS", ""),
            dry_run=self.dry_run,
            max_position_usdc=strat_cfg.get("max_position_usdc", 100),
        )

        self.bayesian = BayesianEngine(
            prior_weight=bay_cfg.get("prior_weight", 0.95),
            min_observations=bay_cfg.get("min_observations", 3),
        )

        self.edge_detector = EdgeDetector(
            min_edge=strat_cfg.get("min_edge", 0.03),
        )

        self.kelly = KellySizer(
            kelly_fraction=strat_cfg.get("kelly_fraction", 0.25),
            max_position_usdc=strat_cfg.get("max_position_usdc", 100),
            max_portfolio_usdc=strat_cfg.get("max_portfolio_usdc", 1000),
        )

        self.risk = RiskManager(
            max_position_usdc=strat_cfg.get("max_position_usdc", 100),
            max_portfolio_usdc=strat_cfg.get("max_portfolio_usdc", 1000),
            max_daily_loss_usdc=risk_cfg.get("max_daily_loss_usdc", 200),
            max_market_concentration=risk_cfg.get("max_market_concentration", 0.3),
            max_loss_per_market_usdc=risk_cfg.get("max_loss_per_market_usdc", 0.0),
        )

        self.positions = PositionManager(
            profit_target_pct=pos_cfg.get("profit_target_pct", 0.08),
            stop_loss_pct=pos_cfg.get("stop_loss_pct", 0.15),
            pre_resolution_hours=pos_cfg.get("pre_resolution_hours", 2.0),
            min_edge_to_hold=pos_cfg.get("min_edge_to_hold", 0.02),
            exit_max_retries=pos_cfg.get("exit_max_retries", 5),
            exit_price_step=pos_cfg.get("exit_price_step", 0.01),
            near_ceiling_price=pos_cfg.get("near_ceiling_price", 0.98),
        )

        self.redeemer = Redeemer(
            private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            wallet_address=os.getenv("POLYMARKET_WALLET_ADDRESS", ""),
        )
        self._redeem_interval_cycles = 20  # check redemption every 20 cycles (~5 min)
        self._cycle_count = 0

        self.news_feed = NewsFeed()
        self.loop_interval = config.get("trading", {}).get("loop_interval_s", 60)
        self._running = False

        # ── Force-close cooldown ────────────────────────────────────────────
        # Prevents re-entering a market after force-close; persists across restarts.
        self._cooldown_file = Path(__file__).parent.parent / "logs" / "force_close_cooldown.json"
        self._force_close_cooldown: dict[str, float] = self._load_cooldown()
        self._cooldown_hours = pos_cfg.get("force_close_cooldown_hours", 24)
        # ────────────────────────────────────────────────────────────────────

        # ── General exit cooldown ───────────────────────────────────────────
        # After ANY close (profit/loss/edge-reversal), block re-entry for N hours.
        self._exit_cooldown_file = Path(__file__).parent.parent / "logs" / "exit_cooldown.json"
        self._exit_cooldown: dict[str, float] = self._load_exit_cooldown()
        self._exit_cooldown_hours = pos_cfg.get("exit_cooldown_hours", 4)
        # ────────────────────────────────────────────────────────────────────

        # ── LLM confidence threshold ────────────────────────────────────────
        # Only trade when LLM confidence meets minimum (avoids default p̂ bets)
        self._min_llm_confidence = bay_cfg.get("min_confidence", 0.5)
        # ────────────────────────────────────────────────────────────────────

        # Adaptive strategy
        self.perf = PerformanceTracker()
        self._strat_cfg = strat_cfg  # live reference for adaptive updates

        # ── Pending orders (limit orders awaiting fill) ──────────────────────
        self._pending_orders_file = Path(__file__).parent.parent / "logs" / "pending_orders.json"
        self._pending_orders: list[dict] = self._load_pending_orders()
        self._pending_order_max_age_s = config.get("trading", {}).get("pending_order_max_age_s", 300)  # 5 min default
        # ────────────────────────────────────────────────────────────────────

        # ── On-chain reconciliation interval ─────────────────────────────────
        self._reconcile_interval_cycles = config.get("trading", {}).get("reconcile_interval_cycles", 10)
        self._wallet_address = os.getenv("POLYMARKET_WALLET_ADDRESS", "")
        # ────────────────────────────────────────────────────────────────────

        self._resync_portfolio_trackers()

    # ── Cooldown helpers ────────────────────────────────────────────────────

    def _load_cooldown(self) -> dict[str, float]:
        if self._cooldown_file.exists():
            try:
                return json.loads(self._cooldown_file.read_text())
            except Exception:
                pass
        return {}

    def _save_cooldown(self) -> None:
        self._cooldown_file.parent.mkdir(parents=True, exist_ok=True)
        self._cooldown_file.write_text(json.dumps(self._force_close_cooldown))

    def _add_cooldown(self, market_id: str) -> None:
        self._force_close_cooldown[market_id] = time.time()
        self._save_cooldown()
        logger.info(f"🚫 {market_id} added to force-close cooldown ({self._cooldown_hours}h)")

    def _in_cooldown(self, market_id: str) -> bool:
        ts = self._force_close_cooldown.get(market_id)
        if ts is None:
            return False
        elapsed_hours = (time.time() - ts) / 3600
        if elapsed_hours >= self._cooldown_hours:
            del self._force_close_cooldown[market_id]
            self._save_cooldown()
            return False
        return True

    # ── General exit cooldown helpers ────────────────────────────────────────

    def _load_exit_cooldown(self) -> dict[str, float]:
        if self._exit_cooldown_file.exists():
            try:
                return json.loads(self._exit_cooldown_file.read_text())
            except Exception:
                pass
        return {}

    def _save_exit_cooldown(self) -> None:
        self._exit_cooldown_file.parent.mkdir(parents=True, exist_ok=True)
        self._exit_cooldown_file.write_text(json.dumps(self._exit_cooldown))

    def _add_exit_cooldown(self, market_id: str) -> None:
        self._exit_cooldown[market_id] = time.time()
        self._save_exit_cooldown()
        logger.info(f"⏳ {market_id} exit-cooldown started ({self._exit_cooldown_hours}h)")

    def _in_exit_cooldown(self, market_id: str) -> bool:
        ts = self._exit_cooldown.get(market_id)
        if ts is None:
            return False
        elapsed_hours = (time.time() - ts) / 3600
        if elapsed_hours >= self._exit_cooldown_hours:
            del self._exit_cooldown[market_id]
            self._save_exit_cooldown()
            return False
        return True

    # ── Pending order helpers ─────────────────────────────────────────────

    def _load_pending_orders(self) -> list[dict]:
        if self._pending_orders_file.exists():
            try:
                return json.loads(self._pending_orders_file.read_text())
            except Exception:
                pass
        return []

    def _save_pending_orders(self) -> None:
        self._pending_orders_file.parent.mkdir(parents=True, exist_ok=True)
        self._pending_orders_file.write_text(json.dumps(self._pending_orders, indent=2))

    @staticmethod
    def _normalize_entry_order(size_usdc: float, entry_price: float, min_shares: float = 5.0) -> tuple[float, float]:
        """Return the actual order size/cost after minimum-order normalization."""
        if entry_price <= 0:
            return 0.0, 0.0

        shares = size_usdc / entry_price if size_usdc > 0 else 0.0
        if 0 < shares < min_shares:
            shares = float(min_shares)
        order_cost = shares * entry_price if shares > 0 else 0.0
        return shares, order_cost

    def _add_pending_order(self, order_id: str, opp, market, token_id: str, entry_price: float, shares: float) -> None:
        self._pending_orders.append({
            "order_id": order_id,
            "market_id": opp.market_id,
            "condition_id": market.condition_id,
            "token_id": token_id,
            "side": opp.side,
            "entry_price": entry_price,
            "size": shares,
            "p_hat": opp.p_hat,
            "market_price": opp.market_price,
            "end_date": market.end_date,
            "created_at": time.time(),
            "recorded_filled_size": 0.0,
        })
        self._save_pending_orders()
        logger.info(f"📋 Pending order saved: {opp.market_id} {opp.side} order_id={order_id}")

    def _check_pending_orders(self) -> None:
        """Check all pending orders for fills; record positions or cancel stale ones."""
        if not self._pending_orders:
            # Reset the recently-confirmed token set at start of each cycle
            self._recently_confirmed_tokens = set()
            return

        # Track tokens that are newly confirmed in this cycle (for phantom-discovery exclusion)
        self._recently_confirmed_tokens = set()

        remaining = []
        for po in self._pending_orders:
            age_s = time.time() - po["created_at"]
            order_id = po["order_id"]

            # Build a minimal Order object to query status
            from src.execution.clob_executor import Order, OrderStatus as OS
            order = Order(
                order_id=order_id,
                token_id=po["token_id"],
                side=OrderSide.BUY,
                price=po["entry_price"],
                size=po["size"],
            )
            order = self.executor.refresh_order_status(order)

            if order.status in (OS.FILLED, OS.PARTIALLY_FILLED) and order.filled_size > 0:
                avg_price = order.filled_avg_price or po["entry_price"]
                recorded_filled_size = float(po.get("recorded_filled_size", 0.0) or 0.0)
                actual_size = max(0.0, order.filled_size - recorded_filled_size)
                if actual_size < 0.01:
                    if order.status == OS.PARTIALLY_FILLED:
                        remaining.append(po)
                    continue
                actual_size_usdc = round(actual_size * avg_price, 4)

                position = Position(
                    market_id=po["market_id"],
                    condition_id=po["condition_id"],
                    token_id=po["token_id"],
                    side=po["side"],
                    entry_price=avg_price,
                    size=actual_size,
                    size_usdc=actual_size_usdc,
                    p_hat_at_entry=po["p_hat"],
                    market_price_at_entry=po["market_price"],
                    end_date=po.get("end_date", ""),
                )
                self.positions.add_position(position)
                self.risk.record_position(po["market_id"], actual_size_usdc)
                self.kelly.record_position(po["market_id"], actual_size_usdc)
                # Mark this token as recently confirmed to prevent phantom-discovery
                # from adding it again during this reconcile window
                self._recently_confirmed_tokens.add(po["token_id"])

                fill_type = "fully" if order.status == OS.FILLED else "partially"
                logger.info(
                    f"✅ Pending order {fill_type} filled: {po['market_id']} {po['side']} "
                    f"{actual_size:.2f} shares @ ${avg_price:.3f} (${actual_size_usdc:.2f} USDC)"
                )

                # If partially filled, keep tracking the remainder
                if order.status == OS.PARTIALLY_FILLED:
                    po["size"] = max(0.0, po["size"] - actual_size)
                    po["recorded_filled_size"] = order.filled_size
                    remaining.append(po)
                continue

            if order.status in (OS.CANCELLED, OS.FAILED):
                logger.warning(f"❌ Pending order {order_id} {po['market_id']}: {order.status.value}, removing")
                continue

            # Still open — check age
            if age_s > self._pending_order_max_age_s:
                # Before cancelling, check if the order partially filled on-chain
                # (CLOB status can lag behind on-chain settlement).
                onchain_bal = self.executor._onchain_token_balance(po["token_id"])
                if onchain_bal != float("inf") and onchain_bal >= 0.01:
                    # Tokens appeared on-chain — the order DID fill (at least partially).
                    # Record the position from on-chain truth instead of cancelling.
                    avg_price = po["entry_price"]
                    actual_size = onchain_bal
                    # Subtract any pre-existing position size for this token
                    existing_pos = self.positions.get_position(po["market_id"])
                    if existing_pos:
                        actual_size = onchain_bal - existing_pos.size
                    if actual_size >= 0.01:
                        actual_size_usdc = round(actual_size * avg_price, 4)
                        position = Position(
                            market_id=po["market_id"],
                            condition_id=po["condition_id"],
                            token_id=po["token_id"],
                            side=po["side"],
                            entry_price=avg_price,
                            size=actual_size,
                            size_usdc=actual_size_usdc,
                            p_hat_at_entry=po["p_hat"],
                            market_price_at_entry=po["market_price"],
                            end_date=po.get("end_date", ""),
                        )
                        self.positions.add_position(position)
                        self.risk.record_position(po["market_id"], actual_size_usdc)
                        self.kelly.record_position(po["market_id"], actual_size_usdc)
                        self._recently_confirmed_tokens.add(po["token_id"])
                        logger.info(
                            f"✅ Pending order {order_id} {po['market_id']} filled on-chain: "
                            f"{actual_size:.2f} shares @ ${avg_price:.3f} "
                            f"(CLOB status lagged, detected via on-chain balance)"
                        )
                    else:
                        logger.info(
                            f"✅ Pending order {order_id} {po['market_id']} already fully "
                            f"accounted for via earlier partial fills; cancelling remainder"
                        )
                    # Cancel any remaining unfilled portion
                    self.executor.cancel_order(order_id)
                    continue

                logger.warning(
                    f"⏰ Pending order {order_id} {po['market_id']} expired after "
                    f"{age_s:.0f}s — cancelling"
                )
                self.executor.cancel_order(order_id)
                # Add exit cooldown to prevent immediate re-entry for the same market.
                self._add_exit_cooldown(po["market_id"])
                continue

            # Still pending, keep tracking
            remaining.append(po)

        self._pending_orders = remaining
        self._save_pending_orders()

    # ── Periodic on-chain reconciliation ─────────────────────────────────

    async def _get_best_bid(self, token_id: str, fallback_price: float) -> float:
        """Fetch best bid from the CLOB orderbook for a token.

        For exit orders we want to sell INTO the bid side so the order fills
        immediately, rather than placing a limit sell at the mid/ask price
        that sits on the book unfilled.

        Falls back to (fallback_price - 2 ticks) if orderbook fetch fails.
        """
        try:
            ob = await self.client.get_orderbook(token_id)
            if ob.bids:
                best_bid = ob.best_bid
                logger.info(
                    f"📕 Best bid for {token_id[:16]}...: ${best_bid:.4f} "
                    f"(mid would be ${fallback_price:.4f})"
                )
                return best_bid
        except Exception as e:
            logger.warning(f"Orderbook fetch failed for {token_id[:16]}...: {e}")

        # Fallback: discount the mid price by 2 ticks to cross the spread
        discounted = round(fallback_price - 0.02, 4)
        logger.info(
            f"📕 Using discounted price for {token_id[:16]}...: "
            f"${discounted:.4f} (mid=${fallback_price:.4f} - 2 ticks)"
        )
        return max(discounted, 0.01)

    async def _ensure_position_prices(
        self, prices: dict[str, float], market_map: dict
    ) -> None:
        """Fetch current prices for open positions missing from the market discovery results.

        Without this, positions in low-volume or delisted markets never get exit checks
        because their market_id isn't in the prices dict.
        """
        missing = [
            pos for pos in self.positions.open_positions
            if pos.market_id not in prices
        ]
        if not missing:
            return

        logger.info(
            f"🔍 Fetching prices for {len(missing)} position(s) not in market discovery"
        )
        for pos in missing:
            try:
                market = await self.client.get_market(pos.market_id)
                if market is None:
                    # Fallback: try by token_id (conditionId filter is broken on Gamma API)
                    market = await self.client.get_market_by_token(pos.token_id)
                if market:
                    prices[market.id] = market.yes_price
                    # Also add to market_map so _execute_exit can find token_id etc.
                    market_map[market.id] = market
                    # If market_id in position differs from fetched id, add both keys
                    if pos.market_id != market.id:
                        prices[pos.market_id] = market.yes_price
                        market_map[pos.market_id] = market
                    logger.info(
                        f"  📈 {pos.market_id}: YES={market.yes_price:.3f} "
                        f"(pos {pos.side} entry=${pos.entry_price:.3f})"
                    )
                else:
                    logger.warning(f"  ⚠️ Could not fetch price for position {pos.market_id}")
            except Exception as e:
                logger.warning(f"  ⚠️ Price fetch failed for {pos.market_id}: {e}")

    async def _periodic_onchain_reconcile(self) -> None:
        """Reconcile local positions against on-chain state (runs every N cycles).

        1. Verify existing positions: adjust size or remove if on-chain balance differs.
        2. Discover phantom positions: tokens held on-chain but missing from local state.
        """
        if self.dry_run or not self._wallet_address:
            return

        logger.info("🔗 Periodic on-chain reconciliation start")
        state_changed = False
        adjusted = 0
        removed = 0

        # ── Step 1: Verify existing local positions against on-chain ──
        for pos in list(self.positions.open_positions):
            # Check if market has resolved — remove losing positions, redeem winners
            if pos.condition_id:
                resolution = self.redeemer.is_resolved(pos.condition_id)
                if resolution is not None:
                    won = (resolution["payout_yes"] > 0) if pos.side == "YES" else (resolution["payout_no"] > 0)
                    logger.warning(
                        f"🏁 On-chain reconcile: {pos.market_id} RESOLVED — "
                        f"{pos.side} {'WON' if won else 'LOST'}"
                    )
                    if not won:
                        # Losing position — tokens are worthless, just remove
                        pnl_loss = -pos.size_usdc
                        self.positions.close_position(pos.market_id)
                        self.risk.close_position(pos.market_id)
                        self.kelly.close_position(pos.market_id)
                        self.risk.record_pnl(pnl_loss)
                        self._add_exit_cooldown(pos.market_id)
                        # ── Record to trade history ──────────────────────
                        _trade = ClosedTrade(
                            market_id=pos.market_id,
                            side=pos.side,
                            entry_price=pos.entry_price,
                            exit_price=0.0,
                            size_usdc=pos.size_usdc,
                            realized_pnl=round(pnl_loss, 4),
                            realized_pnl_pct=-1.0,
                            exit_reason="resolution_reconcile",
                        )
                        self.perf.record_close(_trade)
                        removed += 1
                        state_changed = True
                        continue
                    # Won — leave for the existing _check_redemptions to handle
                    continue

            actual_balance = self.executor._onchain_token_balance(pos.token_id)
            if actual_balance == float("inf"):
                continue  # RPC failed, skip

            if actual_balance < 0.01:
                logger.warning(
                    f"🧹 On-chain reconcile: removing {pos.market_id} — "
                    f"no on-chain balance for token {pos.token_id[:16]}..."
                )
                # Absence on-chain only tells us the position is gone, not how it exited.
                # Do not fabricate realized PnL or a synthetic close price here.
                self.positions.close_position(pos.market_id)
                self.risk.close_position(pos.market_id)
                self.kelly.close_position(pos.market_id)
                self._add_exit_cooldown(pos.market_id)
                removed += 1
                state_changed = True
                continue

            if abs(actual_balance - pos.size) > 0.01:
                old_size = pos.size
                old_usdc = pos.size_usdc
                scale = (actual_balance / pos.size) if pos.size > 0 else 1.0
                pos.size = round(actual_balance, 6)
                pos.size_usdc = round(old_usdc * scale, 4)
                pos.exit_retries = 0
                pos.exit_price_override = 0.0
                pos.force_close_failed = False
                logger.warning(
                    f"📐 On-chain reconcile: {pos.market_id} shares "
                    f"{old_size:.4f}→{pos.size:.4f}, "
                    f"notional ${old_usdc:.2f}→${pos.size_usdc:.2f}"
                )
                adjusted += 1
                state_changed = True

        # ── Step 2: Discover phantom positions via data API ──
        # Tokens held on-chain but not in local positions_state.json
        tracked_tokens = {p.token_id for p in self.positions.open_positions}
        # Also exclude tokens from pending orders (not yet confirmed)
        pending_tokens = {po.get("token_id", "") for po in self._pending_orders}
        tracked_tokens |= pending_tokens
        # CRITICAL: Also exclude tokens that were just added in _check_pending_orders
        # in this very cycle, to prevent phantom-discovery from double-adding them
        # during the 30s on-chain settlement window.
        # Without this, the same on-chain token gets added twice: once as a normal
        # position from pending order fill, then again as a "phantom" position from
        # the on-chain discovery scan, causing 2x position size and rapid buy-sell loops.
        recently_confirmed_tokens = getattr(self, "_recently_confirmed_tokens", set())
        tracked_tokens |= recently_confirmed_tokens

        try:
            remote_positions = await self.client.get_wallet_positions(self._wallet_address)
        except Exception as e:
            logger.warning(f"Phantom position discovery failed: {e}")
            remote_positions = []

        phantoms_found = 0
        # Track market_ids already added as phantoms this cycle.
        # Prevents accumulation bug: parent markets with multiple CTF conditions
        # (e.g. market_id="12" with 4 child tokens) would otherwise accumulate
        # all child tokens into one giant position, vastly exceeding max_position_usdc.
        phantom_market_ids_this_cycle: set[str] = set()

        for rp in remote_positions:
            token_id = rp.get("asset", "")
            if not token_id or token_id in tracked_tokens:
                continue

            # Verify on-chain balance
            onchain_size = self.executor._onchain_token_balance(token_id)
            if onchain_size == float("inf") or onchain_size < 0.01:
                continue

            # Skip resolved markets — losing tokens still have raw balance
            # on-chain but are worthless; don't add them as phantom positions.
            condition_id = rp.get("conditionId", "")
            if condition_id:
                try:
                    resolution = self.redeemer.is_resolved(condition_id)
                    if resolution is not None:
                        logger.info(
                            f"⏭️  Skipping phantom {token_id[:16]}... — "
                            f"market already resolved"
                        )
                        continue
                except Exception:
                    pass
            market_id = rp.get("market", "") or rp.get("marketId", "") or condition_id
            side = rp.get("side", "YES").upper()
            if side not in ("YES", "NO"):
                side = "YES"

            # Skip if this market is in exit cooldown (recently closed/zombie-cleaned)
            if self._in_exit_cooldown(market_id):
                logger.debug(f"⏭️  Skipping phantom {token_id[:16]}... — exit cooldown active")
                continue

            # Try to get entry price from the API position data
            avg_price = float(rp.get("avgPrice", 0) or 0)
            if avg_price <= 0:
                avg_price = float(rp.get("price", 0) or 0)
            if avg_price <= 0:
                # Fallback: estimate from current market price
                avg_price = 0.50

            size_usdc = round(onchain_size * avg_price, 4)

            # Fetch market metadata by token_id (conditionId filter is broken on Gamma API)
            end_date = ""
            if token_id:
                try:
                    market_info = await self.client.get_market_by_token(token_id)
                    if market_info:
                        market_id = market_info.id or market_id
                        end_date = market_info.end_date or ""
                except Exception:
                    pass

            # Skip phantom positions whose end_date is >48h in the past
            # (zombie markets — tokens are worthless or already redeemed)
            if end_date:
                try:
                    from datetime import datetime, timezone
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    hours_past = (datetime.now(timezone.utc) - end_dt).total_seconds() / 3600
                    if hours_past > 48:
                        logger.info(
                            f"⏭️  Skipping phantom {token_id[:16]}... — "
                            f"end_date {end_date} is {hours_past:.0f}h in the past (zombie)"
                        )
                        continue
                except Exception:
                    pass

            # SAFETY: Skip if we already added a phantom for this market_id this cycle.
            # Multi-outcome/parent markets can have many child tokens all sharing the same
            # market_id; accumulating them would blow the max_position_usdc limit.
            if market_id in phantom_market_ids_this_cycle:
                logger.warning(
                    f"⏭️  Skipping phantom {token_id[:16]}... — "
                    f"market {market_id} already added this cycle (multi-token parent market guard)"
                )
                continue

            # SAFETY: Cap phantom size_usdc to max_position_usdc for risk tracking.
            # The actual on-chain size is still recorded in pos.size for correct exit sizing,
            # but risk/Kelly exposure is capped to avoid portfolio overexposure from legacy tokens.
            max_pos = self.config.get("strategy", {}).get("max_position_usdc", 5.0)
            if size_usdc > max_pos:
                logger.warning(
                    f"⚠️  Phantom {market_id} size ${size_usdc:.2f} > max ${max_pos:.2f} — "
                    f"capping tracked USDC to ${max_pos:.2f} (on-chain size={onchain_size:.2f} kept)"
                )
                tracked_size_usdc = max_pos
            else:
                tracked_size_usdc = size_usdc

            position = Position(
                market_id=market_id,
                condition_id=condition_id,
                token_id=token_id,
                side=side,
                entry_price=avg_price,
                size=round(onchain_size, 6),
                size_usdc=size_usdc,
                p_hat_at_entry=avg_price,
                market_price_at_entry=avg_price,
                end_date=end_date,
            )
            self.positions.add_position(position)
            self.risk.record_position(market_id, tracked_size_usdc)
            self.kelly.record_position(market_id, tracked_size_usdc)
            phantom_market_ids_this_cycle.add(market_id)
            phantoms_found += 1
            state_changed = True

            logger.warning(
                f"👻 Phantom position discovered: {market_id} {side} "
                f"{onchain_size:.2f} shares @ ~${avg_price:.3f} "
                f"(${size_usdc:.2f} USDC) — added to local state"
            )

        if state_changed:
            self.positions._save_state()
            self._resync_portfolio_trackers()

        logger.info(
            f"🔗 On-chain reconcile done: adjusted={adjusted}, removed={removed}, "
            f"phantoms={phantoms_found}, open={len(self.positions.open_positions)}"
        )

    # ────────────────────────────────────────────────────────────────────────

    def _resync_portfolio_trackers(self) -> None:
        """Rebuild in-memory exposure trackers from persisted open positions."""
        self.risk.reset_positions()
        self.kelly.reset_positions()
        restored = self.positions.open_positions
        if restored:
            logger.info(
                f"🔄 Syncing {len(restored)} persisted positions → RiskManager & KellySizer"
            )
        for pos in restored:
            self.risk.record_position(pos.market_id, pos.size_usdc)
            self.kelly.record_position(pos.market_id, pos.size_usdc)

    def _set_portfolio_exposure(self, market_id: str, size_usdc: float) -> None:
        """Keep RiskManager and KellySizer aligned with the current position size."""
        self.risk.set_position(market_id, size_usdc)
        self.kelly.set_position(market_id, size_usdc)

    def _reconcile_startup_state(self) -> None:
        """Clean up crash leftovers before trading resumes.

        Reconciliation steps:
        0. Remove zombie positions whose end_date has long passed.
        1. Cancel remote live orders left behind by a crashed process.
        2. Sync CLOB collateral balances.
        3. Reconcile persisted positions against on-chain token balances.
        4. Rebuild in-memory risk/Kelly exposure from the reconciled state.
        """
        logger.info("🩺 Startup reconciliation begin")

        # ── Step 0: Remove zombie positions (end_date far in the past) ────
        # Markets whose resolution date is >48h in the past are almost certainly
        # resolved.  Keeping them causes token-collision bugs when the same token
        # ID is reused by a newer market (Polymarket reuses CTF token IDs across
        # markets/conditions).
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        for pos in list(self.positions.open_positions):
            if pos.end_date:
                try:
                    end_dt = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00"))
                    hours_past = (now - end_dt).total_seconds() / 3600
                    if hours_past > 48:
                        logger.warning(
                            f"🧟 Removing zombie position {pos.market_id} — "
                            f"end_date {pos.end_date} is {hours_past:.0f}h in the past"
                        )
                        self.positions.close_position(pos.market_id)
                        self._add_exit_cooldown(pos.market_id)
                except Exception:
                    pass

        # Check any pending orders from last session before cancelling
        if self._pending_orders:
            logger.info(f"📋 Checking {len(self._pending_orders)} pending order(s) from last session")
            self._check_pending_orders()

        cancelled_orders = self.executor.cancel_all_live_orders()
        if cancelled_orders:
            logger.warning(
                f"🧹 Startup reconciliation cancelled {cancelled_orders} live remote order(s)"
            )

        if self.dry_run:
            self._resync_portfolio_trackers()
            logger.info("🩺 Startup reconciliation complete (dry run)")
            return

        self.executor._sync_clob_balance()

        adjusted = 0
        removed = 0
        state_changed = False
        for pos in list(self.positions.open_positions):
            # Check if market already resolved — clean up immediately
            if pos.condition_id:
                try:
                    resolution = self.redeemer.is_resolved(pos.condition_id)
                    if resolution is not None:
                        won = (resolution["payout_yes"] > 0) if pos.side == "YES" else (resolution["payout_no"] > 0)
                        logger.warning(
                            f"🏁 Startup: {pos.market_id} RESOLVED — "
                            f"{pos.side} {'WON' if won else 'LOST'}, removing from local state"
                        )
                        if not won:
                            self.risk.record_pnl(-pos.size_usdc)
                        self.positions.close_position(pos.market_id)
                        self._add_exit_cooldown(pos.market_id)
                        removed += 1
                        state_changed = True
                        continue
                except Exception as e:
                    logger.debug(f"Resolution check failed for {pos.market_id}: {e}")

            actual_balance = self.executor._onchain_token_balance(pos.token_id)
            if actual_balance == float("inf"):
                logger.warning(
                    f"Startup reconciliation skipped balance check for {pos.market_id} "
                    f"(RPC unavailable)"
                )
                continue

            if actual_balance < 0.01:
                logger.warning(
                    f"🧹 Removing stale local position {pos.market_id}: "
                    f"no on-chain balance for token {pos.token_id[:16]}..."
                )
                self.positions.close_position(pos.market_id)
                self._add_exit_cooldown(pos.market_id)
                removed += 1
                state_changed = True
                continue

            if abs(actual_balance - pos.size) > 0.01:
                old_size = pos.size
                old_usdc = pos.size_usdc
                scale = (actual_balance / pos.size) if pos.size > 0 else 1.0
                pos.size = round(actual_balance, 6)
                pos.size_usdc = round(old_usdc * scale, 4)
                pos.exit_retries = 0
                pos.exit_price_override = 0.0
                pos.force_close_failed = False
                logger.warning(
                    f"📐 Reconciled {pos.market_id}: shares {old_size:.4f}→{pos.size:.4f}, "
                    f"notional ${old_usdc:.2f}→${pos.size_usdc:.2f}"
                )
                adjusted += 1
                state_changed = True

        if state_changed:
            self.positions._save_state()

        self._resync_portfolio_trackers()
        logger.info(
            f"🩺 Startup reconciliation complete: adjusted={adjusted}, removed={removed}, "
            f"open_positions={len(self.positions.open_positions)}"
        )

    async def run_cycle(self) -> None:
        t0 = time.perf_counter()

        # ─── 0. Check pending orders from previous cycles ─────
        self._check_pending_orders()

        # ─── 1. Fetch markets ─────────────────────────────────
        markets = await self.client.get_markets(limit=100, min_volume=1000)

        tradeable = [m for m in markets if 0.10 <= m.yes_price <= 0.90]
        logger.info(f"📊 {len(markets)} markets → {len(tradeable)} tradeable (10-90%)")

        # ─── 2. Update beliefs + detect edges ─────────────────
        market_map = {m.id: m for m in tradeable}

        # Init beliefs for new markets
        for market in tradeable:
            if self.bayesian.get_belief(market.id) is None:
                self.bayesian.init_belief(market.id, market.yes_price)

        # ─── Fast path: check exits with current prices BEFORE LLM ──
        # Stops/TPs use price alone — no need to wait for LLM signals.
        # Use ALL fetched markets (not just tradeable) so positions in markets
        # that moved outside 10-90% range (e.g. near resolution) still get exit checks.
        current_prices_fast = {m.id: m.yes_price for m in markets}
        current_p_hats_fast = {
            m.id: (self.bayesian.get_belief(m.id).p_hat
                   if self.bayesian.get_belief(m.id) else m.yes_price)
            for m in tradeable
        }

        # Ensure all open positions have current prices — fetch individually
        # for any position whose market wasn't in the discovery results
        # (e.g. low volume, fell out of top 100). Without this, exits never trigger.
        await self._ensure_position_prices(current_prices_fast, market_map)

        for exit_sig in self.positions.check_exits(
            current_prices_fast, current_p_hats_fast, skip_edge_reversal=True
        ):
            await self._execute_exit(
                exit_sig.position, exit_sig.current_price, market_map, exit_sig.reason.value
            )

        # Parallel LLM/news fetch for all markets
        all_signals = await asyncio.gather(
            *[self.news_feed.get_llm_signals(m) for m in tradeable],
            return_exceptions=True,
        )

        scan_data = []
        for market, news_signals in zip(tradeable, all_signals):
            if isinstance(news_signals, Exception):
                logger.warning(f"LLM fetch error for {market.id}: {news_signals}")
                news_signals = []
            if news_signals:
                self.bayesian.batch_update(market.id, news_signals)
                logger.info(f"   🧠 {market.id}: {len(news_signals)} LLM signals applied")

            belief = self.bayesian.get_belief(market.id)

            # Calculate time to resolution from end_date (prevents entries near expiry)
            hours_to_resolution = float("inf")
            if market.end_date:
                try:
                    end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                    hours_to_resolution = max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600)
                except Exception:
                    pass

            scan_data.append({
                "market_id": market.id,
                "question": market.question,
                "p_hat": belief.p_hat if belief else market.yes_price,
                "market_price_yes": market.yes_price,
                "volume_24h": market.volume_24h,
                "confidence": 1.0,
                "time_to_resolution_hours": hours_to_resolution,
            })

        # ─── 3. Edge-reversal exits (after LLM update) ───────
        # Same as above: use all markets so out-of-range positions still get exits.
        current_prices = {m.id: m.yes_price for m in markets}
        current_p_hats = {d["market_id"]: d["p_hat"] for d in scan_data}
        # Re-use already-fetched position prices from fast path (market_map already populated)
        await self._ensure_position_prices(current_prices, market_map)
        for exit_sig in self.positions.check_exits(current_prices, current_p_hats):
            pos = exit_sig.position
            await self._execute_exit(pos, exit_sig.current_price, market_map, exit_sig.reason.value)

        # ─── 4. Find new opportunities ─────────────────────────
        opportunities = self.edge_detector.scan_markets(scan_data)

        if not opportunities:
            logger.info("😴 No edge this cycle")
        else:
            # Build set of markets with pending buy orders to prevent duplicate entries
            pending_market_ids = {po["market_id"] for po in self._pending_orders}

            for opp in opportunities:
                # Skip if already have a position in this market
                if self.positions.get_position(opp.market_id):
                    continue

                # Skip if a buy order is already pending for this market
                if opp.market_id in pending_market_ids:
                    logger.debug(f"⏳ {opp.market_id} skipped — pending buy order exists")
                    continue

                # Skip if market is in force-close cooldown
                if self._in_cooldown(opp.market_id):
                    logger.debug(f"⏳ {opp.market_id} skipped — force-close cooldown active")
                    continue

                # Skip if market is in general exit cooldown (any recent close)
                if self._in_exit_cooldown(opp.market_id):
                    logger.debug(f"⏳ {opp.market_id} skipped — exit cooldown active ({self._exit_cooldown_hours}h)")
                    continue

                if not self.bayesian.is_tradeable(opp.market_id):
                    continue

                # Skip if LLM confidence is too low (avoid default-p̂ bets).
                # Use a RECENT window (last 5 signals) rather than all-time average —
                # the all-time average permanently dilutes as low-confidence signals
                # accumulate over many cycles, permanently blocking valid opportunities.
                belief = self.bayesian.get_belief(opp.market_id)
                if belief and belief.update_history:
                    recent_signals = belief.update_history[-5:]  # last 5 signals only
                    avg_conf = sum(s.get("confidence", 0) for s in recent_signals) / len(recent_signals)
                else:
                    avg_conf = 0.0
                if avg_conf < self._min_llm_confidence:
                    logger.info(
                        f"⏭️  {opp.market_id} skipped — recent LLM confidence {avg_conf:.2f} "
                        f"< min {self._min_llm_confidence:.2f}"
                    )
                    continue
                logger.info(
                    f"✅ {opp.market_id} {opp.side} approved — edge={opp.edge:+.3f} "
                    f"p̂={opp.p_hat:.3f} recent_conf={avg_conf:.2f}"
                )

                # Kelly sizing
                bankroll = self.kelly.remaining_capacity
                pos_size = self.kelly.calculate(
                    market_id=opp.market_id,
                    side=opp.side,
                    p_hat=opp.p_hat,
                    market_price=opp.market_price,
                    bankroll=bankroll,
                )
                if pos_size.position_usdc <= 0:
                    continue

                entry_price_est = opp.market_price  # EdgeDetector already normalised to entry side
                if entry_price_est >= self.positions.near_ceiling_price:
                    logger.info(
                        f"⏭️  {opp.market_id} {opp.side} skipped: entry price "
                        f"{entry_price_est:.2f} >= near-ceiling {self.positions.near_ceiling_price:.2f} "
                        f"(upside too small to justify entry)"
                    )
                    continue

                normalized_shares, normalized_cost = self._normalize_entry_order(
                    pos_size.position_usdc, entry_price_est
                )
                if normalized_shares <= 0:
                    continue

                remaining = self.kelly.remaining_capacity
                if normalized_cost > remaining:
                    logger.info(
                        f"⏭️  Skipping {opp.market_id} {opp.side}: "
                        f"actual entry cost ${normalized_cost:.2f} > remaining capacity "
                        f"${remaining:.2f}"
                    )
                    continue

                # Risk check
                risk_check = self.risk.validate_trade(opp.market_id, normalized_cost)
                if not risk_check.approved:
                    logger.warning(f"❌ Risk rejected {opp.market_id}: {risk_check.reason}")
                    continue

                if not self.dry_run:
                    onchain_balance = self.executor._onchain_usdc_balance(fail_closed=False)
                    if onchain_balance is None:
                        logger.warning(
                            f"⚠️  On-chain balance pre-check unavailable for {opp.market_id} "
                            f"(continuing to executor checks)"
                        )
                    elif onchain_balance < normalized_cost:
                        logger.info(
                            f"⏭️  {opp.market_id} skipped: on-chain balance ${onchain_balance:.2f} "
                            f"< order cost ${normalized_cost:.2f}"
                        )
                        continue

                # Execute entry
                await self._execute_entry(opp, normalized_cost, market_map)

        # ─── 5. Portfolio summary ──────────────────────────────
        summary = self.positions.summary()
        if summary["open_positions"] > 0:
            logger.info(
                f"📂 Portfolio: {summary['open_positions']} positions, "
                f"${summary['total_exposure_usdc']:.2f} exposure"
            )

        # ─── 6. Auto-redeem resolved positions ──────────────
        self._cycle_count += 1
        if self._cycle_count % self._redeem_interval_cycles == 0:
            await self._check_redemptions()
            # After possible redemption, resync CLOB balance so freed
            # USDC.e is visible for the next entry attempt.
            self.executor._sync_clob_balance()

        # ─── 7. Periodic on-chain position reconciliation ────
        if self._cycle_count % self._reconcile_interval_cycles == 0:
            await self._periodic_onchain_reconcile()

        elapsed = time.perf_counter() - t0
        logger.info(f"⏱️ Cycle in {elapsed:.2f}s")

    async def _execute_entry(self, opp, size_usdc: float, market_map: dict) -> None:
        """Execute a BUY order for a detected edge."""
        market = market_map.get(opp.market_id)
        if not market:
            return

        # Correct token and side based on YES/NO direction
        if opp.side == "YES":
            token_id = market.yes_token_id
            entry_price = market.yes_price
        else:
            token_id = market.no_token_id
            entry_price = 1.0 - market.yes_price

        shares, _ = self._normalize_entry_order(size_usdc, entry_price)
        if shares <= 0:
            return

        # Pre-flight balance/allowance check (avoid API rejection)
        if not self.executor.has_sufficient_balance(
            side=OrderSide.BUY, price=entry_price, size=shares, token_id=token_id
        ):
            logger.info(
                f"⏭️  Entry skipped for {opp.market_id}: insufficient USDC.e balance/allowance"
            )
            return

        order = self.executor.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=entry_price,
            size=shares,
            condition_id=market.condition_id,
        )

        if order.status == OrderStatus.FAILED:
            logger.warning(f"❌ Entry order failed for {opp.market_id}")
            return

        # Poll for fill: limit orders may not fill instantly
        poll_attempts = 6  # 6 x 5s = 30s max wait
        poll_interval = 5
        for attempt in range(poll_attempts):
            order = self.executor.refresh_order_status(order)
            if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                break
            if order.status in (OrderStatus.CANCELLED, OrderStatus.FAILED):
                break
            if attempt < poll_attempts - 1:
                logger.info(
                    f"⏳ Waiting for fill {opp.market_id}: "
                    f"attempt {attempt + 1}/{poll_attempts}, status={order.status.value}"
                )
                await asyncio.sleep(poll_interval)

        if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED) and order.filled_size > 0:
            # Use actual executed size/avg price (may differ from Kelly size or be partial)
            actual_size = order.filled_size
            avg_price = order.filled_avg_price or entry_price
            actual_size_usdc = round(actual_size * avg_price, 4)

            position = Position(
                market_id=opp.market_id,
                condition_id=market.condition_id,
                token_id=token_id,
                side=opp.side,
                entry_price=avg_price,
                size=actual_size,
                size_usdc=actual_size_usdc,
                p_hat_at_entry=opp.p_hat,
                market_price_at_entry=opp.market_price,
                end_date=market.end_date,
            )
            self.positions.add_position(position)
            self.risk.record_position(opp.market_id, actual_size_usdc)
            self.kelly.record_position(opp.market_id, actual_size_usdc)

            if order.status == OrderStatus.PARTIALLY_FILLED:
                logger.warning(
                    f"⚠️  Entry partially filled for {opp.market_id}: "
                    f"{actual_size:.2f}/{order.size:.2f} shares"
                )

            self._print_entry(opp, avg_price, actual_size, actual_size_usdc, order)
        elif order.status in (OrderStatus.OPEN, OrderStatus.PENDING):
            # Order still on the book — save as pending for next cycle check
            self._add_pending_order(order.order_id, opp, market, token_id, entry_price, shares)
        else:
            logger.warning(
                f"❌ Entry order {opp.market_id}: status={order.status.value}, not recording position"
            )

    async def _execute_exit(self, pos: Position, current_price: float, market_map: dict, reason: str) -> None:
        """Execute a SELL order to close a position.

        On failure: lower price by exit_price_step and retry next cycle.
        After exit_max_retries failures: force-close in tracker and accept the loss.
        """
        from src.execution.clob_executor import OrderStatus

        max_retries = self.positions.exit_max_retries
        price_step  = self.positions.exit_price_step

        # Use discounted price if previous retries failed.
        # Safety floor: exit_price_override must not drop below 30% of current market price
        # (prevents the death-spiral where repeated CLOB failures drive price to $0.01
        # making the order unfillable even when market has liquidity).
        if pos.exit_price_override > 0:
            price_floor = max(round(current_price * 0.30, 4), 0.01)
            if pos.exit_price_override < price_floor:
                logger.warning(
                    f"⚠️  exit_price_override ${pos.exit_price_override:.4f} below floor "
                    f"${price_floor:.4f} for {pos.market_id} — resetting to market price"
                )
                pos.exit_price_override = 0.0
                pos.exit_retries = 0
                self.positions._save_state()
        if pos.exit_price_override > 0:
            sell_price = pos.exit_price_override
        else:
            # Pricing strategy depends on exit urgency:
            #   - profit_target: conservative (mid - 1 tick), preserve gains
            #   - stop_loss / pre_resolution / edge_reversal: aggressive (best bid), speed matters
            urgent = reason in ("stop_loss", "pre_resolution", "edge_reversal")
            if urgent:
                sell_price = await self._get_best_bid(pos.token_id, current_price)
            else:
                # Profit target: start conservative, the retry mechanism will
                # lower the price by 1 tick each cycle if it doesn't fill.
                meta = self.executor.fetch_market_meta(pos.condition_id) if pos.condition_id else {}
                tick = float(meta.get("tick_size", "0.01"))
                sell_price = round(current_price - tick, 4)
        # But never go below 1 tick (avoid 0-price order)
        sell_price = max(round(sell_price, 4), 0.01)

        if pos.force_close_failed:
            # Before retrying, sync CLOB balance allowance for this token
            # (the "not enough balance/allowance" error is often a CLOB cache miss)
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                if self.executor._clob_client:
                    self.executor._clob_client.update_balance_allowance(
                        params=BalanceAllowanceParams(
                            asset_type=AssetType.CONDITIONAL,
                            token_id=pos.token_id,
                            signature_type=0,
                        )
                    )
                    # Reset retry state after successful sync
                    pos.exit_retries = 0
                    pos.exit_price_override = 0.0
                    pos.force_close_failed = False
                    self.positions._save_state()
                    logger.info(
                        f"🔄 CLOB allowance resynced for {pos.market_id} — "
                        f"reset exit retries, retrying sell"
                    )
            except Exception as sync_err:
                logger.warning(
                    f"🚨 force_close_failed still active for {pos.market_id} "
                    f"(CLOB sync failed: {sync_err}) — manual intervention may be required"
                )

        # Pre-flight: use on-chain balance as sell size (source of truth)
        actual_balance = self.executor._onchain_token_balance(pos.token_id)
        if actual_balance == float("inf"):
            actual_balance = pos.size  # RPC failed, fallback
        sell_size = actual_balance
        if sell_size < 0.01:
            logger.warning(
                f"⏭️  Exit skipped for {pos.market_id}: on-chain balance too low ({actual_balance:.4f})"
            )
            return
        if abs(sell_size - pos.size) > 0.001:
            logger.info(
                f"📐 Sell size from chain for {pos.market_id}: {sell_size:.4f} (state was {pos.size:.4f})"
            )

        order = self.executor.place_order(
            token_id=pos.token_id,
            side=OrderSide.SELL,
            price=sell_price,
            size=sell_size,
            condition_id=pos.condition_id,
        )

        # Poll for fill: limit orders may not fill instantly
        for attempt in range(6):
            order = self.executor.refresh_order_status(order)
            if order.status not in (OrderStatus.OPEN, OrderStatus.PENDING):
                break
            if attempt < 5:
                logger.info(
                    f"⏳ Waiting for exit fill {pos.market_id}: "
                    f"attempt {attempt + 1}/6, status={order.status.value}"
                )
                await asyncio.sleep(5)

        if order.status == OrderStatus.FAILED:
            pos.exit_retries += 1
            new_price = round(sell_price - price_step, 4)
            pos.exit_price_override = max(new_price, 0.01)
            self.positions._save_state()

            if pos.exit_retries >= max_retries:
                logger.warning(
                    f"🚨 Exit gave up after {max_retries} retries for {pos.market_id} [{reason}] — "
                    f"marking force_close_failed; tokens may remain in wallet"
                )
                pos.force_close_failed = True
                self.positions._save_state()
                self._add_cooldown(pos.market_id)  # prevent immediate re-entry
                return  # keep position open for retry/manual intervention
            else:
                logger.warning(
                    f"⚠️  Exit retry {pos.exit_retries}/{max_retries} for {pos.market_id} [{reason}] — "
                    f"next attempt @ ${pos.exit_price_override:.4f}"
                )
                return  # keep position open, retry next cycle
        elif order.status in (OrderStatus.OPEN, OrderStatus.PENDING):
            # Order sat on the book for 30s without filling — treat as a soft failure.
            # Increment retry counter and lower price for next attempt so we eventually
            # cross the spread and hit the bid.
            pos.exit_retries += 1

            if pos.exit_retries >= max_retries:
                logger.warning(
                    f"🚨 Exit order unfilled after {max_retries} cycles for {pos.market_id} [{reason}] — "
                    f"marking force_close_failed"
                )
                pos.force_close_failed = True
                pos.exit_price_override = 0.0  # reset so next attempt uses best bid
                self.positions._save_state()
                self._add_cooldown(pos.market_id)
                return

            # After half the retries, escalate: drop the conservative pricing
            # and switch to best bid to force a fill.
            if pos.exit_retries >= max_retries // 2:
                pos.exit_price_override = 0.0  # reset → next cycle will use best bid
                logger.warning(
                    f"⏳ Exit escalated for {pos.market_id} [{reason}] — "
                    f"retry {pos.exit_retries}/{max_retries}, switching to best bid"
                )
            else:
                new_price = round(sell_price - price_step, 4)
                pos.exit_price_override = max(new_price, 0.01)
                logger.warning(
                    f"⏳ Exit order unfilled for {pos.market_id} [{reason}] — "
                    f"retry {pos.exit_retries}/{max_retries}, "
                    f"next attempt @ ${pos.exit_price_override:.4f}"
                )
            self.positions._save_state()
            return
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            filled = order.filled_size if order.filled_size > 0 else 0
            avg_price = order.filled_avg_price if order.filled_avg_price > 0 else sell_price
            if filled > 0 and filled < pos.size:
                # Record partial PnL
                partial_pnl = (avg_price - pos.entry_price) * filled
                self.risk.record_pnl(partial_pnl)
                # Reduce position size to remaining unfilled portion
                remaining = pos.size - filled
                scale = remaining / pos.size if pos.size > 0 else 1.0
                pos.size = round(remaining, 6)
                pos.size_usdc = round(pos.size_usdc * scale, 4)
                pos.exit_retries = 0
                pos.exit_price_override = 0.0
                self._set_portfolio_exposure(pos.market_id, pos.size_usdc)
                self.positions._save_state()
                logger.warning(
                    f"⚠️  Exit partially filled for {pos.market_id} [{reason}] — "
                    f"{filled:.2f} sold @ ${avg_price:.3f}, "
                    f"remaining {pos.size:.2f} shares (${pos.size_usdc:.2f} USDC)"
                )
            else:
                logger.warning(
                    f"⚠️  Exit partially filled for {pos.market_id} [{reason}] — "
                    f"{order.filled_size:.2f}/{sell_size:.2f} shares, no size update"
                )
            return

        effective_price = order.filled_avg_price if order.filled_avg_price > 0 else sell_price
        actual_exit_size = order.filled_size if order.filled_size > 0 else pos.size
        closed_size_usdc = pos.size_usdc
        pnl = (effective_price - pos.entry_price) * actual_exit_size
        pnl_pct = (effective_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0
        self.risk.record_pnl(pnl)
        self.positions.close_position(pos.market_id)
        self.risk.close_position(pos.market_id)
        self.kelly.close_position(pos.market_id)
        # Start general exit cooldown — prevents re-entry for exit_cooldown_hours
        self._add_exit_cooldown(pos.market_id)
        # Sync CLOB balance after exit so freed USDC.e is visible for next entry
        self.executor._sync_clob_balance()

        logger.info(
            f"🚪 Closed {pos.market_id} {pos.side} [{reason}] — "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1%}) | "
            f"entry=${pos.entry_price:.3f} exit=${effective_price:.3f}"
        )

        # Record to performance tracker
        trade = ClosedTrade(
            market_id=pos.market_id,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=effective_price,
            size_usdc=closed_size_usdc,
            realized_pnl=round(pnl, 4),
            realized_pnl_pct=round(pnl_pct, 4),
            exit_reason=reason,
        )
        self.perf.record_close(trade)

        # Adaptive strategy adjustment (every EVAL_EVERY trades)
        if self.perf.should_evaluate():
            current_params = {
                "kelly_fraction":    self._strat_cfg.get("kelly_fraction", 0.25),
                "min_edge":          self._strat_cfg.get("min_edge", 0.05),
                "max_position_usdc": self._strat_cfg.get("max_position_usdc", 5.0),
            }
            new_params = self.perf.suggest_adjustments(current_params)
            if new_params:
                self._strat_cfg.update(new_params)
                self.kelly.kelly_fraction  = new_params["kelly_fraction"]
                self.kelly.max_position_usdc = new_params["max_position_usdc"]
                self.edge_detector.min_edge = new_params["min_edge"]
                logger.info("✅ Strategy parameters updated in-memory")

    async def _check_redemptions(self) -> None:
        """Check all open positions for on-chain resolution and auto-redeem."""
        for pos in list(self.positions.open_positions):
            resolution = await asyncio.to_thread(self.redeemer.is_resolved, pos.condition_id)
            if resolution is None:
                continue

            # Determine if our side won
            if pos.side == "YES":
                payout = resolution["payout_yes"]
            else:
                payout = resolution["payout_no"]

            won = payout > 0
            logger.info(
                f"🏁 Market {pos.market_id} RESOLVED on-chain — "
                f"{pos.side} {'WON ✅' if won else 'LOST ❌'} "
                f"(payout: YES={resolution['payout_yes']}, NO={resolution['payout_no']})"
            )

            # Resolve neg-risk metadata off the event loop; redemption RPCs are also blocking.
            neg_info = await asyncio.to_thread(
                self.redeemer._lookup_neg_risk_info, pos.condition_id, pos.token_id
            )
            neg_risk = bool(neg_info and neg_info.get("neg_risk"))

            redeemed = await asyncio.to_thread(
                self.redeemer.redeem,
                pos.condition_id,
                neg_risk,
                pos.token_id,
            )

            # Bug fix: if we WON but redeem returned 0 (tx failure / already redeemed),
            # do NOT record -100% loss — instead treat as pending redemption and skip close.
            if won and redeemed == 0:
                logger.warning(
                    f"⚠️  {pos.market_id} WON but redeem returned $0 — "
                    f"skipping close, will retry next cycle"
                )
                continue

            # Close position in tracker
            closed_size_usdc = pos.size_usdc
            pnl = redeemed - closed_size_usdc if won else -closed_size_usdc
            self.risk.record_pnl(pnl)
            self.positions.close_position(pos.market_id)
            self.risk.close_position(pos.market_id)
            self.kelly.close_position(pos.market_id)
            self._add_exit_cooldown(pos.market_id)

            from src.strategy.performance_tracker import ClosedTrade
            exit_price = 1.0 if won else 0.0
            trade = ClosedTrade(
                market_id=pos.market_id,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size_usdc=closed_size_usdc,
                realized_pnl=round(pnl, 4),
                realized_pnl_pct=round(pnl / closed_size_usdc, 4) if closed_size_usdc else 0,
                exit_reason="resolution",
            )
            self.perf.record_close(trade)

            logger.info(
                f"🏁 Redeemed {pos.market_id} {pos.side} — "
                f"PnL: ${pnl:+.2f} | redeemed: ${redeemed:.4f} USDC.e"
            )

    def _print_entry(self, opp, entry_price, shares, size_usdc, order) -> None:
        table = Table(title="📈 Entry", show_header=False, box=None)
        table.add_row("Market", opp.market_question[:55])
        table.add_row("Side", f"BUY {opp.side}")
        table.add_row("Edge", f"{opp.edge:+.4f} ({opp.edge_pct})")
        table.add_row("p̂ / market", f"{opp.p_hat:.4f} / {opp.market_price:.4f}")
        table.add_row("Entry price", f"${entry_price:.3f}")
        table.add_row("Shares", f"{shares:.2f}")
        table.add_row("Size", f"${size_usdc:.2f} USDC")
        table.add_row("Mode", "🔵 DRY RUN" if self.dry_run else "🟢 LIVE")
        console.print(table)

    async def run(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

        self._reconcile_startup_state()

        console.print("\n[bold green]🚀 Polymarket Trader[/bold green]")
        console.print(f"  Mode:      {'🔵 MOCK' if self.client.mock else '🟢 LIVE DATA'}")
        console.print(f"  Dry run:   {self.dry_run} {'(no real orders)' if self.dry_run else '⚡ REAL ORDERS'}")
        console.print(f"  Kelly:     {self.kelly.kelly_fraction}x")
        console.print(f"  Min edge:  {self.edge_detector.min_edge}")
        console.print(f"  Interval:  {self.loop_interval}s")
        console.print(f"  Exit:      profit +{self.positions.profit_target_pct:.0%} / "
                      f"stop -{self.positions.stop_loss_pct:.0%} / "
                      f"{self.positions.pre_resolution_hours:.0f}h before resolution\n")

        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
            if self._running:
                await asyncio.sleep(self.loop_interval)

        self.executor.cancel_all()
        await self.client.close()
        console.print("[bold red]🛑 Stopped[/bold red]")

    def _shutdown(self):
        logger.info("Shutdown signal received")
        self._running = False


def main():
    config = load_config()
    setup_logging(config)
    trader = Trader(config)
    asyncio.run(trader.run())


if __name__ == "__main__":
    main()

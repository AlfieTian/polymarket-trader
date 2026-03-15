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
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "logs/trader.log")
    log_path = Path(__file__).parent.parent / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            logging.FileHandler(log_path),
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

    def _reconcile_startup_state(self) -> None:
        """Clean up crash leftovers before trading resumes.

        Reconciliation steps:
        1. Cancel remote live orders left behind by a crashed process.
        2. Sync CLOB collateral balances.
        3. Reconcile persisted positions against on-chain token balances.
        4. Rebuild in-memory risk/Kelly exposure from the reconciled state.
        """
        logger.info("🩺 Startup reconciliation begin")

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

        # ─── 1. Fetch markets ─────────────────────────────────
        markets = await self.client.get_markets(limit=50, min_volume=1000)

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
        for exit_sig in self.positions.check_exits(current_prices_fast, current_p_hats_fast):
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
        for exit_sig in self.positions.check_exits(current_prices, current_p_hats):
            pos = exit_sig.position
            await self._execute_exit(pos, exit_sig.current_price, market_map, exit_sig.reason.value)

        # ─── 4. Find new opportunities ─────────────────────────
        opportunities = self.edge_detector.scan_markets(scan_data)

        if not opportunities:
            logger.info("😴 No edge this cycle")
        else:
            for opp in opportunities:
                # Skip if already have a position in this market
                if self.positions.get_position(opp.market_id):
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
                # Use the average confidence across ALL signals in the belief history
                # (not just recent ones), which guards against the default p̂=0.05 floor.
                belief = self.bayesian.get_belief(opp.market_id)
                if belief and belief.update_history:
                    avg_conf = sum(s.get("confidence", 0) for s in belief.update_history) / len(belief.update_history)
                else:
                    avg_conf = 0.0
                if avg_conf < self._min_llm_confidence:
                    logger.info(
                        f"⏭️  {opp.market_id} skipped — avg LLM confidence {avg_conf:.2f} "
                        f"< min {self._min_llm_confidence:.2f}"
                    )
                    continue
                logger.info(
                    f"✅ {opp.market_id} {opp.side} approved — edge={opp.edge:+.3f} "
                    f"p̂={opp.p_hat:.3f} avg_conf={avg_conf:.2f}"
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

                # Guard: Polymarket minimum order is 5 shares. Estimate minimum cost
                # and skip if we don't have enough remaining portfolio capacity to cover it —
                # this prevents the CLOBExecutor min-order bump from firing a guaranteed
                # "not enough balance" error against the API.
                # NOTE: opp.market_price is already the correct entry-side price:
                #   YES → market_price_yes,  NO → 1 - market_price_yes
                # Do NOT invert again for NO — that would re-introduce a double-conversion bug.
                MIN_POLYMARKET_SHARES = 5
                entry_price_est = opp.market_price  # EdgeDetector already normalised to entry side
                est_min_order_usdc = MIN_POLYMARKET_SHARES * max(entry_price_est, 0.01)
                remaining = self.kelly.remaining_capacity
                if est_min_order_usdc > remaining:
                    logger.info(
                        f"⏭️  Skipping {opp.market_id} {opp.side}: "
                        f"min order ~${est_min_order_usdc:.2f} > remaining capacity "
                        f"${remaining:.2f}"
                    )
                    continue

                # Risk check
                risk_check = self.risk.validate_trade(opp.market_id, pos_size.position_usdc)
                if not risk_check.approved:
                    logger.warning(f"❌ Risk rejected {opp.market_id}: {risk_check.reason}")
                    continue

                # Execute entry
                await self._execute_entry(opp, pos_size.position_usdc, market_map)

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
            self._check_redemptions()
            # After possible redemption, resync CLOB balance so freed
            # USDC.e is visible for the next entry attempt.
            self.executor._sync_clob_balance()

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

        shares = size_usdc / entry_price if entry_price > 0 else 0
        if shares <= 0:
            return

        # Pre-flight balance/allowance check (avoid API rejection)
        if not self.executor.has_sufficient_balance(
            side=OrderSide.BUY, price=entry_price, size=shares, token_id=token_id
        ):
            logger.warning(
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

        order = self.executor.refresh_order_status(order)

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
        else:
            logger.warning(
                f"⏳ Entry not filled for {opp.market_id}: status={order.status.value}"
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
        sell_price = pos.exit_price_override if pos.exit_price_override > 0 else current_price
        # But never go below 1 tick (avoid 0-price order)
        sell_price = max(round(sell_price, 4), 0.01)

        if pos.force_close_failed:
            logger.warning(
                f"🚨 force_close_failed still active for {pos.market_id} — "
                f"retrying exit (manual intervention may be required)"
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

        order = self.executor.refresh_order_status(order)

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
            logger.warning(
                f"⏳ Exit order still open for {pos.market_id} [{reason}] — "
                f"status={order.status.value}, no position change"
            )
            return
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            logger.warning(
                f"⚠️  Exit partially filled for {pos.market_id} [{reason}] — "
                f"{order.filled_size:.2f}/{pos.size:.2f} shares. "
                f"Position retained for manual reconciliation."
            )
            return

        effective_price = sell_price
        pnl = (effective_price - pos.entry_price) * pos.size
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
            f"entry=${pos.entry_price:.3f} exit=${current_price:.3f}"
        )

        # Record to performance tracker
        trade = ClosedTrade(
            market_id=pos.market_id,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=current_price,
            size_usdc=pos.size_usdc,
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

    def _check_redemptions(self) -> None:
        """Check all open positions for on-chain resolution and auto-redeem."""
        for pos in list(self.positions.open_positions):
            resolution = self.redeemer.is_resolved(pos.condition_id)
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

            # Determine if this is a neg_risk market via redeemer's gamma API lookup
            neg_info = self.redeemer._lookup_neg_risk_info(pos.condition_id)
            neg_risk = bool(neg_info and neg_info.get("neg_risk"))

            redeemed = self.redeemer.redeem(pos.condition_id, neg_risk=neg_risk)

            # Close position in tracker
            pnl = redeemed - pos.size_usdc if won else -pos.size_usdc
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
                size_usdc=pos.size_usdc,
                realized_pnl=round(pnl, 4),
                realized_pnl_pct=round(pnl / pos.size_usdc, 4) if pos.size_usdc else 0,
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

#!/usr/bin/env python3
"""
Polymarket Trader — Main Entry Point

Async loop:
  Fetch markets → Bayesian update (LLM news signals) → Detect edge →
  Kelly sizing → Risk check → Execute BUY → Track positions → Execute SELL on exit
"""

import asyncio
import logging
import os
import signal
import sys
import time
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
from src.execution.clob_executor import CLOBExecutor, OrderSide
from src.execution.position_manager import Position, PositionManager
from src.risk.risk_manager import RiskManager
from src.signals.bayesian_engine import BayesianEngine
from src.strategy.edge_detector import EdgeDetector
from src.strategy.kelly_sizer import KellySizer

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
        )

        self.positions = PositionManager(
            profit_target_pct=pos_cfg.get("profit_target_pct", 0.30),
            stop_loss_pct=pos_cfg.get("stop_loss_pct", 0.40),
            pre_resolution_hours=pos_cfg.get("pre_resolution_hours", 2.0),
            min_edge_to_hold=pos_cfg.get("min_edge_to_hold", 0.02),
        )

        self.news_feed = NewsFeed()
        self.loop_interval = config.get("trading", {}).get("loop_interval_s", 60)
        self._running = False

    async def run_cycle(self) -> None:
        t0 = time.perf_counter()

        # ─── 1. Fetch markets ─────────────────────────────────
        markets = await self.client.get_markets(limit=50, min_volume=1000)
        tradeable = [m for m in markets if 0.10 <= m.yes_price <= 0.90]
        logger.info(f"📊 {len(markets)} markets → {len(tradeable)} tradeable (10-90%)")

        # ─── 2. Update beliefs + detect edges ─────────────────
        scan_data = []
        market_map = {m.id: m for m in tradeable}

        for market in tradeable:
            belief = self.bayesian.get_belief(market.id)
            if belief is None:
                self.bayesian.init_belief(market.id, market.yes_price)

            news_signals = await self.news_feed.get_llm_signals(market)
            if news_signals:
                self.bayesian.batch_update(market.id, news_signals)
                logger.info(f"   🧠 {market.id}: {len(news_signals)} LLM signals applied")

            belief = self.bayesian.get_belief(market.id)
            scan_data.append({
                "market_id": market.id,
                "question": market.question,
                "p_hat": belief.p_hat if belief else market.yes_price,
                "market_price_yes": market.yes_price,
                "volume_24h": market.volume_24h,
                "confidence": 1.0,
            })

        # ─── 3. Check existing positions for exits ────────────
        current_prices = {m.id: m.yes_price for m in tradeable}
        current_p_hats = {d["market_id"]: d["p_hat"] for d in scan_data}
        exit_signals = self.positions.check_exits(current_prices, current_p_hats)

        for exit_sig in exit_signals:
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

                if not self.bayesian.is_tradeable(opp.market_id):
                    continue

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

        order = self.executor.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=entry_price,
            size=shares,
            condition_id=market.condition_id,
        )

        if order.status.value not in ("FAILED",):
            # Record position
            position = Position(
                market_id=opp.market_id,
                condition_id=market.condition_id,
                token_id=token_id,
                side=opp.side,
                entry_price=entry_price,
                size=shares,
                size_usdc=size_usdc,
                p_hat_at_entry=opp.p_hat,
                market_price_at_entry=opp.market_price,
                end_date=market.end_date,
            )
            self.positions.add_position(position)
            self.risk.record_position(opp.market_id, size_usdc)
            self.kelly.record_position(opp.market_id, size_usdc)

            self._print_entry(opp, entry_price, shares, size_usdc, order)

    async def _execute_exit(self, pos: Position, current_price: float, market_map: dict, reason: str) -> None:
        """Execute a SELL order to close a position."""
        order = self.executor.place_order(
            token_id=pos.token_id,
            side=OrderSide.SELL,
            price=current_price,
            size=pos.size,
            condition_id=pos.condition_id,
        )

        pnl = (current_price - pos.entry_price) * pos.size
        self.risk.record_pnl(pnl)
        self.positions.close_position(pos.market_id)
        self.risk.close_position(pos.market_id)
        self.kelly.close_position(pos.market_id)

        logger.info(
            f"🚪 Closed {pos.market_id} {pos.side} [{reason}] — "
            f"PnL: ${pnl:+.2f} | "
            f"entry=${pos.entry_price:.3f} exit=${current_price:.3f}"
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

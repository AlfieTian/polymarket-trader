#!/usr/bin/env python3
"""
Polymarket Trader — Main Entry Point

使用方法：
  1. cp .env.template .env && 填入 POLYMARKET_PRIVATE_KEY
  2. python scripts/setup_api_key.py  (自动生成 API 凭证)
  3. python scripts/run_trader.py     (默认 dry-run 安全模式)

Async event loop:
  Fetch markets → Bayesian update → Detect edge → Kelly sizing → Risk check → Execute
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.news_feed import NewsFeed
from src.data.polymarket_client import PolymarketClient
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
    """Main trading orchestrator."""

    def __init__(self, config: dict):
        self.config = config

        pm_cfg = config.get("polymarket", {})
        strat_cfg = config.get("strategy", {})
        bay_cfg = config.get("bayesian", {})
        risk_cfg = config.get("risk", {})

        self.client = PolymarketClient(
            dry_run=pm_cfg.get("dry_run", True),
        )

        self.bayesian = BayesianEngine(
            prior_weight=bay_cfg.get("prior_weight", 0.7),
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

        self.news_feed = NewsFeed()
        self.loop_interval = config.get("trading", {}).get("loop_interval_s", 60)
        self._running = False

    async def run_cycle(self) -> None:
        """Execute one trading cycle."""
        t0 = time.perf_counter()

        markets = await self.client.get_markets(limit=50, min_volume=1000)
        logger.info(f"📊 Fetched {len(markets)} markets")

        # Filter to tradeable price range (10%-90%)
        tradeable = [m for m in markets if 0.10 <= m.yes_price <= 0.90]
        logger.info(f"   Tradeable (10%-90% price range): {len(tradeable)}/{len(markets)}")

        scan_data = []
        for market in tradeable:
            belief = self.bayesian.get_belief(market.id)
            if belief is None:
                self.bayesian.init_belief(market.id, market.yes_price)

            # Fetch and apply LLM news signals for this market
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

        opportunities = self.edge_detector.scan_markets(scan_data)

        if not opportunities:
            logger.info("😴 No edge this cycle")
            return

        for opp in opportunities:
            if not self.bayesian.is_tradeable(opp.market_id):
                continue

            bankroll = self.kelly.remaining_capacity
            pos = self.kelly.calculate(
                market_id=opp.market_id,
                side=opp.side,
                p_hat=opp.p_hat,
                market_price=opp.market_price,
                bankroll=bankroll,
            )

            if pos.position_usdc <= 0:
                continue

            risk_check = self.risk.validate_trade(opp.market_id, pos.position_usdc)
            if not risk_check.approved:
                logger.warning(f"❌ Risk rejected: {risk_check.reason}")
                continue

            # Find token
            market = next(m for m in markets if m.id == opp.market_id)
            token_id = market.yes_token_id if opp.side == "YES" else market.no_token_id
            shares = pos.position_usdc / opp.market_price if opp.market_price > 0 else 0

            # Execute via SDK
            result = self.client.place_order(
                token_id=token_id,
                side="BUY",
                price=round(opp.market_price, 2),
                size=round(shares, 1),
            )

            self.risk.record_position(opp.market_id, pos.position_usdc)
            self.kelly.record_position(opp.market_id, pos.position_usdc)
            self._print_trade(opp, pos, result)

        elapsed = time.perf_counter() - t0
        logger.info(f"⏱️ Cycle in {elapsed:.2f}s")

    def _print_trade(self, opp, pos, result) -> None:
        table = Table(title="📈 Trade", show_header=False)
        table.add_row("Market", opp.market_question[:60])
        table.add_row("Side", opp.side)
        table.add_row("Edge", f"{opp.edge:+.4f} ({opp.edge_pct})")
        table.add_row("p̂", f"{opp.p_hat:.4f}")
        table.add_row("Market", f"{opp.market_price:.4f}")
        table.add_row("Kelly", f"{pos.kelly_full:.4f} × {pos.kelly_fraction_used}")
        table.add_row("Size", f"${pos.position_usdc:.2f}")
        table.add_row("Order", str(result.get("orderID", result.get("order_id", "?")))[:16])
        table.add_row("Mode", "🔵 DRY RUN" if self.client.dry_run else "🟢 LIVE")
        console.print(table)

    async def run(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

        console.print("\n[bold green]🚀 Polymarket Trader[/bold green]")
        console.print(f"  Mode: {'🔵 MOCK' if self.client.mock else '🟢 LIVE'}")
        console.print(f"  Dry run: {self.client.dry_run}")
        console.print(f"  Kelly: {self.kelly.kelly_fraction}x")
        console.print(f"  Min edge: {self.edge_detector.min_edge}")
        console.print(f"  Interval: {self.loop_interval}s\n")

        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
            if self._running:
                await asyncio.sleep(self.loop_interval)

        self.client.cancel_all_orders()
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

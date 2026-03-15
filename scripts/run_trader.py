#!/usr/bin/env python3
"""
Polymarket Trader — Main Entry Point

Async event loop:
1. Fetch active markets
2. Update Bayesian beliefs with signals
3. Detect edge opportunities
4. Size positions (fractional Kelly)
5. Run risk checks
6. Execute approved trades
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.news_feed import NewsFeed, NewsItem, Sentiment
from src.data.polymarket_client import PolymarketClient
from src.execution.clob_executor import CLOBExecutor, OrderSide
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.signals.bayesian_engine import BayesianEngine
from src.strategy.edge_detector import EdgeDetector
from src.strategy.kelly_sizer import KellySizer

console = Console()
logger = logging.getLogger("trader")


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / path
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """Configure structured logging with Rich."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))

    # File handler
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
        self._running = False

        # Initialize components
        pm_config = config.get("polymarket", {})
        self.client = PolymarketClient(
            api_key=pm_config.get("api_key", ""),
            rest_url=pm_config.get("rest_url", "https://gamma-api.polymarket.com"),
            clob_url=pm_config.get("clob_url", "https://clob.polymarket.com"),
            mock=pm_config.get("dry_run", True),
        )

        bay_config = config.get("bayesian", {})
        self.bayesian = BayesianEngine(
            prior_weight=bay_config.get("prior_weight", 0.7),
            min_observations=bay_config.get("min_observations", 3),
        )

        strat_config = config.get("strategy", {})
        self.edge_detector = EdgeDetector(
            min_edge=strat_config.get("min_edge", 0.03),
        )

        self.kelly = KellySizer(
            kelly_fraction=strat_config.get("kelly_fraction", 0.25),
            max_position_usdc=strat_config.get("max_position_usdc", 100),
            max_portfolio_usdc=strat_config.get("max_portfolio_usdc", 1000),
        )

        risk_config = config.get("risk", {})
        self.risk = RiskManager(
            max_position_usdc=strat_config.get("max_position_usdc", 100),
            max_portfolio_usdc=strat_config.get("max_portfolio_usdc", 1000),
            max_daily_loss_usdc=risk_config.get("max_daily_loss_usdc", 200),
            max_market_concentration=risk_config.get("max_market_concentration", 0.3),
        )

        self.executor = CLOBExecutor(
            clob_url=pm_config.get("clob_url", "https://clob.polymarket.com"),
            api_key=pm_config.get("api_key", ""),
            dry_run=pm_config.get("dry_run", True),
        )

        self.order_manager = OrderManager(self.executor)
        self.news_feed = NewsFeed()

        self.loop_interval = config.get("trading", {}).get("loop_interval_s", 60)

    async def run_cycle(self) -> None:
        """Execute one trading cycle."""
        cycle_start = time.perf_counter()

        # 1. Fetch markets
        markets = await self.client.get_markets(limit=20)
        logger.info(f"📊 Fetched {len(markets)} markets")

        # 2. For each market: update beliefs, detect edges
        scan_data = []
        for market in markets:
            # Initialize belief if not exists
            belief = self.bayesian.get_belief(market.id)
            if belief is None:
                self.bayesian.init_belief(market.id, market.yes_price)

            # Apply any pending news signals
            signals = self.news_feed.get_signals()
            if signals:
                self.bayesian.batch_update(market.id, signals)

            belief = self.bayesian.get_belief(market.id)

            scan_data.append({
                "market_id": market.id,
                "question": market.question,
                "p_hat": belief.p_hat if belief else market.yes_price,
                "market_price_yes": market.yes_price,
                "volume_24h": market.volume_24h,
                "confidence": 1.0,
            })

        # 3. Detect edges
        opportunities = self.edge_detector.scan_markets(scan_data)

        if not opportunities:
            logger.info("No edge opportunities found this cycle")
            return

        # 4. Process each opportunity
        for opp in opportunities:
            # Check if enough observations
            if not self.bayesian.is_tradeable(opp.market_id):
                logger.debug(f"Skipping {opp.market_id}: insufficient observations")
                continue

            # 5. Size position
            bankroll = self.kelly.remaining_capacity
            position = self.kelly.calculate(
                market_id=opp.market_id,
                side=opp.side,
                p_hat=opp.p_hat,
                market_price=opp.market_price,
                bankroll=bankroll,
            )

            if position.position_usdc <= 0:
                continue

            # 6. Risk check
            risk_check = self.risk.validate_trade(
                market_id=opp.market_id,
                size_usdc=position.position_usdc,
            )

            if not risk_check.approved:
                logger.warning(f"❌ Risk rejected: {risk_check.reason}")
                continue

            # 7. Execute
            # Find the right token
            market = next(m for m in markets if m.id == opp.market_id)
            token_id = market.yes_token_id if opp.side == "YES" else market.no_token_id
            shares = position.position_usdc / opp.market_price if opp.market_price > 0 else 0

            order = await self.executor.place_limit_order(
                token_id=token_id,
                side=OrderSide.BUY,
                price=opp.market_price,
                size=shares,
            )

            # Record position
            self.risk.record_position(opp.market_id, position.position_usdc)
            self.kelly.record_position(opp.market_id, position.position_usdc)

            self._print_trade(opp, position, order)

        # Sweep stale orders
        cancelled = await self.order_manager.sweep_stale()
        if cancelled:
            logger.info(f"🧹 Swept {len(cancelled)} stale orders")

        elapsed = time.perf_counter() - cycle_start
        logger.info(f"⏱️ Cycle completed in {elapsed:.2f}s")

    def _print_trade(self, opp, position, order) -> None:
        """Pretty-print a trade execution."""
        table = Table(title="📈 Trade Executed", show_header=False)
        table.add_row("Market", opp.market_question[:60])
        table.add_row("Side", opp.side)
        table.add_row("Edge", f"{opp.edge:+.4f} ({opp.edge_pct})")
        table.add_row("p̂ (ours)", f"{opp.p_hat:.4f}")
        table.add_row("p (market)", f"{opp.market_price:.4f}")
        table.add_row("Kelly f*", f"{position.kelly_full:.4f}")
        table.add_row("Position", f"${position.position_usdc:.2f} USDC")
        table.add_row("Order ID", order.order_id)
        table.add_row("Status", f"{'🔵 DRY RUN' if order.dry_run else '🟢 LIVE'}")
        console.print(table)

    async def run(self) -> None:
        """Main async loop with graceful shutdown."""
        self._running = True

        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

        console.print("[bold green]🚀 Polymarket Trader starting...[/bold green]")
        console.print(f"   Loop interval: {self.loop_interval}s")
        console.print(f"   Dry run: {self.executor.dry_run}")
        console.print(f"   Kelly fraction: {self.kelly.kelly_fraction}x")
        console.print(f"   Min edge: {self.edge_detector.min_edge}")
        console.print()

        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)

            if self._running:
                await asyncio.sleep(self.loop_interval)

        # Cleanup
        await self.executor.cancel_all()
        await self.client.close()
        await self.executor.close()
        console.print("[bold red]🛑 Trader stopped[/bold red]")

    def _shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False


def main():
    config = load_config()
    setup_logging(config)

    trader = Trader(config)
    asyncio.run(trader.run())


if __name__ == "__main__":
    main()

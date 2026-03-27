"""
Polymarket API Client

Uses py-clob-client SDK for real trading + REST for market discovery.
Automatically switches between mock and live mode based on credentials.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)

# ── Sports / Esports market ban ───────────────────────────────────────────────

def is_banned_market(question: str, sports_market_type: str = "") -> bool:
    """Return True if the market should be excluded (sports/esports).

    Uses the sportsMarketType field from Gamma API — non-empty means sports/esports.
    """
    return bool(sports_market_type)
# ─────────────────────────────────────────────────────────────────────────────


class MarketStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


@dataclass
class Market:
    """Polymarket market data model."""

    id: str
    condition_id: str
    question: str
    outcomes: list[str]
    outcome_prices: list[float]
    tokens: list[dict]  # [{token_id, outcome}]
    volume_24h: float
    liquidity: float
    end_date: str
    status: MarketStatus
    category: str = ""
    description: str = ""
    sports_market_type: str = ""  # non-empty = sports/esports market (moneyline/spreads/etc)

    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.5

    @property
    def no_price(self) -> float:
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else 1 - self.yes_price

    @property
    def yes_token_id(self) -> str:
        return self.tokens[0]["token_id"] if self.tokens else ""

    @property
    def no_token_id(self) -> str:
        return self.tokens[1]["token_id"] if len(self.tokens) > 1 else ""


@dataclass
class OrderBookEntry:
    price: float
    size: float


@dataclass
class OrderBook:
    token_id: str
    bids: list[OrderBookEntry] = field(default_factory=list)
    asks: list[OrderBookEntry] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


class PolymarketClient:
    """Polymarket client with auto mock/live switching.

    If .env has valid credentials → live mode
    Otherwise → mock mode with synthetic data
    """

    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(
        self,
        private_key: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        dry_run: bool = True,
        rate_limit_rps: float = 5.0,
    ):
        # Try loading from env if not provided
        self.private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY", "")
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY", "")
        self.api_secret = api_secret or os.getenv("POLYMARKET_API_SECRET", "")
        self.api_passphrase = api_passphrase or os.getenv("POLYMARKET_API_PASSPHRASE", "")
        self.dry_run = dry_run

        # Determine mode
        self.has_credentials = bool(self.private_key and self.api_key)
        self.mock = not self.has_credentials

        if self.mock:
            logger.warning("⚠️ No credentials found — running in MOCK mode")
            logger.warning("   Fill .env with your keys to connect to Polymarket")
        else:
            logger.info("🔑 Credentials loaded — LIVE data mode" +
                        (" (dry_run=true, no real orders)" if dry_run else " ⚡ LIVE TRADING"))

        self._rate_limit_interval = 1.0 / rate_limit_rps
        self._last_request_time = 0.0
        self._http: httpx.AsyncClient | None = None
        self._clob_client = None

    def _init_clob_client(self):
        """Lazy-init the py-clob-client SDK."""
        if self._clob_client is None and self.has_credentials:
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase,
                )
                self._clob_client = ClobClient(
                    host=self.CLOB_API,
                    key=self.private_key,
                    chain_id=137,
                    creds=creds,
                )
                logger.info("✅ CLOB client initialized")
            except Exception as e:
                logger.error(f"Failed to init CLOB client: {e}")
                self._clob_client = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._http

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_interval:
            await asyncio.sleep(self._rate_limit_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _get_json(self, url: str, params: dict | None = None) -> dict | list:
        await self._rate_limit()
        client = await self._get_http()
        for attempt in range(3):
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    # ─── Market Data (Gamma API — no auth needed) ─────────────

    async def get_markets(
        self,
        limit: int = 50,
        active_only: bool = True,
        min_volume: float = 0,
    ) -> list[Market]:
        """Fetch prediction markets from Gamma API (public, no auth).

        Works in both mock and live mode — Gamma API is always public.
        """
        if self.mock:
            return _MOCK_MARKETS[:limit]

        try:
            # NOTE: Gamma API's "active" parameter is BROKEN — it returns
            # closed/resolved markets regardless.  Use "closed=false" instead,
            # which actually filters correctly.
            params = {
                "limit": limit,
                "order": "volume24hr",
                "ascending": False,
            }
            if active_only:
                params["closed"] = False  # closed=false = only open markets
            data = await self._get_json(
                f"{self.GAMMA_API}/markets",
                params=params,
            )
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

        markets = []
        banned_count = 0
        resolved_count = 0
        for item in data if isinstance(data, list) else []:
            try:
                market = self._parse_market(item)
                if not market or market.volume_24h < min_volume:
                    continue
                # FILTER: Skip resolved or closed/expired markets
                if market.status in (MarketStatus.RESOLVED, MarketStatus.CLOSED):
                    logger.debug(
                        f"🏁 Skipped {'resolved' if market.status == MarketStatus.RESOLVED else 'closed/expired'} "
                        f"market: {market.question[:60]}"
                    )
                    resolved_count += 1
                    continue
                if is_banned_market(market.question, market.sports_market_type):
                    logger.debug(
                        f"🚫 Banned market (sports/esports): {market.question[:60]}"
                        + (f" [sportsMarketType={market.sports_market_type}]" if market.sports_market_type else "")
                    )
                    banned_count += 1
                    continue
                markets.append(market)
            except Exception as e:
                logger.debug(f"Skipping unparseable market: {e}")

        logger.info(
            f"📊 Fetched {len(markets)} markets from Polymarket "
            f"(skipped {banned_count} sports/esports, {resolved_count} resolved)"
        )
        return markets

    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a single market."""
        if self.mock:
            return next((m for m in _MOCK_MARKETS if m.id == market_id), None)

        try:
            data = await self._get_json(f"{self.GAMMA_API}/markets/{market_id}")
            return self._parse_market(data)
        except Exception as e:
            logger.error(f"Failed to fetch market {market_id}: {e}")
            return None

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch CLOB order book for a token."""
        if self.mock:
            return _mock_orderbook(token_id)

        try:
            data = await self._get_json(
                f"{self.CLOB_API}/book",
                params={"token_id": token_id},
            )
            bids = [
                OrderBookEntry(price=float(b["price"]), size=float(b["size"]))
                for b in data.get("bids", [])
            ]
            asks = [
                OrderBookEntry(price=float(a["price"]), size=float(a["size"]))
                for a in data.get("asks", [])
            ]
            return OrderBook(token_id=token_id, bids=bids, asks=asks)
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {token_id}: {e}")
            return OrderBook(token_id=token_id)

    # ─── Trading (CLOB SDK — needs auth) ──────────────────────

    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict:
        """Place a limit order via CLOB SDK (synchronous).

        Args:
            token_id: Token to trade
            side: "BUY" or "SELL"
            price: Limit price (0-1, increments of 0.01)
            size: Number of shares

        Returns:
            Order result dict
        """
        if self.dry_run:
            result = {
                "orderID": f"dry-{int(time.time())}",
                "status": "DRY_RUN",
                "token_id": token_id,
                "side": side,
                "price": price,
                "size": size,
            }
            logger.info(f"[DRY RUN] {side} {size:.1f} @ ${price:.2f} — {token_id[:16]}...")
            return result

        self._init_clob_client()
        if not self._clob_client:
            raise RuntimeError("CLOB client not available — check credentials")

        from py_clob_client.order_builder.constants import BUY, SELL

        order_side = BUY if side == "BUY" else SELL

        # Build and sign order
        order_args = {
            "token_id": token_id,
            "price": price,
            "size": size,
            "side": order_side,
        }

        try:
            signed_order = self._clob_client.create_and_post_order(order_args)
            logger.info(f"✅ Order placed: {side} {size:.1f} @ ${price:.2f}")
            return signed_order
        except Exception as e:
            logger.error(f"❌ Order failed: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order via CLOB SDK."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel order {order_id}")
            return True

        self._init_clob_client()
        if not self._clob_client:
            return False

        try:
            self._clob_client.cancel(order_id)
            logger.info(f"✅ Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Cancel failed: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if self.dry_run:
            logger.info("[DRY RUN] Cancel all orders")
            return True

        self._init_clob_client()
        if not self._clob_client:
            return False

        try:
            self._clob_client.cancel_all()
            logger.info("✅ All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"❌ Cancel all failed: {e}")
            return False

    def get_open_orders(self) -> list[dict]:
        """Get open orders from CLOB."""
        if self.mock or self.dry_run:
            return []

        self._init_clob_client()
        if not self._clob_client:
            return []

        try:
            return self._clob_client.get_orders()
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    # ─── Wallet Positions (Data API) ────────────────────────────

    DATA_API = "https://data-api.polymarket.com"

    async def get_wallet_positions(self, wallet_address: str) -> list[dict]:
        """Fetch all positions for a wallet from Polymarket data API.

        Returns list of dicts with keys like:
            asset, conditionId, size, avgPrice, market (slug), side, etc.
        """
        if self.mock:
            return []

        try:
            data = await self._get_json(
                f"{self.DATA_API}/positions",
                params={"user": wallet_address.lower()},
            )
            if not isinstance(data, list):
                return []
            # Filter to positions with non-zero size
            return [p for p in data if float(p.get("size", 0)) > 0]
        except Exception as e:
            logger.warning(f"Failed to fetch wallet positions from data API: {e}")
            return []

    async def get_market_by_condition(self, condition_id: str) -> Market | None:
        """Fetch a market by its condition ID via Gamma API.

        NOTE: The Gamma API's `conditionId` query parameter is BROKEN —
        it ignores the value and returns all markets sorted by ID.
        We must fetch by market ID or token ID instead.
        This method is kept for backward compatibility but will likely
        return None.  Prefer get_market() or get_market_by_token().
        """
        logger.warning(
            f"get_market_by_condition({condition_id[:20]}...) called — "
            f"Gamma API conditionId filter is broken, result may be wrong"
        )
        return None

    async def get_market_by_token(self, token_id: str) -> Market | None:
        """Fetch a market by its CLOB token ID via Gamma API."""
        if self.mock:
            return None
        try:
            data = await self._get_json(
                f"{self.GAMMA_API}/markets",
                params={"clob_token_ids": token_id},
            )
            items = data if isinstance(data, list) else []
            if items:
                return self._parse_market(items[0])
        except Exception as e:
            logger.warning(f"Failed to fetch market by token {token_id[:16]}...: {e}")
        return None

    # ─── Helpers ──────────────────────────────────────────────

    def _parse_market(self, item: dict) -> Market | None:
        """Parse a Gamma API market response into Market dataclass."""
        try:
            # Parse JSON string fields
            outcomes_raw = item.get("outcomes", "[]")
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw

            prices_raw = item.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            prices = [float(p) for p in prices]

            tokens_raw = item.get("clobTokenIds", "[]")
            token_ids = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw

            token_list = [
                {"token_id": tid, "outcome": out}
                for tid, out in zip(token_ids, outcomes)
            ]

            # Determine market status from API fields
            is_resolved = bool(item.get("isResolved") or item.get("resolved"))
            is_closed = bool(item.get("closed") or item.get("active") == False)
            # Also check endDate: if end_date is in the past by >24h, treat as closed
            end_date_str = item.get("endDate", "")
            if not is_closed and end_date_str:
                try:
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    now_utc = datetime.now(timezone.utc)
                    if end_dt < now_utc:
                        is_closed = True
                except (ValueError, TypeError):
                    pass

            if is_resolved:
                status = MarketStatus.RESOLVED
            elif is_closed:
                status = MarketStatus.CLOSED
            else:
                status = MarketStatus.ACTIVE

            return Market(
                id=item.get("id", ""),
                condition_id=item.get("conditionId", ""),
                question=item.get("question", "Unknown"),
                outcomes=outcomes,
                outcome_prices=prices,
                tokens=token_list,
                volume_24h=float(item.get("volume24hr", 0)),
                liquidity=float(item.get("liquidityClob", 0) or 0),
                end_date=end_date_str,
                status=status,
                category=item.get("category", ""),
                description=item.get("description", ""),
                sports_market_type=item.get("sportsMarketType", "") or "",
            )
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Parse error: {e}")
            return None

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()


# ─── Mock Data ────────────────────────────────────────────────

_MOCK_MARKETS = [
    Market(
        id="mock-btc-100k",
        condition_id="0xmock1",
        question="Will BTC exceed $100,000 by March 31, 2026?",
        outcomes=["Yes", "No"],
        outcome_prices=[0.72, 0.28],
        tokens=[
            {"token_id": "mock-token-yes-1", "outcome": "Yes"},
            {"token_id": "mock-token-no-1", "outcome": "No"},
        ],
        volume_24h=125000.0,
        liquidity=450000.0,
        end_date="2026-03-31T00:00:00Z",
        status=MarketStatus.ACTIVE,
        category="Crypto",
    ),
    Market(
        id="mock-fed-rate-cut",
        condition_id="0xmock2",
        question="Will the Fed cut rates in Q1 2026?",
        outcomes=["Yes", "No"],
        outcome_prices=[0.35, 0.65],
        tokens=[
            {"token_id": "mock-token-yes-2", "outcome": "Yes"},
            {"token_id": "mock-token-no-2", "outcome": "No"},
        ],
        volume_24h=89000.0,
        liquidity=320000.0,
        end_date="2026-03-31T23:59:59Z",
        status=MarketStatus.ACTIVE,
        category="Economics",
    ),
    Market(
        id="mock-oil-100",
        condition_id="0xmock3",
        question="Will oil prices exceed $100/barrel by April 2026?",
        outcomes=["Yes", "No"],
        outcome_prices=[0.58, 0.42],
        tokens=[
            {"token_id": "mock-token-yes-3", "outcome": "Yes"},
            {"token_id": "mock-token-no-3", "outcome": "No"},
        ],
        volume_24h=67000.0,
        liquidity=210000.0,
        end_date="2026-04-30T00:00:00Z",
        status=MarketStatus.ACTIVE,
        category="Commodities",
    ),
]


def _mock_orderbook(token_id: str) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=[
            OrderBookEntry(price=0.50, size=1000),
            OrderBookEntry(price=0.49, size=2000),
            OrderBookEntry(price=0.48, size=3000),
        ],
        asks=[
            OrderBookEntry(price=0.52, size=800),
            OrderBookEntry(price=0.53, size=1500),
            OrderBookEntry(price=0.55, size=2500),
        ],
    )

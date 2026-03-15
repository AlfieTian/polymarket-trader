"""
Polymarket API Client

REST + WebSocket client for market data and order management.
Supports mock mode for testing without API keys.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


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
    tokens: list[dict]  # {token_id, outcome}
    volume_24h: float
    liquidity: float
    end_date: str
    status: MarketStatus
    category: str = ""
    description: str = ""

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


# ─── Mock data for testing ────────────────────────────────────

MOCK_MARKETS = [
    Market(
        id="mock-market-1",
        condition_id="0xabc123",
        question="Will BTC exceed $100,000 by March 31, 2026?",
        outcomes=["Yes", "No"],
        outcome_prices=[0.72, 0.28],
        tokens=[
            {"token_id": "token-yes-1", "outcome": "Yes"},
            {"token_id": "token-no-1", "outcome": "No"},
        ],
        volume_24h=125000.0,
        liquidity=450000.0,
        end_date="2026-03-31T00:00:00Z",
        status=MarketStatus.ACTIVE,
        category="Crypto",
    ),
    Market(
        id="mock-market-2",
        condition_id="0xdef456",
        question="Will the Fed cut rates in Q1 2026?",
        outcomes=["Yes", "No"],
        outcome_prices=[0.35, 0.65],
        tokens=[
            {"token_id": "token-yes-2", "outcome": "Yes"},
            {"token_id": "token-no-2", "outcome": "No"},
        ],
        volume_24h=89000.0,
        liquidity=320000.0,
        end_date="2026-03-31T23:59:59Z",
        status=MarketStatus.ACTIVE,
        category="Economics",
    ),
    Market(
        id="mock-market-3",
        condition_id="0x789ghi",
        question="Will oil prices exceed $100/barrel by April 2026?",
        outcomes=["Yes", "No"],
        outcome_prices=[0.58, 0.42],
        tokens=[
            {"token_id": "token-yes-3", "outcome": "Yes"},
            {"token_id": "token-no-3", "outcome": "No"},
        ],
        volume_24h=67000.0,
        liquidity=210000.0,
        end_date="2026-04-30T00:00:00Z",
        status=MarketStatus.ACTIVE,
        category="Commodities",
    ),
]


class PolymarketClient:
    """Async Polymarket API client with mock mode support."""

    def __init__(
        self,
        api_key: str = "",
        rest_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        mock: bool = True,
        rate_limit_rps: float = 5.0,
    ):
        self.api_key = api_key
        self.rest_url = rest_url
        self.clob_url = clob_url
        self.mock = mock
        self._rate_limit_interval = 1.0 / rate_limit_rps
        self._last_request_time = 0.0
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_interval:
            await asyncio.sleep(self._rate_limit_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _get(self, url: str, params: dict | None = None) -> dict:
        await self._rate_limit()
        client = await self._get_client()
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

    async def get_markets(
        self,
        limit: int = 50,
        active_only: bool = True,
    ) -> list[Market]:
        """Fetch active prediction markets.

        Args:
            limit: Maximum markets to return
            active_only: Only return active (unresolved) markets

        Returns:
            List of Market objects
        """
        if self.mock:
            logger.info(f"[MOCK] Returning {len(MOCK_MARKETS)} mock markets")
            return MOCK_MARKETS[:limit]

        data = await self._get(
            f"{self.rest_url}/markets",
            params={"limit": limit, "active": active_only},
        )

        markets = []
        for item in data:
            try:
                prices = json.loads(item.get("outcomePrices", "[]"))
                prices = [float(p) for p in prices]
                tokens = json.loads(item.get("clobTokenIds", "[]"))
                outcomes = json.loads(item.get("outcomes", "[]"))

                token_list = [
                    {"token_id": tid, "outcome": out}
                    for tid, out in zip(tokens, outcomes)
                ]

                market = Market(
                    id=item["id"],
                    condition_id=item.get("conditionId", ""),
                    question=item.get("question", ""),
                    outcomes=outcomes,
                    outcome_prices=prices,
                    tokens=token_list,
                    volume_24h=float(item.get("volume24hr", 0)),
                    liquidity=float(item.get("liquidityClob", 0)),
                    end_date=item.get("endDate", ""),
                    status=MarketStatus.ACTIVE,
                    category=item.get("category", ""),
                    description=item.get("description", ""),
                )
                markets.append(market)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse market: {e}")
                continue

        logger.info(f"Fetched {len(markets)} markets from Polymarket")
        return markets

    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a single market by ID."""
        if self.mock:
            return next((m for m in MOCK_MARKETS if m.id == market_id), None)

        data = await self._get(f"{self.rest_url}/markets/{market_id}")
        # Parse same as above (simplified)
        return None  # TODO: full parsing

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch order book for a specific token.

        Args:
            token_id: The CLOB token ID

        Returns:
            OrderBook with bids and asks
        """
        if self.mock:
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

        data = await self._get(
            f"{self.clob_url}/book",
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

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

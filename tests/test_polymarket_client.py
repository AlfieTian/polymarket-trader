"""Tests for PolymarketClient query semantics."""

import asyncio
import json

from src.data.polymarket_client import PolymarketClient


def _market_item(outcomes, prices, token_ids, market_id="m1"):
    """Build a Gamma-API-shaped market dict for _parse_market."""
    return {
        "id": market_id,
        "conditionId": "0xcond",
        "question": "Will X happen?",
        "outcomes": json.dumps(outcomes),
        "outcomePrices": json.dumps([str(p) for p in prices]),
        "clobTokenIds": json.dumps(token_ids),
        "volume24hr": 1000,
        "endDate": "",
    }


def test_parse_market_rejects_non_yes_no_outcomes():
    client = PolymarketClient(private_key="pk", api_key="key")
    # Two outcomes, but not Yes/No — index [1] would be a bogus "NO" leg.
    item = _market_item(["Trump", "Biden"], [0.6, 0.4], ["t0", "t1"])
    assert client._parse_market(item) is None
    # Categorical (3+ outcomes) is rejected too.
    item3 = _market_item(["A", "B", "C"], [0.5, 0.3, 0.2], ["t0", "t1", "t2"])
    assert client._parse_market(item3) is None


def test_parse_market_normalises_reversed_outcome_order():
    client = PolymarketClient(private_key="pk", api_key="key")
    # API returns ["No","Yes"] — parser must reorder so index 0 is Yes.
    item = _market_item(["No", "Yes"], [0.3, 0.7], ["no_tok", "yes_tok"])
    market = client._parse_market(item)
    assert market is not None
    assert market.yes_price == 0.7
    assert market.no_price == 0.3
    assert market.yes_token_id == "yes_tok"
    assert market.no_token_id == "no_tok"


def test_parse_market_keeps_correct_outcome_order():
    client = PolymarketClient(private_key="pk", api_key="key")
    item = _market_item(["Yes", "No"], [0.7, 0.3], ["yes_tok", "no_tok"])
    market = client._parse_market(item)
    assert market is not None
    assert market.yes_price == 0.7
    assert market.no_price == 0.3
    assert market.yes_token_id == "yes_tok"
    assert market.no_token_id == "no_tok"


def test_get_markets_active_only_uses_closed_false():
    client = PolymarketClient(private_key="pk", api_key="key")
    seen = {}

    async def fake_get_json(url: str, params: dict | None = None):
        seen["url"] = url
        seen["params"] = params
        return []

    client._get_json = fake_get_json

    asyncio.run(client.get_markets(active_only=True))

    assert seen["params"]["closed"] is False


def test_get_markets_inactive_filter_does_not_force_closed_only():
    client = PolymarketClient(private_key="pk", api_key="key")
    seen = {}

    async def fake_get_json(url: str, params: dict | None = None):
        seen["url"] = url
        seen["params"] = params
        return []

    client._get_json = fake_get_json

    asyncio.run(client.get_markets(active_only=False))

    assert "closed" not in seen["params"]

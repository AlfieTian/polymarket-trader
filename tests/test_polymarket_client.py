"""Tests for PolymarketClient query semantics."""

import asyncio

from src.data.polymarket_client import PolymarketClient


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

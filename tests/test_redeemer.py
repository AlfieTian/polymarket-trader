"""Tests for Redeemer token-aware neg-risk flows."""

import pytest

from src.execution.redeemer import Redeemer


def test_can_redeem_uses_token_id_for_neg_risk_lookup(monkeypatch):
    redeemer = Redeemer.__new__(Redeemer)

    monkeypatch.setattr(
        redeemer,
        "is_resolved",
        lambda condition_id: {"payout_yes": 1, "payout_no": 0},
    )
    monkeypatch.setattr(redeemer, "is_neg_risk_determined", lambda question_id: True)
    monkeypatch.setattr(
        redeemer,
        "_lookup_neg_risk_info",
        lambda condition_id, token_id="": {"neg_risk": True, "question_id": "0x01"} if token_id == "tok-1" else None,
    )

    assert redeemer.can_redeem("cid-1", token_id="tok-1") == (True, True)


def test_check_and_redeem_all_passes_token_id(monkeypatch):
    redeemer = Redeemer.__new__(Redeemer)
    calls = []

    monkeypatch.setattr(
        redeemer,
        "can_redeem",
        lambda condition_id, token_id="": (token_id == "tok-1", True),
    )

    def fake_redeem(condition_id: str, neg_risk: bool = False, token_id: str = "") -> float:
        calls.append((condition_id, neg_risk, token_id))
        return 12.5

    monkeypatch.setattr(redeemer, "redeem", fake_redeem)

    results = redeemer.check_and_redeem_all(
        [{"market_id": "m1", "condition_id": "cid-1", "token_id": "tok-1"}]
    )

    assert calls == [("cid-1", True, "tok-1")]
    assert results == [{
        "market_id": "m1",
        "condition_id": "cid-1",
        "token_id": "tok-1",
        "amount": 12.5,
        "neg_risk": True,
    }]


def test_lookup_neg_risk_info_enriches_clob_hit_with_gamma_question_id(monkeypatch):
    redeemer = Redeemer.__new__(Redeemer)

    def fake_urlopen_json(url: str, timeout: int = 10):
        if "clob.polymarket.com/neg-risk" in url:
            return {"neg_risk": True}
        if "gamma-api.polymarket.com/markets" in url:
            return [{
                "negRisk": True,
                "questionID": "0x01",
                "negRiskMarketID": "nr-1",
                "question": "Example market",
            }]
        return None

    monkeypatch.setattr(Redeemer, "_urlopen_json", staticmethod(fake_urlopen_json))

    result = redeemer._lookup_neg_risk_info("cid-1", token_id="tok-1")

    assert result == {
        "neg_risk": True,
        "question_id": "0x01",
        "neg_risk_market_id": "nr-1",
        "question": "Example market",
    }


def test_redeemer_falls_back_to_secondary_rpc(monkeypatch):
    rpc_attempts = []

    class _FakeMiddleware:
        def inject(self, *_args, **_kwargs):
            return None

    class _FakeEth:
        def contract(self, **_kwargs):
            return object()

    class _FakeProvider:
        def __init__(self, rpc: str, request_kwargs=None):
            self.rpc = rpc
            self.request_kwargs = request_kwargs or {}

    class _FakeClient:
        def __init__(self, provider: _FakeProvider):
            self.rpc = provider.rpc
            self.middleware_onion = _FakeMiddleware()
            self.eth = _FakeEth()

        def is_connected(self):
            rpc_attempts.append(self.rpc)
            return self.rpc.endswith("publicnode.com")

    class _FakeWeb3:
        @staticmethod
        def HTTPProvider(rpc: str, request_kwargs=None):
            return _FakeProvider(rpc, request_kwargs)

        @staticmethod
        def to_checksum_address(address: str):
            return address

        def __new__(cls, provider):
            return _FakeClient(provider)

    monkeypatch.setattr("src.execution.redeemer.Web3", _FakeWeb3)

    redeemer = Redeemer("pk", "0xabc")
    # Construction should NOT connect; trigger lazy init
    assert rpc_attempts == []
    assert redeemer._ensure_connected() is True

    assert rpc_attempts == [
        "https://polygon-rpc.com",
        "https://polygon-bor-rpc.publicnode.com",
    ]
    assert redeemer.w3.rpc == "https://polygon-bor-rpc.publicnode.com"


def test_redeemer_degrades_gracefully_when_all_rpcs_fail(monkeypatch):
    class _FakeMiddleware:
        def inject(self, *_args, **_kwargs):
            return None

    class _FakeEth:
        def contract(self, **_kwargs):
            return object()

    class _FakeProvider:
        def __init__(self, rpc: str, request_kwargs=None):
            self.rpc = rpc
            self.request_kwargs = request_kwargs or {}

    class _FakeClient:
        def __init__(self, provider: _FakeProvider):
            self.rpc = provider.rpc
            self.middleware_onion = _FakeMiddleware()
            self.eth = _FakeEth()

        def is_connected(self):
            return False

    class _FakeWeb3:
        @staticmethod
        def HTTPProvider(rpc: str, request_kwargs=None):
            return _FakeProvider(rpc, request_kwargs)

        @staticmethod
        def to_checksum_address(address: str):
            return address

        def __new__(cls, provider):
            return _FakeClient(provider)

    monkeypatch.setattr("src.execution.redeemer.Web3", _FakeWeb3)

    # Construction succeeds even with all RPCs down
    redeemer = Redeemer("pk", "0xabc")
    # Methods degrade gracefully instead of crashing
    assert redeemer.is_resolved("0x" + "00" * 32) is None
    assert redeemer.redeem("0x" + "00" * 32) == 0.0

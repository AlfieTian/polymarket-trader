"""Tests for Redeemer token-aware neg-risk flows."""

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

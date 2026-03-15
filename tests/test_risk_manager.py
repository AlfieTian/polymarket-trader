"""Tests for RiskManager persistence across restarts."""

import json
import time

from src.risk import risk_manager as risk_module
from src.risk.risk_manager import RiskManager


def test_daily_risk_state_persists_across_restart(tmp_path, monkeypatch):
    state_file = tmp_path / "risk_state.json"
    history_file = tmp_path / "trade_history.json"
    history_file.write_text("[]")

    monkeypatch.setattr(risk_module, "_RISK_STATE_FILE", state_file)
    monkeypatch.setattr(risk_module, "_TRADE_HISTORY_FILE", history_file)

    rm = RiskManager(max_daily_loss_usdc=5)
    rm.record_pnl(-6)

    assert rm.is_halted
    assert state_file.exists()

    restarted = RiskManager(max_daily_loss_usdc=5)
    check = restarted.validate_trade("market-1", 1.0)

    assert restarted.is_halted
    assert not check.approved
    assert "Trading halted" in check.reason


def test_daily_risk_state_resets_on_new_day(tmp_path, monkeypatch):
    state_file = tmp_path / "risk_state.json"
    history_file = tmp_path / "trade_history.json"
    history_file.write_text("[]")

    monkeypatch.setattr(risk_module, "_RISK_STATE_FILE", state_file)
    monkeypatch.setattr(risk_module, "_TRADE_HISTORY_FILE", history_file)

    yesterday = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400))
    state_file.write_text(
        json.dumps(
            {
                "daily_pnl": -10.0,
                "daily_reset_date": yesterday,
                "halted": True,
            }
        )
    )

    rm = RiskManager(max_daily_loss_usdc=5)
    check = rm.validate_trade("market-1", 1.0)

    assert not rm.is_halted
    assert check.approved

    persisted = json.loads(state_file.read_text())
    assert persisted["daily_pnl"] == 0.0
    assert persisted["halted"] is False

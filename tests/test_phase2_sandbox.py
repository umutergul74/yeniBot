from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from yenibot.phase2 import CostScenario
from yenibot.phase2 import Phase2StrategyContract
from yenibot.phase2 import load_phase2_gate
from yenibot.phase2 import run_long_only_backtest
from yenibot.phase2.reporting import write_phase2_sandbox_report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _pending_report_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(
        path / "phase2_readiness.json",
        {
            "ready_for_phase2": False,
            "blockers": ["future_unseen_oos_not_ready"],
            "next_action": "wait_for_new_future_oos_rows",
        },
    )
    _write_json(
        path / "future_oos_readiness.json",
        {
            "evaluation_completed": False,
            "primary_candidate_passed": None,
            "promotion_allowed": False,
        },
    )
    _write_json(
        path / "report_consistency_audit.json",
        {
            "operator_next_step": {
                "consistency_status": "passed",
                "failed_checks": [],
                "next_action": "wait_for_new_future_oos_rows",
            }
        },
    )
    return path


def _ready_report_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(
        path / "phase2_readiness.json",
        {"ready_for_phase2": True, "blockers": [], "next_action": "phase2_allowed"},
    )
    _write_json(
        path / "future_oos_readiness.json",
        {
            "evaluation_completed": True,
            "primary_candidate_passed": True,
            "promotion_allowed": True,
        },
    )
    _write_json(
        path / "report_consistency_audit.json",
        {
            "operator_next_step": {
                "consistency_status": "passed",
                "failed_checks": [],
            }
        },
    )
    return path


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_close_time": pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 103, 110, 104, 105, 106, 107, 108],
            "low": [99, 100, 95, 102, 103, 104, 105, 106],
            "close": [100.5, 102, 103, 103.5, 104.5, 105.5, 106.5, 107.5],
            "atr_14": [1.0] * 8,
        }
    )


def test_pending_gate_allows_sandbox_but_blocks_official(tmp_path: Path) -> None:
    gate = load_phase2_gate(_pending_report_dir(tmp_path))

    assert gate.official_allowed is False
    assert gate.evidence_status == "sandbox_not_promotable_until_future_oos_passes"
    gate.assert_mode_allowed("sandbox")
    with pytest.raises(RuntimeError):
        gate.assert_mode_allowed("official")


def test_ready_gate_allows_official(tmp_path: Path) -> None:
    gate = load_phase2_gate(_ready_report_dir(tmp_path))

    assert gate.official_allowed is True
    gate.assert_mode_allowed("official")


def test_backtest_uses_next_bar_and_conservative_same_bar_exit(tmp_path: Path) -> None:
    gate = load_phase2_gate(_pending_report_dir(tmp_path))
    signals = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2026-01-01 01:00:00", tz="UTC")],
            "prob_long": [0.80],
        }
    )
    contract = Phase2StrategyContract(threshold=0.5, take_profit_atr=2.0, stop_loss_atr=5.0)

    result = run_long_only_backtest(
        _bars(),
        signals,
        gate=gate,
        contract=contract,
        cost_scenario=CostScenario(name="zero", entry_fee_bps=0, exit_fee_bps=0),
        mode="sandbox",
    )

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == pd.Timestamp("2026-01-01 02:00:00", tz="UTC")
    assert trade["entry_price"] == 102
    assert trade["exit_reason"] == "stop_loss_same_bar_conservative"
    assert trade["exit_price"] == 97
    assert trade["evidence_status"] == "sandbox_not_promotable_until_future_oos_passes"


def test_report_writer_marks_sandbox_as_not_promotable(tmp_path: Path) -> None:
    gate = load_phase2_gate(_pending_report_dir(tmp_path / "reports"))
    signals = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2026-01-01 01:00:00", tz="UTC")],
            "prob_long": [0.80],
        }
    )
    result = run_long_only_backtest(_bars(), signals, gate=gate)

    payload = write_phase2_sandbox_report(tmp_path / "phase2", result)

    assert payload["metadata"]["not_promotable_reason"] == (
        "future_oos_or_phase2_gate_not_passed"
    )
    assert (tmp_path / "phase2" / "phase2_trade_ledger.csv").exists()
    assert (tmp_path / "phase2" / "phase2_sandbox_report.md").exists()

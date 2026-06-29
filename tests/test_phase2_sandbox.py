from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from yenibot.experiment.artifacts import _write_experiment_slim_bundle
from yenibot.phase2 import CostScenario
from yenibot.phase2 import Phase2StrategyContract
from yenibot.phase2 import build_phase2_sandbox_inputs
from yenibot.phase2 import load_phase2_gate
from yenibot.phase2 import run_long_only_backtest
from yenibot.automation.phase2_sandbox import main as phase2_sandbox_main
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


def _frozen_prediction_fixture(tmp_path: Path) -> tuple[Path, Path]:
    report_dir = _pending_report_dir(tmp_path / "reports")
    checkpoint_dir = tmp_path / "checkpoints"
    scope_dir = (
        checkpoint_dir
        / "experiments"
        / "source_run"
        / "baseline_profile"
        / "replacement_recent3"
    )
    scope_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        report_dir / "frozen_candidate_manifest.json",
        {
            "available": True,
            "candidate_id": "control_recent3_equal_v2",
            "candidate_type": "recency_profile",
            "source_run_id": "source_run",
            "manifest_hash": "abc123",
            "expected_manifest_hash": "abc123",
            "threshold": {
                "value": 0.5,
                "source": "test_threshold",
                "selected_from": "pre_anchor_walk_forward_validation",
            },
            "components": [
                {
                    "profile": "baseline_profile",
                    "fold_scope": "replacement_recent3",
                    "scope_relative_path": "baseline_profile/replacement_recent3",
                    "model_count": 3,
                }
            ],
        },
    )
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=7,
                freq="1h",
                tz="UTC",
            ),
            "open": [100, 101, 102, 103, 104, 105, 106],
            "high": [101, 102, 104, 104, 106, 106, 108],
            "low": [99, 100, 101, 100, 103, 104, 105],
            "close": [100.5, 101.5, 103.0, 102.0, 105.0, 105.5, 107.0],
            "atr_14": [1.0] * 7,
            "prob_long": [0.9, 0.3, 0.8, 0.6, 0.4, 0.7, 0.2],
            "split": ["val", "test", "test", "test", "test", "test", "test"],
            "fold": [1, 1, 1, 1, 2, 2, 2],
            "label": [1, 0, 1, 0, 1, 1, 0],
            "forward_return": [0.01, -0.01, 0.02, -0.02, 0.01, 0.03, -0.01],
        }
    )
    predictions.to_csv(scope_dir / "predictions_all.csv", index=False)
    return report_dir, checkpoint_dir


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


def test_adapter_builds_inputs_from_frozen_candidate_predictions(tmp_path: Path) -> None:
    report_dir, checkpoint_dir = _frozen_prediction_fixture(tmp_path)

    result = build_phase2_sandbox_inputs(
        report_dir=report_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "phase2",
    )

    bars = pd.read_csv(result.bars_path)
    signals = pd.read_csv(result.signals_path)
    input_manifest = json.loads(result.input_manifest_path.read_text(encoding="utf-8"))
    assert result.candidate_id == "control_recent3_equal_v2"
    assert result.threshold == 0.5
    assert result.rows_read == 7
    assert result.rows_after_split_filter == 6
    assert result.bar_count == 6
    assert result.signal_count == 6
    assert list(bars.columns[:6]) == [
        "bar_close_time",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
    ]
    assert set(signals["split"]) == {"test"}
    assert signals["candidate_id"].eq("control_recent3_equal_v2").all()
    assert input_manifest["adapter_version"] == "phase2_input_adapter_v1"


def test_adapter_prefers_current_report_manifest_over_stale_preflight_path(
    tmp_path: Path,
) -> None:
    report_dir, checkpoint_dir = _frozen_prediction_fixture(tmp_path)
    stale_manifest_path = tmp_path / "stale" / "manifest_old.json"
    stale_manifest_path.parent.mkdir(parents=True)
    _write_json(
        stale_manifest_path,
        {
            "available": False,
            "candidate_id": "control_recent3_equal_v2",
            "candidate_type": "recency_profile",
            "source_run_id": "source_run",
            "manifest_hash": "abc123",
            "expected_manifest_hash": "<fill_after_05_generates_manifest_hash>",
            "unavailable_reasons": [
                "expected_manifest_hash_mismatch:"
                "<fill_after_05_generates_manifest_hash>:abc123"
            ],
            "threshold": {"value": 0.5},
            "components": [],
        },
    )
    _write_json(
        report_dir / "future_oos_preflight.json",
        {
            "primary_candidate": {
                "candidate_id": "control_recent3_equal_v2",
                "manifest_path": str(stale_manifest_path),
            }
        },
    )

    result = build_phase2_sandbox_inputs(
        report_dir=report_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "phase2",
    )

    assert result.candidate_manifest_path == report_dir / "frozen_candidate_manifest.json"
    assert result.signal_count == 6


def test_cli_auto_builds_inputs_and_writes_sandbox_report(tmp_path: Path) -> None:
    report_dir, checkpoint_dir = _frozen_prediction_fixture(tmp_path)
    output_dir = tmp_path / "phase2_cli"

    exit_code = phase2_sandbox_main(
        [
            "--report-dir",
            str(report_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            "sandbox",
            "--entry-fee-bps",
            "0",
            "--exit-fee-bps",
            "0",
            "--entry-slippage-bps",
            "0",
            "--exit-slippage-bps",
            "0",
        ]
    )

    report = json.loads((output_dir / "phase2_sandbox_report.json").read_text())
    assert exit_code == 0
    assert (output_dir / "phase2_bars.csv").exists()
    assert (output_dir / "phase2_signals.csv").exists()
    assert (output_dir / "phase2_input_manifest.json").exists()
    assert report["metadata"]["input_adapter"]["candidate_id"] == (
        "control_recent3_equal_v2"
    )
    assert report["summary"]["evidence_status"] == (
        "sandbox_not_promotable_until_future_oos_passes"
    )


def test_slim_bundle_includes_phase2_sandbox_directory(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "experiments" / "run"
    phase2_dir = report_dir / "phase2_sandbox"
    phase2_dir.mkdir(parents=True)
    (phase2_dir / "phase2_sandbox_report.json").write_text("{}", encoding="utf-8")
    (phase2_dir / "phase2_sandbox_report.md").write_text("# report\n", encoding="utf-8")
    (phase2_dir / "phase2_trade_ledger.csv").write_text("a\n", encoding="utf-8")

    slim_path, latest_path = _write_experiment_slim_bundle(
        output_dir=tmp_path / "reports",
        run_id="run",
        report_dir=report_dir,
    )

    with zipfile.ZipFile(slim_path) as archive:
        names = set(archive.namelist())
    assert latest_path.exists()
    assert "run/phase2_sandbox/phase2_sandbox_report.json" in names
    assert "run/phase2_sandbox/phase2_sandbox_report.md" in names
    assert "run/phase2_sandbox/phase2_trade_ledger.csv" in names

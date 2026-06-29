from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from yenibot.automation.phase2_lab import main as phase2_lab_main


def _write_phase2_bundle(path: Path) -> Path:
    source = path / "source" / "phase2_sandbox"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "bar_close_time": pd.date_range(
                "2026-01-01",
                periods=18,
                freq="1h",
                tz="UTC",
            ),
            "open": [100.0] * 18,
            "high": [
                100.4,
                100.5,
                102.4,
                100.2,
                100.3,
                100.4,
                102.5,
                100.2,
                100.3,
                100.4,
                100.5,
                102.6,
                100.2,
                100.3,
                100.4,
                100.5,
                102.7,
                100.2,
            ],
            "low": [
                99.8,
                99.7,
                99.6,
                99.7,
                99.8,
                99.7,
                99.6,
                99.7,
                99.8,
                99.7,
                99.8,
                99.6,
                99.7,
                99.8,
                99.7,
                99.8,
                99.6,
                99.7,
            ],
            "close": [100.0] * 18,
            "atr_14": [1.0] * 18,
        }
    ).to_csv(source / "phase2_bars.csv", index=False)
    pd.DataFrame(
        {
            "decision_time": pd.date_range(
                "2026-01-01",
                periods=17,
                freq="1h",
                tz="UTC",
            ),
            "prob_long": [
                0.90,
                0.40,
                0.88,
                0.60,
                0.42,
                0.91,
                0.30,
                0.89,
                0.55,
                0.41,
                0.92,
                0.35,
                0.87,
                0.58,
                0.45,
                0.93,
                0.44,
            ],
            "fold": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            "split": ["test"] * 17,
        }
    ).to_csv(source / "phase2_signals.csv", index=False)
    (source / "phase2_input_manifest.json").write_text(
        json.dumps(
            {
                "adapter_version": "phase2_input_adapter_v1",
                "result": {
                    "candidate_id": "control_recent3_equal_v2",
                    "threshold": 0.5,
                },
                "candidate_manifest": {
                    "candidate_id": "control_recent3_equal_v2",
                    "threshold": {"value": 0.5},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle = path / "phase2_latest_sandbox_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for item in source.rglob("*"):
            archive.write(item, item.relative_to(path / "source"))
    return bundle


def test_phase2_lab_runs_from_existing_bundle_and_writes_decision_report(
    tmp_path: Path,
) -> None:
    bundle = _write_phase2_bundle(tmp_path)
    output_dir = tmp_path / "lab"

    exit_code = phase2_lab_main(
        [
            "--bundle",
            str(bundle),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "test_run",
        ]
    )

    run_dir = output_dir / "test_run"
    decision = json.loads(
        (run_dir / "phase2_lab_decision_report.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert decision["lab_version"] == "phase2_local_lab_v1"
    assert decision["promotion_allowed"] is False
    assert decision["clean_confirmation_required"] is True
    assert decision["baseline_strategy_id"] == "baseline_fixed_atr_v1"
    assert decision["hypotheses"]
    assert (
        run_dir
        / "phase2_sandbox"
        / "phase2_strategy_forensics_summary.csv"
    ).exists()
    assert (
        run_dir
        / "phase2_sandbox"
        / "strategy_variants"
        / "score_margin_05_time_stop_6bar_v1"
        / "base"
        / "phase2_bootstrap_summary.json"
    ).exists()
    assert (run_dir / "phase2_lab_decision_report.md").exists()

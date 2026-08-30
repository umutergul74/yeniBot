"""Prepare one immutable label-free forward-shadow v2 deployment block."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.config import load_config
from yenibot.experiment.configuration import profile_config
from yenibot.features import filter_feature_columns, select_feature_columns
from yenibot.phase2.forward_shadow import (
    SHADOW_MANIFEST_VERSION,
    SHADOW_PROCESS_ID,
    _canonical_json,
    _utc,
    fit_initial_shadow_payoff,
    plan_shadow_block,
    predict_label_free_artifacts,
    seal_shadow_manifest,
    select_shadow_training_window,
)
from yenibot.phase2.full_oof import file_sha256
from yenibot.training.trainer import train_one_fold


SPEC_CANONICAL_SHA256 = "9e21b37a6608f1a3213c493df89424ddc54198fb60c520149695d4658fa599ba"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_identity(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    if subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet"], check=False
    ).returncode or subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False
    ).returncode:
        raise ValueError("Tracked repository changes must be committed before preparation")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_tree_clean": True,
    }


def _frame_membership_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    membership = frame[columns].copy()
    membership["timestamp"] = pd.to_datetime(
        membership.timestamp, utc=True
    ).map(lambda value: value.isoformat())
    return hashlib.sha256(
        membership.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
    }


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key, value in metrics.items():
        numeric = float(value)
        output[str(key)] = numeric if np.isfinite(numeric) else None
    return output


def _canonical_spec(path: Path) -> tuple[dict[str, Any], str]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    canonical_hash = hashlib.sha256(_canonical_json(spec)).hexdigest()
    if canonical_hash != SPEC_CANONICAL_SHA256:
        raise ValueError("Forward-shadow v2 machine contract differs from preregistration")
    if spec.get("version") != SHADOW_PROCESS_ID:
        raise ValueError("Forward-shadow v2 process identity mismatch")
    return spec, canonical_hash


def prepare_forward_shadow(
    *,
    labeled_path: str | Path,
    oof_targets_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "config.yaml",
    spec_path: str | Path = "configs/forward_shadow_v2.json",
    repo_dir: str | Path = ".",
    device: str | None = None,
) -> dict[str, Any]:
    """Fit one predeclared block in staging, parity-check it, then publish atomically."""

    labeled_path = Path(labeled_path)
    targets_path = Path(oof_targets_path)
    output = Path(output_dir)
    config_path = Path(config_path)
    spec_path = Path(spec_path)
    repo = Path(repo_dir).resolve()
    if not labeled_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError("Labeled data and pinned OOF targets are both required")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a forward-shadow block")
    spec, spec_hash = _canonical_spec(spec_path)
    expected_targets_hash = str(
        spec["source_evidence"]["initial_oof_targets_sha256"]
    )
    if file_sha256(targets_path) != expected_targets_hash:
        raise ValueError("Initial OOF target artifact differs from the pinned history")
    git_identity = _git_identity(repo)
    prepared_at = pd.Timestamp.now(tz="UTC")
    block = plan_shadow_block(spec, prepared_at=prepared_at)
    cfg = profile_config(dict(load_config(config_path)), str(spec["profile"]))
    labeled = pd.read_parquet(labeled_path)
    selected, fold = select_shadow_training_window(
        labeled, spec=spec, block_ordinal=int(block["ordinal"])
    )
    maturity = int(spec["model_schedule"]["minimum_label_maturity_hours"])
    if selected.timestamp.max() + pd.Timedelta(hours=maturity) > prepared_at:
        raise ValueError("Latest labeled training row has not reached minimum maturity")
    feature_columns = filter_feature_columns(select_feature_columns(selected), cfg)
    membership_columns = [
        "timestamp",
        "label",
        f"fwd_return_{int((cfg.get('labeling', {}) or {}).get('max_holding_bars', 10))}h",
        *feature_columns,
    ]
    missing_membership = [column for column in membership_columns if column not in selected]
    if missing_membership:
        raise ValueError(f"Training membership columns are missing: {missing_membership}")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        result = train_one_fold(
            selected,
            fold,
            feature_columns,
            cfg,
            checkpoint_dir=staging,
            device=device,
        )
        ordinal = int(block["ordinal"])
        model_path = staging / f"model_fold_{ordinal:03d}.pt"
        scaler_path = staging / f"scaler_fold_{ordinal:03d}.pkl"
        hmm_path = staging / f"hmm_fold_{ordinal:03d}.pkl"
        validation = (
            result["predictions"]
            .loc[lambda value: value.split.eq("val")]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        minimum_cdf = int(spec["score_transform"]["minimum_reference_rows"])
        if len(validation) < minimum_cdf:
            raise ValueError("Selected checkpoint produced too few validation CDF rows")
        cdf_path = staging / "validation_cdf.npy"
        np.save(cdf_path, validation.prob_long.to_numpy(dtype=float), allow_pickle=False)
        targets = pd.read_csv(targets_path, float_precision="round_trip")
        payoff = fit_initial_shadow_payoff(targets, spec=spec)
        locked_at = pd.Timestamp.now(tz="UTC")
        context_start = _utc(block["context_start_inclusive"])
        minimum_lock_lead = int(
            spec["model_schedule"]["minimum_lock_to_context_hours"]
        )
        if context_start < locked_at + pd.Timedelta(hours=minimum_lock_lead):
            raise RuntimeError("Preparation finished too late for safe manifest registration")
        block["locked_at_utc"] = locked_at.isoformat()
        feature_hash = hashlib.sha256(_canonical_json(feature_columns)).hexdigest()
        manifest_payload: dict[str, Any] = {
            "manifest_version": SHADOW_MANIFEST_VERSION,
            "process_id": SHADOW_PROCESS_ID,
            "candidate_id": f"{SHADOW_PROCESS_ID}::{block['block_id']}",
            "registration_status": "sealed_manifest_awaiting_git_registration",
            "block": block,
            "model": {
                "profile": spec["profile"],
                "feature_columns": feature_columns,
                "feature_columns_sha256": feature_hash,
                "seed": int((cfg.get("project", {}) or {}).get("random_seed", 42))
                + ordinal,
                "fit_selection_data": "train_and_validation_only",
                "post_fit_audit_influences_selection": False,
            },
            "payoff_layer": payoff,
            "training_audit": {
                "source_labeled_sha256": file_sha256(labeled_path),
                "selected_membership_sha256": _frame_membership_hash(
                    selected, membership_columns
                ),
                "selected_rows": int(len(selected)),
                "train_rows": int(len(fold.train)),
                "validation_rows": int(len(fold.val)),
                "post_fit_audit_rows": int(len(fold.test)),
                "selected_start_utc": selected.timestamp.min().isoformat(),
                "selected_end_utc": selected.timestamp.max().isoformat(),
                "validation_prediction_rows": int(len(validation)),
                "validation_metrics_are_diagnostics_not_selection_changes": True,
                "validation_metrics": _finite_metrics(result["val_metrics"]),
                "post_fit_metrics_are_parity_diagnostics_only": True,
                "post_fit_audit_metrics": _finite_metrics(result["test_metrics"]),
            },
            "source_evidence": {
                **spec["source_evidence"],
                "initial_oof_targets_verified_sha256": expected_targets_hash,
            },
            "contract": {
                "canonical_spec_sha256": spec_hash,
                "spec_file_sha256": file_sha256(spec_path),
                "config_file_sha256": file_sha256(config_path),
                "git": git_identity,
            },
            "artifacts": {
                "model": _artifact(model_path, staging),
                "scaler": _artifact(scaler_path, staging),
                "hmm": _artifact(hmm_path, staging),
                "validation_cdf": _artifact(cdf_path, staging),
            },
        }
        provisional = seal_shadow_manifest(manifest_payload)
        audit_source = selected.iloc[fold.test].reset_index(drop=True)
        observed = predict_label_free_artifacts(
            audit_source,
            manifest=provisional,
            artifact_root=staging,
            config=cfg,
            device=device,
            evidence_only=False,
        )
        expected = (
            result["predictions"]
            .loc[lambda value: value.split.eq("test")]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        parity = (
            len(observed) == len(expected)
            and np.array_equal(
                observed.timestamp.to_numpy(), expected.timestamp.to_numpy()
            )
            and np.array_equal(
                observed.raw_score.to_numpy(), expected.prob_long.to_numpy()
            )
        )
        if not parity or not len(observed):
            raise ValueError("Saved artifact label-free prediction parity failed")
        manifest_payload["post_fit_parity"] = {
            "passed": True,
            "rows": int(len(observed)),
            "exact_timestamp_match": True,
            "exact_score_match": True,
            "labels_or_returns_required_by_inference": False,
        }
        manifest = seal_shadow_manifest(manifest_payload)
        _write_json(staging / "forward_shadow_manifest.json", manifest)
        validation.to_parquet(staging / "validation_predictions.parquet", index=False)
        expected.to_parquet(staging / "post_fit_audit_predictions.parquet", index=False)
        result["history"].to_csv(staging / "training_history.csv", index=False)
        report = {
            "status": "sealed_manifest_awaiting_git_registration",
            "candidate_id": manifest["candidate_id"],
            "block_id": block["block_id"],
            "manifest_hash": manifest["manifest_hash"],
            "context_start_inclusive": block["context_start_inclusive"],
            "evidence_start_inclusive": block["evidence_start_inclusive"],
            "confirmation_clock_started": False,
            "next_required_action": (
                "commit_exact_manifest_then_create_registration_before_context_start"
            ),
        }
        _write_json(staging / "preparation_report.json", report)
        os.replace(staging, output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", required=True)
    parser.add_argument("--oof-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--spec", default="configs/forward_shadow_v2.json")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    report = prepare_forward_shadow(
        labeled_path=args.labeled,
        oof_targets_path=args.oof_targets,
        output_dir=args.output_dir,
        config_path=args.config,
        spec_path=args.spec,
        repo_dir=args.repo_dir,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

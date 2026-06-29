from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.contracts import Phase2StrategyContract
from yenibot.phase2.risk import Phase2RiskPolicy


DEFAULT_FORWARD_LOCK_PATH = Path(__file__).with_name("forward_lock.json")


@dataclass(frozen=True)
class Phase2ForwardInputs:
    bars: pd.DataFrame
    signals: pd.DataFrame
    boundary_report: dict[str, Any]


def _to_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def canonical_forward_lock_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("lock_hash", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_phase2_forward_lock(
    path: str | Path = DEFAULT_FORWARD_LOCK_PATH,
) -> dict[str, Any]:
    lock_path = Path(path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    configured_hash = str(payload.get("lock_hash", "")).strip().lower()
    computed_hash = canonical_forward_lock_hash(payload)
    if configured_hash != computed_hash:
        raise ValueError(
            "Phase 2 forward lock hash mismatch: "
            f"configured={configured_hash!r}, computed={computed_hash!r}"
        )
    if payload.get("success_gates", {}).get("automatic_promotion_allowed") is not False:
        raise ValueError("The clean-confirmation lock must disable automatic promotion.")
    if (
        payload.get("selection_evidence", {}).get(
            "automatic_winner_selection_used"
        )
        is not False
    ):
        raise ValueError("The forward lock cannot originate from automatic selection.")
    fit_end = _to_utc(payload["frozen_model"]["fit_data_end"])
    confirmation_start = _to_utc(
        payload["confirmation_window"]["decision_time_start_exclusive"]
    )
    if confirmation_start < fit_end:
        raise ValueError("Clean confirmation cannot begin before frozen fit end.")
    if int(payload["confirmation_window"]["minimum_trade_count"]) <= 0:
        raise ValueError("Clean confirmation minimum trade count must be positive.")
    if int(payload["confirmation_window"]["minimum_coverage_days"]) <= 0:
        raise ValueError("Clean confirmation coverage must be positive.")
    roles = [str(item.get("role", "")) for item in payload.get("candidates", [])]
    if len(roles) != 2 or len(set(roles)) != 2:
        raise ValueError("The forward lock must contain two uniquely named candidates.")
    return payload


def locked_strategy_contracts(
    lock: dict[str, Any],
) -> dict[str, Phase2StrategyContract]:
    frozen = lock["frozen_model"]
    contracts: dict[str, Phase2StrategyContract] = {}
    for item in lock["candidates"]:
        role = str(item["role"])
        contract = replace(
            DEFAULT_PHASE2_CONTRACT,
            strategy_id=str(item["strategy_id"]),
            candidate_id=str(frozen["candidate_id"]),
            threshold=float(frozen["threshold"]),
            **dict(item["contract"]),
        )
        contract.validate()
        contracts[role] = contract
    return contracts


def locked_risk_policy(lock: dict[str, Any]) -> Phase2RiskPolicy:
    policy = Phase2RiskPolicy(**dict(lock["risk_policy"]))
    policy.validate()
    return policy


def validate_forward_input_manifest(
    lock: dict[str, Any],
    input_manifest: dict[str, Any],
) -> dict[str, Any]:
    frozen = lock["frozen_model"]
    candidate = input_manifest.get("candidate_manifest", {}) or {}
    result = input_manifest.get("result", {}) or {}
    threshold_payload = candidate.get("threshold", {}) or {}
    observed_threshold = result.get("threshold", threshold_payload.get("value"))
    checks = {
        "candidate_id": (
            str(result.get("candidate_id") or candidate.get("candidate_id") or "")
            == str(frozen["candidate_id"])
        ),
        "candidate_manifest_hash": (
            str(candidate.get("manifest_hash") or "").lower()
            == str(frozen["candidate_manifest_hash"]).lower()
        ),
        "threshold": (
            observed_threshold is not None
            and abs(float(observed_threshold) - float(frozen["threshold"])) <= 1e-12
        ),
        "source_run_id": (
            str(candidate.get("source_run_id") or "")
            == str(frozen["source_run_id"])
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Forward input manifest does not match the frozen lock: "
            f"{failed}"
        )
    return {"passed": True, "checks": checks, "failed_checks": []}


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def filter_clean_confirmation_inputs(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lock: dict[str, Any],
) -> Phase2ForwardInputs:
    bar_time_column = DEFAULT_PHASE2_CONTRACT.bar_time_column
    decision_time_column = DEFAULT_PHASE2_CONTRACT.decision_time_column
    if bar_time_column not in bars.columns:
        raise ValueError(f"Bars are missing {bar_time_column!r}.")
    if decision_time_column not in signals.columns:
        raise ValueError(f"Signals are missing {decision_time_column!r}.")

    clean_bars = bars.copy()
    clean_signals = signals.copy()
    clean_bars[bar_time_column] = clean_bars[bar_time_column].map(_to_utc)
    clean_signals[decision_time_column] = clean_signals[decision_time_column].map(
        _to_utc
    )
    cutoff = _to_utc(
        lock["confirmation_window"]["decision_time_start_exclusive"]
    )
    accepted_signals = clean_signals.loc[
        clean_signals[decision_time_column] > cutoff
    ].copy()
    accepted_bars = clean_bars.loc[clean_bars[bar_time_column] > cutoff].copy()

    frozen = lock["frozen_model"]
    if "candidate_id" in accepted_signals.columns and not accepted_signals.empty:
        candidate_ids = set(accepted_signals["candidate_id"].dropna().astype(str))
        if candidate_ids != {str(frozen["candidate_id"])}:
            raise ValueError(
                "Post-lock signals contain an unexpected candidate id: "
                f"{sorted(candidate_ids)}"
            )
    if "threshold" in accepted_signals.columns and not accepted_signals.empty:
        thresholds = pd.to_numeric(
            accepted_signals["threshold"],
            errors="coerce",
        ).dropna()
        if not thresholds.map(
            lambda value: abs(value - float(frozen["threshold"])) <= 1e-12
        ).all():
            raise ValueError("Post-lock signals contain an unexpected threshold.")

    accepted_signals = accepted_signals.sort_values(
        decision_time_column
    ).reset_index(drop=True)
    accepted_bars = accepted_bars.sort_values(bar_time_column).reset_index(drop=True)
    earliest = (
        accepted_signals[decision_time_column].min()
        if not accepted_signals.empty
        else None
    )
    latest = (
        accepted_signals[decision_time_column].max()
        if not accepted_signals.empty
        else None
    )
    coverage_days = (
        float((latest - earliest).total_seconds() / 86_400.0)
        if earliest is not None and latest is not None
        else 0.0
    )
    report = {
        "lock_version": lock["lock_version"],
        "lock_hash": lock["lock_hash"],
        "cutoff_exclusive": cutoff.isoformat(),
        "input_signal_count": int(len(clean_signals)),
        "excluded_pre_anchor_signal_count": int(
            (clean_signals[decision_time_column] <= cutoff).sum()
        ),
        "accepted_signal_count": int(len(accepted_signals)),
        "input_bar_count": int(len(clean_bars)),
        "accepted_bar_count": int(len(accepted_bars)),
        "earliest_accepted_decision_time": (
            earliest.isoformat() if earliest is not None else None
        ),
        "latest_accepted_decision_time": (
            latest.isoformat() if latest is not None else None
        ),
        "coverage_days": coverage_days,
        "waiting_for_post_anchor_data": bool(accepted_signals.empty),
        "accepted_signals_sha256": _frame_fingerprint(accepted_signals),
        "accepted_bars_sha256": _frame_fingerprint(accepted_bars),
        "fit_operations_performed": 0,
        "selection_operations_performed": 0,
    }
    return Phase2ForwardInputs(
        bars=accepted_bars,
        signals=accepted_signals,
        boundary_report=report,
    )


def forward_lock_snapshot(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        **lock,
        "resolved_contracts": {
            role: asdict(contract)
            for role, contract in locked_strategy_contracts(lock).items()
        },
        "resolved_risk_policy": asdict(locked_risk_policy(lock)),
    }

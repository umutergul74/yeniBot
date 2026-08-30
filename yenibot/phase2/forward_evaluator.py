"""Mature-only evaluation for registered block-prequential shadow evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT, CostScenario
from yenibot.phase2.costs import net_long_return
from yenibot.phase2.engine import (
    _resolve_exit,
    run_long_only_backtest,
)
from yenibot.phase2.forward_shadow import (
    _utc,
    validate_shadow_manifest,
    validate_shadow_registration,
)
from yenibot.phase2.readiness import Phase2Gate


@dataclass(frozen=True)
class ForwardBlockEvaluation:
    report: dict[str, Any]
    trades: dict[str, pd.DataFrame]
    equity: dict[str, pd.DataFrame]
    opportunities: pd.DataFrame


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(selected[column]):
            selected[column] = pd.to_datetime(
                selected[column], utc=True
            ).map(lambda value: value.isoformat())
    return hashlib.sha256(
        selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _market_bars(
    market: pd.DataFrame, *, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    required = ["timestamp", "open", "high", "low", "close", "atr_14"]
    missing = [column for column in required if column not in market]
    if missing:
        raise ValueError(f"Forward evaluation market is missing columns: {missing}")
    frame = market[required].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    frame[required[1:]] = frame[required[1:]].apply(pd.to_numeric, errors="raise")
    if (
        frame.timestamp.isna().any()
        or frame.timestamp.duplicated().any()
        or not np.isfinite(frame[required[1:]].to_numpy(dtype=float)).all()
        or (frame.atr_14 <= 0).any()
    ):
        raise ValueError("Forward evaluation market contains invalid bars")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    block = manifest["block"]
    evidence_start = _utc(block["evidence_start_inclusive"])
    evidence_end = _utc(block["evidence_end_inclusive"])
    holding = int(DEFAULT_PHASE2_CONTRACT.max_holding_bars)
    expected_open = pd.date_range(
        evidence_start - pd.Timedelta(hours=1),
        evidence_end + pd.Timedelta(hours=holding - 1),
        freq="h",
        tz="UTC",
    )
    selected = frame.loc[frame.timestamp.isin(expected_open)].copy()
    if not pd.DatetimeIndex(selected.timestamp).equals(expected_open):
        raise ValueError("Forward outcome market is incomplete or non-contiguous")
    selected["bar_open_time"] = selected.timestamp
    selected["bar_close_time"] = selected.timestamp + pd.Timedelta(hours=1)
    return selected, expected_open


def shadow_block_preflight(
    ledger: pd.DataFrame,
    market: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    registration: dict[str, Any],
) -> dict[str, Any]:
    checked = validate_shadow_manifest(manifest)
    validate_shadow_registration(registration, manifest=checked)
    block = checked["block"]
    expected = pd.date_range(
        _utc(block["evidence_start_inclusive"]),
        _utc(block["evidence_end_inclusive"]),
        freq="h",
        tz="UTC",
    )
    reasons: list[str] = []
    frame = ledger.copy()
    required = [
        "decision_time",
        "evidence_role",
        "process_id",
        "candidate_id",
        "block_id",
        "block_manifest_hash",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        reasons.append(f"ledger_missing_columns:{','.join(missing)}")
        decisions = pd.DatetimeIndex([])
        timely = frame.iloc[0:0]
    else:
        frame["decision_time"] = pd.to_datetime(
            frame.decision_time, utc=True, errors="coerce"
        )
        identities = {
            "process_id": checked["process_id"],
            "candidate_id": checked["candidate_id"],
            "block_id": block["block_id"],
            "block_manifest_hash": checked["manifest_hash"],
        }
        if any(
            not frame[key].astype(str).eq(str(value)).all()
            for key, value in identities.items()
        ):
            reasons.append("ledger_identity_mismatch")
        if frame.decision_time.isna().any() or frame.decision_time.duplicated().any():
            reasons.append("ledger_decision_identity_invalid")
        timely = frame.loc[frame.evidence_role.eq("timely_shadow")].copy()
        decisions = pd.DatetimeIndex(timely.decision_time)
        if not decisions.equals(expected):
            reasons.append("timely_evidence_clock_incomplete")
        if len(frame) != len(expected):
            reasons.append("ledger_row_count_differs_from_sealed_evidence_hours")
        if not frame.evidence_role.eq("timely_shadow").all():
            reasons.append("late_or_replayed_rows_present")
    market_complete = False
    try:
        _market_bars(market, manifest=checked)
        market_complete = True
    except ValueError:
        reasons.append("mature_outcome_market_incomplete")
    return {
        "ready_for_complete_block_evaluation": not reasons,
        "block_id": block["block_id"],
        "expected_evidence_rows": int(len(expected)),
        "recorded_rows": int(len(frame)),
        "timely_rows": int(len(timely)),
        "market_complete": market_complete,
        "failed_checks": reasons,
        "fit_operations_performed": 0,
        "selection_operations_performed": 0,
    }


def _gate() -> Phase2Gate:
    return Phase2Gate(
        report_dir=Path("forward_shadow_v2"),
        ready_for_phase2=False,
        report_consistency_passed=True,
        future_oos_evaluation_completed=True,
        future_oos_candidate_passed=False,
        promotion_allowed=False,
        blockers=("forward_shadow_evidence_only",),
        advisories=("never_live_or_auto_promote",),
    )


def _policy_summary(result) -> dict[str, Any]:
    completed = result.trades.loc[
        result.trades.get("trade_status", pd.Series(dtype=str)).eq("completed")
    ].copy()
    occupied = float(completed.get("holding_hours", pd.Series(dtype=float)).sum())
    net_sum = float(completed.get("net_return", pd.Series(dtype=float)).sum())
    wins = float(
        completed.loc[completed.get("net_return", pd.Series(dtype=float)).gt(0), "net_return"].sum()
    ) if not completed.empty else 0.0
    losses = float(
        -completed.loc[completed.get("net_return", pd.Series(dtype=float)).lt(0), "net_return"].sum()
    ) if not completed.empty else 0.0
    return {
        "trade_count": int(len(completed)),
        "compounded_return": float(result.summary["compounded_return"]),
        "completed_trade_compounded_return": float(
            result.summary["completed_trade_compounded_return"]
        ),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "max_hourly_marked_drawdown": float(result.summary["max_drawdown"]),
        "net_return_sum": net_sum,
        "winning_return_sum": wins,
        "losing_return_abs_sum": losses,
        "occupied_hours": occupied,
        "net_bps_per_occupied_hour": (
            net_sum * 10_000.0 / occupied if occupied > 0 else None
        ),
        "data_contract_complete": bool(result.summary["data_contract_complete"]),
        "censored_position_count": int(result.summary["censored_position_count"]),
    }


def _opportunity_payoffs(
    bars: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    scenario: CostScenario,
) -> pd.DataFrame:
    contract = replace(
        DEFAULT_PHASE2_CONTRACT,
        threshold=0.5,
        score_column="candidate_score",
    )
    by_open = {time: index for index, time in enumerate(bars.bar_open_time)}
    rows = []
    for row in ledger.sort_values("decision_time").itertuples(index=False):
        decision = _utc(row.decision_time)
        entry_idx = by_open.get(decision)
        payoff = np.nan
        outcome_time = pd.NaT
        if (
            entry_idx is not None
            and entry_idx > 0
            and bars.loc[entry_idx - 1, "bar_close_time"] == decision
        ):
            atr = float(bars.loc[entry_idx - 1, "atr_14"])
            entry_price = float(bars.loc[entry_idx, "open"])
            resolution = _resolve_exit(
                bars,
                contract,
                entry_idx=entry_idx,
                entry_price=entry_price,
                atr=atr,
            )
            outcome_time = bars.loc[
                resolution.exit_idx,
                "bar_open_time" if resolution.exit_at_open else "bar_close_time",
            ]
            if not resolution.exit_reason.endswith("_censored"):
                payoff = net_long_return(
                    scenario,
                    entry_price=entry_price,
                    exit_price=resolution.exit_price,
                    holding_hours=(outcome_time - decision).total_seconds() / 3600.0,
                )["net_return"]
        rows.append(
            {
                "decision_time": decision,
                "score_percentile": float(row.score_percentile),
                "adverse_opportunity_net_return": payoff,
                "outcome_time": outcome_time,
            }
        )
    return pd.DataFrame(rows)


def rank_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(outcomes)
    if valid.sum() < 3:
        return np.nan
    return float(
        pd.Series(scores[valid]).rank(method="average").corr(
            pd.Series(outcomes[valid]).rank(method="average")
        )
    )


def circular_block_mean_interval(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or block_length <= 0
        or block_length > len(values)
        or replicates < 100
    ):
        raise ValueError("Invalid circular block bootstrap inputs")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block_length))
    draws = np.empty(replicates)
    offsets = np.arange(block_length)
    for replicate in range(replicates):
        starts = rng.integers(0, len(values), size=blocks)
        indices = ((starts[:, None] + offsets) % len(values)).ravel()[: len(values)]
        draws[replicate] = values[indices].mean()
    return {
        "actual_mean": float(values.mean()),
        "lower95": float(np.quantile(draws, 0.025)),
        "upper95": float(np.quantile(draws, 0.975)),
    }


def evaluate_complete_shadow_block(
    ledger: pd.DataFrame,
    market: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    registration: dict[str, Any],
) -> ForwardBlockEvaluation:
    preflight = shadow_block_preflight(
        ledger,
        market,
        manifest=manifest,
        registration=registration,
    )
    if not preflight["ready_for_complete_block_evaluation"]:
        raise ValueError(
            f"Forward block is not complete: {preflight['failed_checks']}"
        )
    checked = validate_shadow_manifest(manifest)
    bars, _ = _market_bars(market, manifest=checked)
    evidence = ledger.sort_values("decision_time").reset_index(drop=True)
    scenarios = {
        scenario.name: scenario
        for scenario in DEFAULT_PHASE2_CONTRACT.cost_scenarios
        if scenario.name in {"base", "adverse"}
    }
    reports: dict[str, Any] = {}
    trades: dict[str, pd.DataFrame] = {}
    equity: dict[str, pd.DataFrame] = {}
    for policy, score_column in (
        ("candidate", "candidate_score"),
        ("atr_only", "atr_score"),
    ):
        signals = evidence[["decision_time", score_column]].copy()
        contract = replace(
            DEFAULT_PHASE2_CONTRACT,
            strategy_id=f"forward_shadow_v2_{policy}",
            candidate_id=checked["candidate_id"],
            score_column=score_column,
            threshold=0.5,
        )
        for scenario_name, scenario in scenarios.items():
            key = f"{policy}_{scenario_name}"
            result = run_long_only_backtest(
                bars,
                signals,
                gate=_gate(),
                contract=contract,
                cost_scenario=scenario,
                mode="sandbox",
            )
            reports[key] = _policy_summary(result)
            trades[key] = result.trades
            equity[key] = result.equity
    opportunities = _opportunity_payoffs(
        bars, evidence, scenario=scenarios["adverse"]
    )
    ic = rank_ic(
        opportunities.score_percentile.to_numpy(),
        opportunities.adverse_opportunity_net_return.to_numpy(),
    )
    report = {
        "status": "complete_forward_shadow_block_evaluated",
        "process_id": checked["process_id"],
        "candidate_id": checked["candidate_id"],
        "block_id": checked["block"]["block_id"],
        "block_ordinal": int(checked["block"]["ordinal"]),
        "manifest_hash": checked["manifest_hash"],
        "registration_hash": registration["registration_hash"],
        "preflight": preflight,
        "policies": reports,
        "candidate_minus_atr_block_return": {
            scenario: (
                reports[f"candidate_{scenario}"]["compounded_return"]
                - reports[f"atr_only_{scenario}"]["compounded_return"]
            )
            for scenario in ("base", "adverse")
        },
        "score_payoff_rank_ic_adverse": ic,
        "opportunity_rows": int(
            opportunities.adverse_opportunity_net_return.notna().sum()
        ),
        "market_membership_sha256": _frame_hash(
            bars,
            ["bar_open_time", "open", "high", "low", "close", "atr_14"],
        ),
        "fit_operations_performed": 0,
        "selection_operations_performed": 0,
        "promotion_allowed": False,
        "live_trading_allowed": False,
    }
    return ForwardBlockEvaluation(report, trades, equity, opportunities)

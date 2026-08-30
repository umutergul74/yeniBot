"""Audit frozen fills; download only missing held-event minute mark intervals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.full_oof import build_full_oof_inputs, file_sha256
from yenibot.phase2.ledger_risk_audit import (
    funding_price_scenarios,
    held_funding_mask,
    replay_frozen_ledger,
    validate_frozen_ledger,
    validate_funding_grid,
)

PROBE_SHA256 = "41237d1a3298276b60add605c90cc3d4cef5f63bf2716c16e50f95516d560524"
FUNDING_SHA256 = "25119a0e1b66f09709091ad7dbe685810f2271476507c015a94c3b41881658dc"
ENDPOINT = "https://fapi.binance.com/fapi/v1/markPriceKlines"


def _write(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _check_hash(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"Changed pinned artifact: {path}")


def artifact_path(root, relative):
    """Read historical Windows manifests on Colab without permitting traversal."""
    root = root.resolve()
    relative = str(relative).replace("\\", "/")
    if ":" in relative or relative.startswith("/"):
        raise ValueError("Artifact must use a relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Artifact path escapes its bundle")
    return path


def parse_mark_minute(payload, minute):
    expected = int(minute.timestamp() * 1000)
    if not isinstance(payload, list) or len(payload) != 1 or len(payload[0]) < 7:
        raise ValueError("Missing/ambiguous Binance mark minute")
    row = payload[0]
    if int(row[0]) != expected or int(row[6]) != expected + 59999:
        raise ValueError("Mark minute does not cover requested settlement time")
    values = np.array(row[1:5], dtype=float)
    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or values[1] < values.max()
        or values[2] > values.min()
    ):
        raise ValueError("Invalid mark-minute OHLC geometry")
    return dict(zip(["open", "high", "low", "close"], values.tolist()))


def fetch_mark_minutes(minutes, output):
    path = output / "raw_mark_minutes"
    path.mkdir(exist_ok=True)
    result = {}
    with requests.Session() as session:
        for i, minute in enumerate(sorted(minutes)):
            file = path / (minute.strftime("%Y%m%d_%H%M") + ".json")
            if file.exists():
                cached = json.loads(file.read_text(encoding="utf-8"))
                if (
                    cached["endpoint"] != ENDPOINT
                    or cached["minute"] != minute.isoformat()
                ):
                    raise ValueError("Cached mark-minute request identity differs")
                payload = cached["payload"]
            else:
                start = int(minute.timestamp() * 1000)
                response = session.get(
                    ENDPOINT,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "1m",
                        "startTime": start,
                        "endTime": start + 59999,
                        "limit": 1,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                parse_mark_minute(payload, minute)
                _write(
                    file,
                    {
                        "endpoint": ENDPOINT,
                        "minute": minute.isoformat(),
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "payload": payload,
                    },
                )
            result[minute] = parse_mark_minute(payload, minute)
            if i == 0 or (i + 1) % 10 == 0 or i == len(minutes) - 1:
                print(f"Mark-minute data checkpoint {i + 1}/{len(minutes)}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--funding",
        default="data/raw/snapshots/20260830_integrity_v2/btc_funding_rates.parquet",
    )
    args = parser.parse_args(argv)
    probe = Path(args.probe_dir)
    report_path = probe / "prequential_probe_result.json"
    funding_path = Path(args.funding)
    _check_hash(report_path, PROBE_SHA256)
    _check_hash(funding_path, FUNDING_SHA256)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for relative, expected in report["artifact_sha256"].items():
        _check_hash(artifact_path(probe, relative), expected)
    source = json.loads((probe / "source_manifest.json").read_text(encoding="utf-8"))
    bars, frozen, _ = build_full_oof_inputs(args.scope_dir, spec=source["spec"])
    bars["fold"] = frozen.fold.to_numpy()
    bars = bars.loc[bars.fold.isin(report["evaluation_fold_ids"])].reset_index(
        drop=True
    )
    ledgers = {
        "candidate": pd.read_csv(
            probe / "attribution/phase2_attribution_trade_ledger.csv",
            float_precision="round_trip",
        ),
        "atr_only": pd.read_csv(
            probe / "atr_only_base_trades.csv", float_precision="round_trip"
        ),
    }
    for name, ledger in ledgers.items():
        ledgers[name] = validate_frozen_ledger(
            ledger.loc[ledger.cost_scenario.eq("base")].copy(), bars
        )
    funding = validate_funding_grid(
        pd.read_parquet(funding_path),
        start=bars.bar_open_time.min(),
        end=bars.bar_close_time.max(),
    )
    needed = held_funding_mask(funding, list(ledgers.values()))
    missing = funding.mark_price.isna() & needed
    minutes = set(funding.loc[missing, "timestamp"].dt.floor("min"))
    output = Path(args.output_dir)
    identity = {
        "version": "frozen_ledger_risk_audit_v1",
        "probe_sha256": PROBE_SHA256,
        "funding_sha256": FUNDING_SHA256,
        "implementation_sha256": {
            str(p): file_sha256(p)
            for p in (
                Path(__file__),
                Path("yenibot/phase2/ledger_risk_audit.py"),
                Path("yenibot/phase2/accounting.py"),
                Path("yenibot/phase2/costs.py"),
            )
        },
    }
    if (output / "risk_audit_result.json").exists():
        raise FileExistsError("Completed risk audit is immutable")
    if output.exists() and any(output.iterdir()):
        if (
            not (output / "audit_identity.json").exists()
            or json.loads((output / "audit_identity.json").read_text()) != identity
        ):
            raise ValueError("Cannot resume a different/unknown audit")
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "audit_identity.json", identity)
    print(
        f"Source verified; {len(minutes)} missing held-event mark minutes to retrieve",
        flush=True,
    )
    mark_bars = fetch_mark_minutes(minutes, output)
    scenarios = funding_price_scenarios(funding, mark_bars, needed_mask=needed)
    for name, events in scenarios.items():
        events.to_csv(output / f"funding_{name}.csv", index=False)
    summaries = {}
    for policy, ledger in ledgers.items():
        summaries[policy] = {}
        for cost in ("base", "adverse"):
            scenario = next(
                c for c in DEFAULT_PHASE2_CONTRACT.cost_scenarios if c.name == cost
            )
            for mode, events in (("original_fixed_funding", None), *scenarios.items()):
                for liquidate in (False, True):
                    key = f"{cost}_{mode}_{'terminal_liquidated' if liquidate else 'terminal_marked'}"
                    summary, curve, trades = replay_frozen_ledger(
                        bars,
                        ledger,
                        contract=DEFAULT_PHASE2_CONTRACT,
                        scenario=scenario,
                        funding_events=events,
                        liquidate_terminal=liquidate,
                    )
                    if mode == "original_fixed_funding":
                        expected = (
                            report["actual"][cost]
                            if policy == "candidate"
                            else report["same_cohort_controls"][policy][cost]
                        )
                        if not np.isclose(
                            summary["completed_trade_return"],
                            expected["compounded_return"],
                            rtol=1e-12,
                            atol=1e-12,
                        ):
                            raise RuntimeError(
                                "Original completed-trade result does not reconcile"
                            )
                    summaries[policy][key] = summary
                    curve.to_csv(output / f"{policy}_{key}_equity.csv", index=False)
                    trades.to_csv(output / f"{policy}_{key}_trades.csv", index=False)
            _write(
                output / "run_checkpoint.json",
                {
                    "stage": "replaying_frozen_ledgers",
                    "last_policy": policy,
                    "last_cost": cost,
                    "fit_operations": 0,
                },
            )
            print(f"Frozen-ledger replay completed: {policy}/{cost}", flush=True)
    result = {
        **identity,
        "status": "accounting_audit_complete_original_acceptance_still_failed",
        "fit_operations": 0,
        "new_trade_selection_operations": 0,
        "evaluation_fold_ids": report["evaluation_fold_ids"],
        "evaluation_rows": len(bars),
        "historical_rate_events_in_grid": len(funding),
        "union_held_events": int(needed.sum()),
        "union_missing_settlement_marks": int(missing.sum()),
        "retrieved_mark_minutes": len(mark_bars),
        "exact_complete_historical_funding_available": not bool(missing.any()),
        "funding_price_sensitivity_assumption": "Missing settlement mark lies within official one-minute mark OHLC; this is not an exact settlement reconstruction or guaranteed bound.",
        "original_failed_criteria": report["failed_criteria"],
        "summaries": summaries,
        "limitations": [
            "Historical policy/profile selection remains",
            "Independent fold chaining is not a continuous live portfolio",
            "Hourly contract-close equity misses intrabar and exchange mark-price liquidation risk",
            "Intrabar exits retain original close-time proxy",
            "Fixed unit reference notional is not an equal-risk comparison",
            "Missing mark-price ranges are sensitivities, not exact funding",
        ],
        "promotion_allowed": False,
        "live_trading_allowed": False,
        "artifact_sha256": {
            p.relative_to(output).as_posix(): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "run_checkpoint.json"
        },
    }
    _write(output / "risk_audit_result.json", result)
    _write(
        output / "run_checkpoint.json",
        {
            "stage": "completed",
            "fit_operations": 0,
            "original_acceptance_gate_passed": False,
        },
    )
    print(
        "Risk accounting audit complete; original failed gates unchanged.", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

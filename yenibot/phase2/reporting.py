from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.phase2.engine import Phase2BacktestResult


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def phase2_sandbox_markdown(result: Phase2BacktestResult) -> str:
    summary = result.summary
    metadata = result.metadata
    lines = [
        "# Phase 2 Sandbox Report",
        "",
        f"Mode: `{metadata.get('mode')}`",
        f"Evidence status: `{summary.get('evidence_status')}`",
        "",
        "> This report is sandbox evidence unless the Phase 2 gate is fully open.",
        "",
        "## Summary",
        "",
        f"- Strategy: `{metadata.get('contract', {}).get('strategy_id')}`",
        f"- Exit policy: `{metadata.get('contract', {}).get('exit_policy')}`",
        "- Minimum score margin: "
        f"`{metadata.get('contract', {}).get('min_score_margin')}`",
        f"- Trade count: `{summary.get('trade_count')}`",
        f"- Mean net return: `{summary.get('mean_net_return')}`",
        f"- Sum net return: `{summary.get('sum_net_return')}`",
        f"- Hit rate: `{summary.get('hit_rate')}`",
        f"- Profit factor: `{summary.get('profit_factor')}`",
        f"- Compounded return: `{summary.get('compounded_return')}`",
        f"- Final equity: `{summary.get('final_equity')}`",
        f"- Maximum drawdown: `{summary.get('max_drawdown')}`",
        f"- Cost scenario: `{summary.get('cost_scenario')}`",
        f"- Accounting version: `{summary.get('accounting_version')}`",
        f"- Equity basis: `{metadata.get('equity_basis')}`",
        f"- Funding basis: `{summary.get('funding_basis')}`",
        f"- Censored positions (not completed exits): `{summary.get('censored_position_count')}`",
        f"- Selected signals: `{summary.get('selected_signal_count')}`",
        "- Entry-filter passed/skipped: "
        f"`{summary.get('entry_filter_passed_count')}` / "
        f"`{summary.get('entry_filter_skipped_count')}`",
        f"- Skipped stale entries: `{summary.get('skipped_stale_entry_count')}`",
        "- Skipped signals during open position: "
        f"`{summary.get('skipped_during_open_position_count')}`",
        f"- Data-gap censored positions: `{summary.get('data_gap_forced_close_count')}`",
        f"- Max entry delay hours: `{summary.get('max_entry_delay_hours')}`",
        "",
        "## Gate",
        "",
        f"- Official allowed: `{metadata.get('gate', {}).get('official_allowed')}`",
        f"- Blockers: `{metadata.get('gate', {}).get('blockers')}`",
        f"- Next action: `{metadata.get('gate', {}).get('next_action')}`",
    ]
    return "\n".join(lines) + "\n"


def write_phase2_sandbox_report(
    output_dir: str | Path,
    result: Phase2BacktestResult,
) -> dict[str, Any]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(path / "phase2_trade_ledger.csv", index=False)
    result.equity.to_csv(path / "phase2_equity_curve.csv", index=False)
    payload = {
        "summary": _json_ready(result.summary),
        "metadata": _json_ready(result.metadata),
    }
    (path / "phase2_sandbox_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "phase2_sandbox_report.md").write_text(
        phase2_sandbox_markdown(result),
        encoding="utf-8",
    )
    return payload

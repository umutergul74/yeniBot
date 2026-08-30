"""Download a versioned, closed-candle market snapshot without training/reselection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

from yenibot.data.binance import (
    download_full_klines,
    download_funding_rates,
    interval_to_milliseconds,
)
from yenibot.data.validation import validate_full_kline_frame
from yenibot.experiment.oos_integrity import file_sha256


def closed_boundary(now, interval):
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    return now.floor(pd.Timedelta(milliseconds=interval_to_milliseconds(interval)))


def refresh_snapshot(config, output_dir: Path, *, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Snapshots are immutable; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config["binance"]
    symbol, start = cfg["symbol"], cfg["start_date"]
    intervals = list(
        dict.fromkeys(
            [
                cfg["primary_interval"],
                cfg["htf_interval"],
                *cfg.get("intrabar_intervals", []),
            ]
        )
    )

    def session():
        http = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        http.mount("https://", HTTPAdapter(max_retries=retry))
        return http

    def fetch(interval):
        end = closed_boundary(now, interval)
        print(f"Downloading {symbol} {interval}: {start} -> {end}", flush=True)
        with session() as http:
            raw = download_full_klines(
                symbol,
                interval,
                start,
                end,
                base_url=cfg["base_url"],
                vision_base_url=cfg["vision_base_url"],
                data_source=cfg.get("data_source", "auto"),
                session=http,
                request_sleep_seconds=cfg.get("request_sleep_seconds", 0.15),
            )
        raw = raw.loc[raw.close_time < end].copy()
        frame = validate_full_kline_frame(
            raw,
            interval,
            max_gap_multiplier=cfg.get("intrabar_max_gap_multiplier", 8)
            if interval in cfg.get("intrabar_intervals", [])
            else cfg.get("max_gap_multiplier", 2),
            zero_volume_policy=cfg.get("zero_volume_policy", "error"),
        )
        if frame.timestamp.iloc[0] != pd.Timestamp(start, tz="UTC"):
            raise ValueError(f"{interval}: requested history start missing")
        interval_delta = pd.Timedelta(milliseconds=interval_to_milliseconds(interval))
        if frame.timestamp.iloc[-1] + interval_delta != end:
            raise ValueError(
                f"{interval}: newest closed candle missing; snapshot not current"
            )
        path = output_dir / f"btc_{interval}.parquet"
        frame.to_parquet(path, index=False)
        result = {
            "file": path.name,
            "rows": len(frame),
            "data_start": frame.timestamp.min().isoformat(),
            "last_bar_open": frame.timestamp.max().isoformat(),
            "closed_through_exclusive": end.isoformat(),
            "sha256": file_sha256(path),
            "validation": dict(frame.attrs),
        }
        print(f"Validated {interval}: {len(frame)} closed bars", flush=True)
        return interval, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        files = dict(pool.map(fetch, intervals))
    funding_end = closed_boundary(now, "1h")
    with session() as http:
        funding = download_funding_rates(
            symbol, start, funding_end, base_url=cfg["base_url"], session=http
        )
    funding_path = output_dir / "btc_funding_rates.parquet"
    funding.to_parquet(funding_path, index=False)
    files["funding"] = {
        "file": funding_path.name,
        "rows": len(funding),
        "sha256": file_sha256(funding_path),
        "data_start": str(funding.timestamp.min()),
        "data_end": str(funding.timestamp.max()),
        "missing_mark_price_count": int(
            (funding.mark_price.isna() | funding.mark_price.le(0)).sum()
        ),
    }
    missing_marks = funding.loc[
        funding.mark_price.isna() | funding.mark_price.le(0), "timestamp"
    ]
    files["funding"].update(
        {
            "exact_cost_history_complete": missing_marks.empty,
            "missing_mark_price_first": str(missing_marks.min())
            if not missing_marks.empty
            else None,
            "missing_mark_price_last": str(missing_marks.max())
            if not missing_marks.empty
            else None,
        }
    )
    manifest = {
        "version": "closed_market_snapshot_v1",
        "created_at_utc": now.isoformat(),
        "symbol": symbol,
        "files": files,
        "fit_operations_performed": 0,
        "evaluation_operations_performed": 0,
        "source": "Binance USDT-M REST with configured archive fallback",
        "research_use": "unselected_market_data_only_not_a_new_oos_pass",
        "futures_positioning_metrics": "not_refreshed_not_required_by_retained_control",
        "complete": True,
    }
    with (output_dir / "snapshot_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Snapshot complete: {output_dir}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    refresh_snapshot(config, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

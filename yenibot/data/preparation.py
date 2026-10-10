"""Resumable raw-data preparation; existing research artifacts are never replaced."""
from pathlib import Path
import json

import pandas as pd

from yenibot.data.binance import (
    download_full_klines, download_futures_metrics_from_vision, interval_to_milliseconds,
)
from yenibot.data.funding import download_funding_history, validate_funding_history
from yenibot.data.http import ArchiveSession
from yenibot.data.shared_store import prepare_shared_table
from yenibot.data.validation import validate_full_kline_frame
from yenibot.notebook_runtime import publish_table, verified_table


def _bounds(frame, start, end, cadence):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if frame.empty:
        raise ValueError("Empty dataset")
    times = frame["timestamp"]
    if (times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing
            or (times < start).any() or (times >= end).any()
            or times.iloc[0] - start >= cadence or end - times.iloc[-1] > cadence):
        raise ValueError("Dataset does not cover requested cutoff; retry after Binance publishes archives")


def prepare_raw_data(cfg, data_dir, *, archive_cache, shared_store=None):
    """Pin shared raw partitions when enabled; retain legacy per-workspace compatibility."""
    settings = cfg["binance"]
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    start, end, symbol = settings["start_date"], settings["end_date"], settings["symbol"]
    if not end:
        raise ValueError("Resolve an explicit data cutoff before preparing data")
    if shared_store is not None:
        print("Shared raw store:", Path(shared_store))

    def table(name, download, validate, provenance, validate_complete=None):
        path = raw / name
        exists = path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists()
        provenance = dict(provenance, symbol=symbol, start=start, end_exclusive=end)
        if exists:
            frame = verified_table(path)
            stored = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())["provenance"]
            if any(stored.get(key) != value for key, value in provenance.items()):
                raise ValueError(f"Existing dataset source contract differs: {name}")
        elif shared_store is not None:
            # Version this contract whenever normalization/source semantics change.
            contract = {"normalization_version": 1, "dataset": name,
                        "symbol": symbol, "base_url": settings["base_url"],
                        "vision_base_url": settings["vision_base_url"],
                        "data_source": settings.get("data_source", "auto"),
                        "zero_volume_policy": settings.get("zero_volume_policy", "error"),
                        "max_gap_multiplier": settings.get("max_gap_multiplier", 2),
                        "intrabar_max_gap_multiplier": settings.get("intrabar_max_gap_multiplier", 8)}
            frame = prepare_shared_table(path, shared_store, contract, start, end,
                                         download, validate, provenance, validate_complete=validate_complete)
        else:
            frame = validate(download(start, end), start, end)
            if "funding_sources" in frame.attrs:
                provenance["sources"] = frame.attrs["funding_sources"]
            publish_table(frame, path, provenance=provenance)
        status = "Verified existing:" if exists else ("Pinned shared snapshot:" if shared_store is not None else "Published:")
        print(status, name, len(frame),
              frame["timestamp"].min(), frame["timestamp"].max())

    with ArchiveSession(archive_cache) as http:
        # Resolve the previously failing source before spending time on larger downloads.
        funding = settings.get("funding_rates", {})
        if funding.get("enabled", False):
            table(funding.get("filename", "btc_funding_rates.parquet"),
                  lambda lo, hi: download_funding_history(symbol, lo, hi, session=http,
                      base_url=settings["base_url"], vision_base_url=settings["vision_base_url"]),
                  lambda frame, lo, hi: validate_funding_history(frame, symbol, lo, hi),
                  {"kind": "normalized_funding"},
                  validate_complete=lambda frame: validate_funding_history(frame, symbol, start, end))

        intervals = [settings["primary_interval"], settings["htf_interval"],
                     *settings.get("intrabar_intervals", [])]
        for interval in dict.fromkeys(intervals):
            def validate_complete(frame):
                return validate_full_kline_frame(frame, interval,
                    max_gap_multiplier=settings.get("intrabar_max_gap_multiplier", 8)
                    if interval in settings.get("intrabar_intervals", []) else settings.get("max_gap_multiplier", 2),
                    zero_volume_policy=settings.get("zero_volume_policy", "error"))

            def validate(frame, lo, hi):
                # Boundary check before zero-volume policy avoids mistaking dropped bars for unpublished data.
                _bounds(frame, lo, hi, pd.Timedelta(milliseconds=interval_to_milliseconds(interval)))
                frame = validate_complete(frame)
                print(interval, "dropped zero-volume:", frame.attrs.get("dropped_zero_volume_rows"),
                      "gaps:", frame.attrs.get("gap_count_gt_expected"), "max:", frame.attrs.get("max_gap"))
                return frame

            table(f"btc_{interval}.parquet",
                  lambda lo, hi: download_full_klines(symbol, interval, lo, hi, session=http,
                      base_url=settings["base_url"], vision_base_url=settings["vision_base_url"],
                      data_source=settings.get("data_source", "auto"), limit=settings.get("limit", 1500),
                      request_sleep_seconds=settings.get("request_sleep_seconds", .15)),
                  validate, {"kind": "normalized_klines", "interval": interval,
                             "source_policy": settings.get("data_source", "auto")},
                  validate_complete=validate_complete)

        metrics = settings.get("futures_metrics", {})
        if metrics.get("enabled", False):
            def validate_metrics(frame, lo, hi):
                _bounds(frame, lo, hi, pd.Timedelta(minutes=5))
                return frame

            table(metrics.get("filename", "btc_futures_metrics.parquet"),
                  lambda lo, hi: download_futures_metrics_from_vision(symbol, lo, hi, session=http,
                      vision_base_url=settings["vision_base_url"],
                      request_sleep_seconds=metrics.get("request_sleep_seconds", 0)),
                  validate_metrics, {"kind": "normalized_futures_metrics"})
    print("01: all configured raw datasets and manifests verified.")

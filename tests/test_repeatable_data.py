"""Offline regression coverage for automatic ingestion, source failure and resumption."""
from datetime import datetime, timezone
from io import BytesIO
import json
import zipfile

import pandas as pd
import pytest
import requests

from yenibot.data import funding, preparation
from yenibot.data.http import ArchiveSession
from yenibot.notebook_runtime import resolve_data_settings


def response(code=200, *, payload=None, content=b""):
    result = requests.Response()
    result.status_code = code
    result._content = json.dumps(payload).encode() if payload is not None else content
    return result


def rows(start="2026-09-30", periods=6):
    return [{"symbol": "BTCUSDT", "fundingTime": int(stamp.timestamp() * 1000) + 6,
             "fundingRate": "0.0001"}
            for stamp in pd.date_range(start, periods=periods, freq="8h", tz="UTC")]


class History:
    def __init__(self):
        self.rows = rows()
        self.posts = 0
        self.bad_schema = False

    def get(self, url, **kwargs):
        if "fapi/v1" in url:
            return response(451)
        if "2026-10" in url:
            return response(404)
        csv = "calc_time,funding_interval_hours,last_funding_rate\n"
        csv += "".join(f"{row['fundingTime']},8,{row['fundingRate']}\n" for row in self.rows[:3])
        data = BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("funding.csv", csv)
        return response(content=data.getvalue())

    def post(self, url, *, json, **kwargs):
        self.posts += 1
        batch = [{"symbol": row["symbol"], "calcTime": row["fundingTime"],
                  "lastFundingRate": row["fundingRate"], "markPrice": "100"}
                 for row in reversed(self.rows)] if json["page"] == 1 else []
        return response(payload={"success": not self.bad_schema, "code": "000000", "data": batch})


def test_451_uses_monthly_archive_and_unpublished_month_website():
    http = History()
    frame = funding.download_funding_history("BTCUSDT", "2026-09-30", "2026-10-02", session=http)
    assert len(frame) == 6
    assert frame.timestamp.iloc[0] == pd.Timestamp("2026-09-30T00:00:00.006Z")
    assert frame.mark_price.iloc[:3].isna().all()  # archive has no mark prices; never fabricate them
    assert http.posts == 1
    assert funding.WEBSITE_URL in frame.attrs["funding_sources"]


@pytest.mark.parametrize("fault", ["stale", "gap", "nonfinite", "symbol", "schema"])
def test_source_errors_cannot_be_reported_as_complete(fault):
    http = History()
    if fault == "stale":
        http.rows.pop()
    elif fault == "gap":
        http.rows.pop(3)
    elif fault == "nonfinite":
        http.rows[-1]["fundingRate"] = "NaN"
    elif fault == "symbol":
        http.rows[-1]["symbol"] = "ETHUSDT"
    else:
        http.bad_schema = True
    with pytest.raises(ValueError):
        funding.download_funding_history("BTCUSDT", "2026-09-30", "2026-10-02", session=http)


def test_website_repeated_pages_stop():
    http = History()
    post = http.post
    http.post = lambda url, **kwargs: post(url, json={"page": 1})
    with pytest.raises(ValueError, match="did not advance"):
        funding._website_history(http, "BTCUSDT", 0, 9999999999999)


def test_rest_success_does_not_call_website_and_rejects_stale(monkeypatch):
    frame = funding.funding_rates_to_dataframe(rows())
    monkeypatch.setattr(funding, "download_funding_rates", lambda *a, **k: frame)
    result = funding.download_funding_history("BTCUSDT", "2026-09-30", "2026-10-02", session=object())
    assert len(result) == 6
    frame.drop(frame.index[-1], inplace=True)
    with pytest.raises(ValueError, match="incomplete"):
        funding.download_funding_history("BTCUSDT", "2026-09-30", "2026-10-02", session=object())


def test_cache_reuses_success_never_caches_missing_and_detects_corruption(tmp_path, monkeypatch):
    calls = []

    def get(self, url, **kwargs):
        calls.append(url)
        return response(404 if "missing" in url else 200, content=b"archive bytes")

    monkeypatch.setattr(requests.Session, "get", get)
    url = "https://data.binance.vision/data/test.zip"
    with ArchiveSession(tmp_path) as http:
        assert http.get(url).content == http.get(url).content
        assert len(calls) == 1
        for _ in range(2):
            assert http.get(url.replace("test", "missing")).status_code == 404
        assert len(calls) == 3
        next(tmp_path.glob("*.zip")).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            http.get(url)


def test_latest_cutoff_is_frozen_and_new_date_or_environment_has_separate_identity():
    args = ("latest_complete_day", "auto", "a" * 40, {"seed": 42})
    now = datetime(2026, 10, 10, 15, tzinfo=timezone.utc)
    cutoff, name = resolve_data_settings(*args, now=now, environment={"python": "3.13.16"})
    assert cutoff == "2026-10-10T00:00:00+00:00"
    assert (cutoff, name) == resolve_data_settings(*args, now=now, environment={"python": "3.13.16"})
    assert resolve_data_settings(*args, now=now.replace(day=17), environment={})[1] != name
    assert resolve_data_settings(*args, now=now, environment={"python": "3.13.17"})[1] != name
    assert resolve_data_settings(cutoff, name, "a" * 40, {}, now=now.replace(day=17)) == (cutoff, name)
    with pytest.raises(ValueError, match="DATA_END_UTC"):
        resolve_data_settings("", "", "a" * 40, {})


def test_preparation_resumes_verified_artifacts_and_refuses_changed_contract(tmp_path, monkeypatch):
    config = {"binance": {"symbol": "BTCUSDT", "start_date": "2026-09-30",
        "end_date": "2026-10-02", "base_url": "unused", "vision_base_url": "unused",
        "primary_interval": "1h", "htf_interval": "4h", "funding_rates": {"enabled": True},
        "futures_metrics": {"enabled": True}}}
    funding_calls, kline_calls, metric_calls = [], [], []

    def download(*args, **kwargs):
        funding_calls.append(True)
        return funding.funding_rates_to_dataframe(rows())

    def klines(symbol, interval, start, end, **kwargs):
        kline_calls.append(interval)
        times = pd.date_range(start, end, inclusive="left", freq=interval, tz="UTC")
        return pd.DataFrame({"timestamp": times, "open": 100, "high": 101, "low": 99, "close": 100,
            "volume": 10, "close_time": times + pd.Timedelta(interval) - pd.Timedelta(milliseconds=1),
            "quote_volume": 1000, "num_trades": 5, "taker_buy_base_vol": 5,
            "taker_buy_quote_vol": 500, "ignore": 0})

    monkeypatch.setattr(preparation, "download_funding_history", download)
    monkeypatch.setattr(preparation, "download_full_klines", klines)

    def metrics(symbol, start, end, *, vision_base_url, request_sleep_seconds, session):
        metric_calls.append(symbol)
        return pd.DataFrame({"timestamp": pd.date_range(start, end, inclusive="left", freq="5min", tz="UTC")})

    monkeypatch.setattr(preparation, "download_futures_metrics_from_vision", metrics)
    for _ in range(2):
        preparation.prepare_raw_data(config, tmp_path / "data", archive_cache=tmp_path / "cache")
    assert len(funding_calls) == 1 and kline_calls == ["1h", "4h"]
    assert metric_calls == ["BTCUSDT"]
    config["binance"]["end_date"] = "2026-10-03"
    with pytest.raises(ValueError, match="contract differs"):
        preparation.prepare_raw_data(config, tmp_path / "data", archive_cache=tmp_path / "cache")
    config["binance"]["end_date"] = "2026-10-02"
    (tmp_path / "data/raw/btc_funding_rates.parquet.manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        preparation.prepare_raw_data(config, tmp_path / "data", archive_cache=tmp_path / "cache")


def test_bounds_rejects_unpublished_tail():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-10-01", periods=12, freq="1h", tz="UTC")})
    with pytest.raises(ValueError, match="cutoff"):
        preparation._bounds(frame, "2026-10-01", "2026-10-02", pd.Timedelta(hours=1))

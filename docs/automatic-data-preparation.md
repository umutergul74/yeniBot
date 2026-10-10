# Repeatable Colab data preparation

Use notebook `01_data_preparation.ipynb` in a Python 3.13 CPU runtime. Pin
`REPO_COMMIT` to a reviewed full commit once and save your notebook copy. Each
later visit can run that same copy from the first cell, with Drive mounted:

```python
RESEARCH_ID = "auto"
DATA_END_UTC = "latest_complete_day"
```

The cutoff resolves once to today's 00:00 UTC, exclusively: no incomplete
current-day bars. The workspace name incorporates that date and a fingerprint
of code, configuration and environment. A different date or environment gets a
different workspace; previous research data and holdout evidence remain intact.
The notebook never follows a moving code branch or upgrades its package lock.

Notebook 01 prints concrete `REPO_COMMIT`, `RESEARCH_ID`, and `DATA_END_UTC`
assignments. Copy those exact values into notebooks 02 and 03, then the required
downstream stages. Do not copy `auto` or `latest_complete_day` into downstream
notebooks: that could select a different dataset on another day. For a precise
rerun, use the printed explicit settings in notebook 01 too.

## Automatic funding sources

1. Try Binance USDT-M funding REST history.
2. If network/HTTP access fails (including Colab HTTP 451), read Binance's public
   monthly funding archives.
3. For months without an archive, fetch Binance's public website funding
   history, paginate backwards, and select only the requested date intervals.

No manual JSON upload, proxy, credentials, alternative exchange or synthetic
funding values are used. Original millisecond settlement times are preserved.
Archive mark prices are unavailable and remain missing; funding features consume
funding rates, not these mark prices. Source URLs are recorded in the normalized
dataset manifest. The website service is not a versioned public API contract:
schema changes, missing symbols, nonfinite rates, pagination failures and funding
gaps over eight hours plus one second of settlement jitter fail closed.

Binance sources:
- https://github.com/binance/binance-public-data
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History
- Public website endpoint observed 2026-10-10:
  `https://www.binance.com/bapi/futures/v1/public/future/common/get-funding-rate-history`

## Resume and freshness

Completed normalized files are reused only after their hashes and source
contracts verify. Partial or conflicting artifacts are not overwritten. A Drive
cache of successful ZIP downloads avoids fetching the historical archives every
week. Its hashes check cache integrity, not independent provider authenticity.
Missing archives (404) and REST/website responses are never cached. Only one
notebook writer should use a given workspace/cache at a time.

Transient 429/5xx and connection failures have bounded retries. Kline and metric
start/end coverage is checked before publication; funding also checks continuity.
If Binance has not published the requested final day, the notebook reports an
error and retains completed work. Retry later with the same explicit cutoff to
resume. It never silently substitutes yesterday's dataset for today's request.
External availability cannot be guaranteed, including website access from every
Colab region.

The prior manually recovered workspace remains valid and unchanged. Start a new
automatic workspace with this code; do not adopt its old artifacts into a new
code/configuration identity. Research eligibility and failed holdout gates are
unchanged by a successful ingestion run.

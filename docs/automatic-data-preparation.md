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

Normalized raw data now lives once under `MyDrive/yeniBot/raw_store/`, in immutable
Parquet partitions. Initial history is split at month boundaries; later refreshes
add only uncovered ranges. For example, extending a cutoff from October 10 to
October 17 downloads and writes only October 10–17. The downloader may read a
provider's whole monthly ZIP to obtain that range, but the ZIP cache is reused.

Each research workspace stores small `*.parquet.manifest.json` snapshots in its
`data/raw/` directory, not another full set of raw Parquets. A snapshot pins its
cutoff, partition list, hashes, row count and source/normalization contract. Later
appends never alter an earlier snapshot. `verified_table` reads either these new
snapshots or legacy Parquets; notebook 02 uses that API and manifest-aware
existence checks. New raw snapshot identities hash the canonical descriptor;
legacy and processed table identities remain byte hashes. Processed parents pin
the appropriate identity through `table_identity`.

A code commit or Python patch change still creates a separate research workspace
but reuses raw partitions when the ingestion contract is unchanged. Raw source,
symbol, cleaning or gap-policy changes use a separate store contract. Developers
must increment `normalization_version` when ingestion semantics change; it is
deliberately independent of unrelated feature/training commits. Historical source
corrections require an explicit new store contract, never replacement of pinned
bytes. Feature and label outputs remain per-workspace.

The first run with this storage version builds the shared store once using the
existing ZIP cache where available. Older research Parquets are neither imported
automatically nor deleted. They remain available to reproduce those experiments.
Seeing their old copies in Drive after upgrade is expected. Do not delete shared
partitions while any workspace references them; there is no automatic cleanup.

Completed data is reused only after hashes and source contracts verify. Partial
or conflicting artifacts stop execution rather than being overwritten. A failed
refresh retains completed partitions for resumption, and does not publish a
partial snapshot. Cache hashes check integrity, not independent provider
authenticity. Missing archives (404) and REST/website responses are never cached.
Only one notebook writer should use a given workspace/shared store/cache at a time.

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

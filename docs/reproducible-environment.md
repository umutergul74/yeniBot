# Locked environment and synthetic smoke

The target is **Linux x86_64 / CPython 3.13**. Select a Colab runtime that
actually reports Python 3.13; the notebook checks this before setup.
Different Python/platform targets fail before installation.

## What is frozen

`requirements/locks/linux-py313.txt` pins the reviewed project/development
dependency closure with SHA-256 hashes. `requirements/environment.toml` links
that lock to its source requirements with canonical UTF-8/LF hashes. Changing
requirements without regenerating the reviewed lock fails static checks.

The installer uses a NEW venv with no system-site packages, an explicit PyPI
index, hash checking and binary wheels only. It refuses existing destination
directories and report files. The existing Windows environment is not upgraded.
A pipless venv is populated through the host pip's documented `--python` option
(pip >=22.3). This avoids requiring ensurepip on Colab. Target pip is hash-pinned
in the lock alongside application dependencies. OS, drivers and Python patch
versions are not locked by this requirements file.

## Validation sequence

Use a clean checkout of the published commit and Python 3.13 on Linux:

```sh
python scripts/prepare_environment.py --directory /tmp/yenibot-new-env --report /tmp/yenibot-smoke-new.json
/tmp/yenibot-new-env/bin/python scripts/quality.py full
```

Or open `notebooks/environment_smoke.ipynb` in a CPU Colab runtime, choose the
Python 3.13 runtime, pin the published Git commit and run its cells. It only
creates an isolated environment, performs synthetic TCN/GRU forward/backward
and a verified Parquet round trip, and writes a report. It does not mount Drive,
download market data, read trained weights or evaluate holdout performance.

`--gpu` is opt-in for an already authorized GPU runtime. The lock currently
resolves PyTorch 2.14.1 and CUDA 13 runtime wheels; a successful CPU smoke does
not establish compatibility with the Colab GPU driver. Record a successful GPU
smoke before GPU training. Do not suppress a CUDA failure or reinterpret it as
a model-performance result.

The 00–05 research notebooks use Python 3.13 and the same hashed lock in their
Colab kernel. They require a session restart after package changes and verify
installed versions and pip consistency before workspace initialization. Colab's
preinstalled extra packages are not isolated; conflicts stop setup and must be
resolved before research. The smoke uses a separate isolated environment and
does not switch the research notebook kernel.
Existing frozen artifacts retain their
historical environment; this lock is for prospective research, not automatic
artifact migration.

CI uses Python 3.13, creates the same isolated environment and runs the smoke,
quick CPU suite and dependency audit. The manual full integration workflow uses
the same lock. CI evidence validates Linux CPU behavior, not Drive/GPU behavior.

## Explicit lock maintenance

Install `uv==0.13.0` in a separate tooling venv (never globally), then run:

```sh
python scripts/compile_environment.py --uv /path/to/tooling/bin/uv
python scripts/verify_environment.py --structure-only
```

Review version changes, rerun audit and the full suite in the clean target
environment, and publish through a PR. The generator never runs during normal
installation or CI. See [uv cross-platform resolution](https://docs.astral.sh/uv/concepts/resolution/)
and [pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/).

## Colab order for the Python 3.13 migration

1. Fresh CPU runtime: `environment_smoke.ipynb`, with the published full commit.
   Share the JSON report. On failure share the saved setup log; do not run later cells.
2. If GPU research is planned, repeat the smoke in a fresh GPU runtime with
   `RUN_GPU = True`. CPU evidence cannot substitute for the GPU check.
3. Once those checks pass, prospective research follows `01_data_preparation`,
   `02_feature_engineering`, `03_labeling`, then `04_training_walk_forward` and
   `05_diagnostics_validation`. Use the same commit, research ID and data cutoff.
   `00_phase1_auto_run` is an alternative to 04 with diagnostics, not another
   training step to run afterward. Existing research eligibility and frozen OOS
   gates still apply; passing environment checks does not authorize bypassing them.

The kernel installer intentionally stops after it changes packages. Choose
**Restart session**, then rerun from the first cell. Do not delete the runtime at
that point, because doing so removes the installed packages. If `pip check`
reports a conflict with Colab's extra packages, preserve that output and stop.

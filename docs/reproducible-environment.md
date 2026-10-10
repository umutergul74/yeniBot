# Locked environment and synthetic smoke

The prospective target is **Linux x86_64 / CPython 3.12**. Colab offers a
`2026.07` past runtime with Python 3.12.13, documented in its
[runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html).
This does not assert that an arbitrary currently open Colab session uses that
runtime. Different Python/platform targets fail before installation.

## What is frozen

`requirements/locks/linux-py312.txt` pins the 65-package project/development
dependency closure with SHA-256 hashes. `requirements/environment.toml` links
that lock to its source requirements with canonical UTF-8/LF hashes. Changing
requirements without regenerating the reviewed lock fails static checks.

The installer uses a NEW venv with no system-site packages, an explicit PyPI
index, hash checking and binary wheels only. It refuses existing destination
directories and report files. The existing Windows environment is not upgraded.
Pip bootstrapping comes from the target Python's venv/ensurepip; it is not part
of the locked application dependency closure. OS, drivers and Python patch
versions are not locked by this requirements file.

## Validation sequence

Use a clean checkout of the published commit and Python 3.12 on Linux:

```sh
python scripts/prepare_environment.py --directory /tmp/yenibot-new-env --report /tmp/yenibot-smoke-new.json
/tmp/yenibot-new-env/bin/python scripts/quality.py full
```

Or open `notebooks/environment_smoke.ipynb` in a CPU Colab runtime, choose the
Python 3.12 runtime, pin the published Git commit and run its cells. It only
creates an isolated environment, performs synthetic TCN/GRU forward/backward
and a verified Parquet round trip, and writes a report. It does not mount Drive,
download market data, read trained weights or evaluate holdout performance.

`--gpu` is opt-in for an already authorized GPU runtime. The lock currently
resolves PyTorch 2.14.1 and CUDA 13 runtime wheels; a successful CPU smoke does
not establish compatibility with the Colab GPU driver. Record a successful GPU
smoke before GPU training. Do not suppress a CUDA failure or reinterpret it as
a model-performance result.

The original 00–05 research notebooks still use their existing installation
path. They are **not yet migrated to execute inside this isolated environment**.
Do not assume running the smoke switches their kernel or locks those notebooks.
The next integration step is to run their orchestration with the isolated
interpreter after real Colab validation. Existing frozen artifacts retain their
historical environment; this lock is for prospective research, not automatic
artifact migration.

CI uses Python 3.12, creates the same isolated environment and runs the smoke,
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

# Main research hardening — 2026-10-09

## Scope and Git state

This is prospective hardening of the original Phase 1 pipeline, based on main
`7badc52f416c64bcbdb09174eb911e015c4dcd95`, on local branch
`codex/main-research-hardening`. It does not contain the separate advisor protocol.
No real training, download, holdout evaluation or deployment was run. Commit,
branch publication and a PR were subsequently authorized by the user; merging
main remains separate. Git identity tests use disposable synthetic repositories.

After fetching origin, local main and origin/main had zero commits of divergence;
the advisor branch and its remote also had zero divergence. The original checkout
was on the advisor branch and contained uncommitted changes. Matching commit tips
does not mean those working files are backed up on GitHub. This work was prepared
in a separate managed worktree for publication on the hardening branch. For Colab,
use its verified published commit; these changes do not automatically update main.

Fetch initially failed because five Google Drive/Windows `desktop.ini` files were
under `.git/refs`. Those exact files were moved to
`reports/engineering/git-ref-metadata-backup-20261009` in the original checkout;
fetch subsequently succeeded. Do not synchronize Git internals through Drive.

## Correctness changes

- All `fwd_return_*` targets and `forward_return` are excluded from automatic
  features and rejected in explicitly supplied model/HMM inputs.
- Missing configured return horizons fail before fitting; a 24-bar experiment
  cannot silently consume the 10-bar target. Notebook label-quality checks and
  plots use the configured horizon.
- Walk-forward windows and steps require positive integers; gaps require
  nonnegative integers. Zero step can no longer cause an unbounded split loop.
- Duplicate YAML keys and nonfinite raw OHLC/volume data fail explicitly.
- Training cache signature v3 covers HMM inputs and the consumed target horizon.
  Existing v2 caches are not accepted as v3 evidence; no old artifacts were changed.

Hypothesis: changing a target horizon or HMM input could previously change training
without a trustworthy input boundary or cache identity. Falsifiers are synthetic
tests showing target exclusion, rejection of a missing horizon, and signature
changes when consumed inputs change. Tests are engineering evidence, not evidence
of predictive performance or satisfaction of Phase 1 promotion gates.

## Prospective Colab workflow

Use notebooks 01 → 02 → 03 → 04 → 05 for new research; 00 is a combined
training/diagnostics alternative after data preparation, not a downloader.

1. Review and publish the code through the normal authorized Git process first.
   Pin the full 40-character published commit in `REPO_COMMIT` in every notebook.
2. Set the same `RESEARCH_ID` and explicit past timezone-aware `DATA_END_UTC` in
   each notebook. Use a fresh runtime for setup. Do not install over imported code.
3. Setup checks subprocess failures, pip consistency, clean Git identity, config
   identity and installed Python/package inventory. It creates
   `MyDrive/yeniBot/research/<RESEARCH_ID>/workspace.json` and isolated directories.
4. Normalized raw/feature/labeled Parquet files have SHA-256 sidecars and parent
   hashes. Reads validate the stored bytes, rows and column order. Different output
   cannot overwrite an existing dataset. Interrupted publication without a valid
   sidecar stops; investigate it explicitly rather than adopting it automatically.
5. Configured funding/futures/intrabar failures stop the workflow. Runtime release
   defaults to off and failures keep their logs available.
6. In notebook 05, set `EXPERIMENT_RUN_ID` to the exact ID returned by 00 or 04.
   Diagnostics no longer selects whichever experiment happens to be newest.

One writer per workspace is required. Drive is not a transactional artifact store;
exclusive creation is not a distributed lock guarantee. Sidecar provenance states
the requested source policy, not the exact provider object/revision or signed origin.
Hash integrity does not establish trust in pickle/joblib/model files.

The full installed package inventory is intentionally strict. A different Colab
image or CPU/GPU package set may reject an existing workspace. Restore the recorded
environment or start a new research identity and rebuild; never edit the manifest
to bypass the mismatch. This detects drift but is **not** a portable environment
lock or a GPU-determinism guarantee.

These notebooks deliberately do not automatically adopt historical shared Drive
data or checkpoints. Earliest required rerun for prospective research is notebook
01 in a new workspace. Historical frozen OOS must use its original verified code,
data and artifacts; never rerun training to repair missing frozen evidence.

## Verification

Use the existing project interpreter:

```powershell
python scripts/quality.py static
python scripts/quality.py quick
python scripts/quality.py full --report reports/engineering/new-full-result.json
```

The runner caps elapsed test time and disables CUDA by default. Full tests include
synthetic experiment/training integration. Notebook contract tests compile all six
notebooks, verify deterministic regeneration and execute selected data/label cells
with synthetic inputs. They do not claim a live Colab/GPU integration pass.

Local evidence for this change:

- Static Ruff and repository checks: passed (zero repository errors).
- Research-boundary, notebook-runtime and input-integrity regression set:
  117 passed in 15.47 seconds.
- Notebook integration contracts, including failed clone, configured 24-bar label
  validation, explicit diagnostics run selection and preserving failure logs:
  10 passed in 16.37 seconds.
- Full CPU suite: 283 passed in 894.20 seconds; runner test subprocess 899.87
  seconds, exit 0. Evidence: `reports/engineering/main-hardening-full.json`.
  That run collected before the ten notebook integration contracts were added;
  those ten were run separately against the final notebook edits. This is not a
  claim that the final combined 293-test collection was rerun end to end.

The notebook setup templates live in `scripts/harden_phase1_notebooks.py`; rerun
that script after changing its templates and inspect the resulting notebook diff.

## Remaining requirements before a new parameter campaign

Update 2026-10-10: the prospective Linux/Python 3.12 hash lock, isolated installer
and synthetic smoke are described in [reproducible environment](reproducible-environment.md).
The [read-only evidence inventory](research-readiness-2026-10-10.md) separates
main's frozen candidate from the advisor's already-seen September test. The
original 00–05 notebooks have not yet been migrated to the isolated interpreter.

- Build and test a clean, isolated environment with reviewed, hash-locked
  dependencies for the target Colab/Linux runtime. Current requirements contain
  ranges. The earlier local dependency audit reported advisories; this work did
  not upgrade or certify that environment. CI now audits the installed project
  dependency closure; consult the PR checks for remote CI results after publication.
- Run a small authorized Colab integration check from the published commit,
  including restart/resume and Drive publication. No paid GPU job was started here.
- Pre-register parameter ranges, trial budget, seeds, selection metric and eligible
  time intervals. Seen final tests/holdouts cannot become a fresh selection set.
  Repeated selection on CV test metrics makes those metrics development evidence;
  only untouched prospective evidence can support the final claim.
- Keep existing Phase 1 gates and frozen reservations unchanged. Professional
  engineering makes comparisons trustworthy; it does not guarantee improved returns.

Primary references: [Colab runtime limits](https://research.google.com/colaboratory/faq.html),
[pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/),
[scikit-learn evaluation and model selection](https://scikit-learn.org/stable/modules/cross_validation.html).

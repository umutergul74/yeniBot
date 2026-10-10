# Colab training: current experiment and execution order

The current Phase 1 configuration reproduces a historical control. It is not a
new parameter search: `candidate_profiles` is empty. The control is
`baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility`.
The separately printed active feature profile is not evidence of an additional
candidate. The matrix configuration selects the control.

Training uses TCN+GRU with train-only scaling/HMM, 5040 training bars, 1080
validation bars, 720 test bars, 720-bar steps, purge 24 and embargo 6. Maximum
epochs are 100, early stopping monitors rank IC with patience 15. Seed audit uses
42, 43, 44 on the configured eight folds. These are historical settings, not a
recommendation to tune them against previously seen holdout results.

The reported 40401 labeled rows split into 32500 selection rows, 4320 reserved
holdout rows and 3581 post-anchor rows. The anchor remains fixed at May 13, 2026
08:00 UTC; recent data does not automatically enter training. `Future OOS ready`
indicates window availability, not artifact readiness or passing performance.
Policy status remains `failed_clean_holdout_review`: the old top-10 policy is a
benchmark only. Do not promote it or run notebook 05 without reviewing frozen
artifact availability and one-shot evaluation eligibility.

## Notebook sequence

For a new experiment use one reviewed full commit across 01–05. Run 01 (download),
02 (features), 03 (labels), then 04 (training). Copy the exact resolved research ID
and cutoff from 01 into all later notebooks. New code requires a new workspace;
never rewrite the old manifest. Existing raw-store partitions can be reused by
01; features and labels remain workspace-specific.

Every notebook has the same first five code cells:

1. Settings: starts `if "RESEARCH" in globals():`. Set commit, research ID, cutoff,
   Drive base and GPU requirement. Research Python defaults to exact 3.13.16.
2. Drive: starts `from google.colab import drive`.
3. Checkout: starts `import re` and checks out the pinned commit.
4. Environment: starts `_ENVIRONMENT_READY = False`. It creates or verifies an
   isolated version-specific environment, installs hash-locked packages, checks
   the GPU when requested, and starts the research kernel. A different Colab host
   patch version is handled automatically using a separate Python installation.
5. Workspace: starts `if not globals().get("_ENVIRONMENT_READY", False):`.
   This must succeed before domain work.

In 04, code cell 6 starts training; cell 7 finishes the session. Read the experiment
configuration before running cell 6. Stop at the first error. Rerunning cell 4
closes the previous kernel and resets its variables; rerun cell 5 afterward.
No manual diagnosis or temporary Python installer cells belong between them.

For a GPU account, share the same Drive folder with editor permission and add a
My Drive shortcut named yeniBot. Do not run two writers on the same workspace.
GPU hardware can differ, but interpreter/package identity checks remain strict.
The bootstrap uses uv in an isolated tooling environment; network/download
availability is still required and errors are retained in setup logs.

Existing experiments pinned to 477a0638 must continue with their original code.
They may use the previously prepared 3.13.16 environment. Do not point that
workspace at a new commit merely to obtain this notebook setup improvement.

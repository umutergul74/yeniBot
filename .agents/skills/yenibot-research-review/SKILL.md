---
name: yenibot-research-review
description: Review yeniBot research changes for temporal leakage, frozen artifact identity and valid evaluation; use for feature, label, training, holdout or metric changes.
---

Identify whether the change targets the original Phase 1 pipeline or the advisor
protocol. Read only the relevant sections of `SKILLS.md`, `config.yaml` or
`configs/advisor_*.yaml`; advisor final-test changes also require
`docs/advisor-final-test-protocol.md`. Do not transfer a decision between protocols.

Trace each changed input's source timestamp, availability time, fit scope and
label maturity. Scalers/HMM fit on train; early stopping, thresholds and optional
probability calibration use validation; test and holdout remain evaluation-only.
Check purge/embargo against label horizon, completed MTF bars, duplicate/nonfinite
inputs, and chronological splits. Scores are not calibrated probabilities or PnL.

For reused predictions, inspect training signatures for every consumed input,
feature order and preprocessing setting. Frozen OOS must verify hashes before
deserialization and perform no fitting. Never regenerate missing frozen evidence
as a repair. Hashes do not authorize loading artifacts from an untrusted source.

Use synthetic data and temporary output paths to reproduce the issue. Run affected
tests first; `python scripts/quality.py full` covers broad orchestration changes.
Do not run real training or a one-shot final evaluation just to verify code.

Deliver: hypothesis and falsifier, changed contracts, earliest required notebook
rerun, exact test evidence, and remaining uncertainty. Research promotion requires
the active pre-registered gates, not a favorable retrospective result.

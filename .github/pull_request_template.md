## Summary

Describe the change and the concrete failure mode or research hypothesis.

## Validation

- [ ] `pytest -q` passes
- [ ] No data, checkpoints, secrets, runtime state, or report bundles are committed
- [ ] Configuration changes are in `config.yaml`
- [ ] Temporal availability and leakage assumptions are documented
- [ ] Train-only fitting boundaries are preserved
- [ ] Existing rejected experiments were reviewed before adding a candidate

## Notebook Impact

Required rerun:

- [ ] `01` data preparation
- [ ] `02` feature engineering
- [ ] `03` labeling
- [ ] `04` training
- [ ] `05` diagnostics only
- [ ] No notebook rerun

## Frozen OOS Impact

- [ ] No effect on frozen model artifacts or training signatures
- [ ] Creates a new candidate and requires a new pre-registered OOS window
- [ ] Diagnostics/report-only change

Explain:

## Evidence

List the metrics and artifacts used to accept or reject the change. Do not use
the already-seen holdout for tuning.

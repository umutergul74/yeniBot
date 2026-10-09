# yeniBot engineering rules

Python 3.11+ quantitative ML research, not a live trading service. The tracked
branch is based on main and contains the original Phase 1 pipeline. The advisor
protocol lives on a separate branch; do not mix their artifacts or decisions.
Use `docs/architecture.md` for ownership and `docs/main-hardening.md` for workflow.

- Start with `git status --short` and targeted `rg` in `yenibot`, `tests`,
  `configs`, or `scripts`. Do not recursively read output/data/checkpoints,
  notebooks, archives, or copied research repositories without a task need.
- Preserve pre-existing edits and untracked artifacts.
- Standing user authorization (2026-10-10): perform task-related commits, pushes,
  PR creation/updates and merges without asking again. Merge only after relevant
  checks pass for the current PR head and blocking review findings are resolved;
  then synchronize the local main safely. Never include unrelated local changes.
  This authorization does not cover force pushes or discarding existing work.
- Do not run migrations, deploy, trade, or start paid services without authorization.
- Keep credentials out of code, logs and prompts. Model/joblib/pickle artifacts
  are trusted-code inputs; a checksum proves identity, not trustworthiness.
- Preserve Phase 1 gates and one-shot holdout reservations. Do not build or
  activate Phase 2 execution while research eligibility remains unverified.
- Keep notebook orchestration thin. New experiment logic belongs in
  `yenibot/experiment/`; `yenibot/experiments.py` stays a compatibility facade.
  Preserve report schemas and frozen-artifact identities.
- For research changes use `.agents/skills/yenibot-research-review/SKILL.md`.
  Read only relevant sections of `SKILLS.md` and the active protocol/runbook.
  Never use a seen final test/holdout to select a new policy.
- Use the existing Python environment; do not upgrade global packages. Run
  `python scripts/quality.py static`, then affected tests. Use `quick` for the
  normal CPU suite and `full` for experiment/training-wide changes.
- Add regression tests for correctness changes. Do not hide failed checks or
  change research thresholds to make tests pass. See `docs/main-hardening.md`.
- Default to one agent. For independent audits the user authorizes at most two
  bounded read-only reviewers; no recursive delegation or concurrent editing
  of the same files. Parent verifies findings and owns integration.
- Keep task handoffs compact: changed paths, decisions, exact check results,
  remaining risks. No claimed token savings without measured usage.

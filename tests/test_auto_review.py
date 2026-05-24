from __future__ import annotations

import json

import pandas as pd

from yenibot.automation import review_experiment_report, write_auto_review


def _write_minimal_report(path, *, missing_selected: bool = False, future_oos_ready: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    control = "control_profile"
    challenger = "candidate_profile"
    pd.DataFrame(
        [
            {
                "profile": control,
                "fold_scope": "full",
                "mean_rank_ic": 0.05,
                "std_rank_ic": 0.07,
                "positive_ic_fraction": 0.80,
                "test_f1_at_constrained_threshold": 0.46,
                "top_10_lift_global": 1.10,
            },
            {
                "profile": challenger,
                "fold_scope": "full",
                "mean_rank_ic": 0.052,
                "std_rank_ic": 0.09,
                "positive_ic_fraction": 0.70,
                "test_f1_at_constrained_threshold": 0.44,
                "top_10_lift_global": 1.14,
            },
        ]
    ).to_csv(path / "profile_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile": "blend_prob_mean",
                "fold_scope": "blend_full",
                "mean_rank_ic": 0.055,
                "std_rank_ic": 0.075,
                "positive_ic_fraction": 0.78,
                "top_10_lift_global": 1.15,
            }
        ]
    ).to_csv(path / "profile_blend.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile": challenger,
                "fold_scope": "holdout_profile",
                "mean_rank_ic": 0.06,
                "top_10_lift_global": 1.12,
                "top_10_forward_return_global": 0.002,
                "holdout_signal_pass": True,
            }
        ]
    ).to_csv(path / "holdout_evaluation.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": challenger,
                "candidate_type": "profile",
                "future_oos_priority_score": 0.7,
                "cv_mean_rank_ic": 0.052,
                "holdout_mean_rank_ic": 0.06,
            }
        ]
    ).to_csv(path / "future_oos_candidate_plan.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "failed_clean_holdout_review",
                "action": "wait_for_new_unseen_bars_keep_control_profile",
                "future_oos_ready": future_oos_ready,
                "future_oos_preferred_ready": False,
                "new_bars_since_anchor": 250,
                "min_new_bars_remaining": 470,
                "preferred_new_bars_remaining": 1900,
                "min_ready_at": "2026-06-12 08:00:00+00:00",
                "preferred_ready_at": "2026-08-11 08:00:00+00:00",
                "holdout_roll_forward_locked": True,
            }
        ]
    ).to_csv(path / "experiment_policy_guard.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected": True,
                "profile": control,
                "fold_scope": "full",
            }
        ]
    ).to_csv(path / "experiment_selection.csv", index=False)
    missing = pd.DataFrame(
        [{"profile": "missing_profile", "fold_scope": "full"}]
        if missing_selected
        else [],
        columns=["profile", "fold_scope"],
    )
    missing.to_csv(path / "missing_selected_profiles.csv", index=False)
    (path / "decision_report.json").write_text(
        json.dumps(
            {
                "run_id": "test_run",
                "control_profile": control,
                "holdout_boundary_passed": True,
                "recommendation": "keep_control_profile",
            }
        ),
        encoding="utf-8",
    )
    (path / "training_execution_summary.json").write_text(
        json.dumps(
            {
                "training_executed_count": 2,
                "training_skipped_count": 0,
                "all_training_scopes_reused": False,
            }
        ),
        encoding="utf-8",
    )


def test_auto_review_waits_for_future_oos_when_no_cv_candidate(tmp_path) -> None:
    _write_minimal_report(tmp_path)

    review = review_experiment_report(tmp_path)

    assert review["report_completeness"]["complete"] is True
    assert review["next_action"]["action"] == "wait_for_new_unseen_bars_keep_control"
    assert review["next_action"]["do_not_promote_from_current_holdout"] is True
    assert review["cv"]["control"]["profile"] == "control_profile"


def test_auto_review_flags_missing_selected_profiles(tmp_path) -> None:
    _write_minimal_report(tmp_path, missing_selected=True)

    review = review_experiment_report(tmp_path)

    assert review["report_completeness"]["complete"] is False
    assert review["next_action"]["action"] == "fix_missing_selected_profiles"


def test_write_auto_review_outputs_files(tmp_path) -> None:
    _write_minimal_report(tmp_path)

    result = write_auto_review(tmp_path)

    assert (tmp_path / "auto_review.md").exists()
    assert (tmp_path / "auto_review.json").exists()
    assert (tmp_path / "next_actions.json").exists()
    next_actions = json.loads((tmp_path / "next_actions.json").read_text(encoding="utf-8"))
    assert next_actions["action"] == "wait_for_new_unseen_bars_keep_control"
    assert result["auto_review_path"].endswith("auto_review.md")

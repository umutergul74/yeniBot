"""Holdout boundary guard helpers."""

from __future__ import annotations

import pandas as pd


def holdout_boundary_passed(audit: pd.DataFrame) -> bool:
    """Return whether any blocking training scope reached the reserved holdout."""

    if audit.empty:
        return True
    if "blocking" in audit.columns:
        blocking = audit["blocking"].map(
            lambda value: bool(value)
            if isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
        passed = audit["passed"].map(
            lambda value: bool(value)
            if isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
        return not bool((blocking & ~passed).any())
    return bool(audit["passed"].astype(bool).all())

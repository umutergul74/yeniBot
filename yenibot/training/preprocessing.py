from __future__ import annotations

import json
from fnmatch import fnmatch
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def _cfg(config: Any, path: list[str], default: Any = None) -> Any:
    current = config
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def _matching_columns(columns: list[str], settings: dict[str, Any]) -> list[str]:
    exact = {str(column) for column in settings.get("columns", []) or []}
    patterns = [str(pattern) for pattern in settings.get("patterns", []) or []]
    return [
        column
        for column in columns
        if column in exact or any(fnmatch(column, pattern) for pattern in patterns)
    ]


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average")))


def _standardized_label_gap(feature: pd.Series, labels: pd.Series) -> float:
    pair = pd.concat([feature, labels], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    scale = float(pair.iloc[:, 0].std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return float("nan")
    means = pair.groupby(pair.columns[1], observed=True)[pair.columns[0]].mean()
    if 0 not in means.index or 1 not in means.index:
        return float("nan")
    return float((means.loc[1] - means.loc[0]) / scale)


def _json_float_list(values: list[float]) -> str:
    return json.dumps([float(value) if np.isfinite(value) else None for value in values])


class CausalFoldPreprocessor:
    """Train-fold-only clipping, reliability masking, and robust scaling."""

    AUDIT_COLUMNS = [
        "feature",
        "clip_enabled",
        "clip_lower",
        "clip_upper",
        "train_clip_fraction",
        "stability_checked",
        "block_count",
        "prior_rank_ic",
        "recent_rank_ic",
        "prior_label_gap",
        "recent_label_gap",
        "rank_ic_sign_agreement",
        "rank_ic_reversal",
        "label_gap_reversal",
        "masked",
        "mask_reason",
        "block_rank_ics",
        "block_label_gaps",
    ]

    def __init__(self, feature_columns: list[str], config: Any):
        self.feature_columns = list(feature_columns)
        self.settings = dict(_cfg(config, ["training", "preprocessing"], {}) or {})
        self.scaler = RobustScaler()
        self.clip_bounds: dict[str, tuple[float, float]] = {}
        self.masked_columns: list[str] = []
        self.audit_records: list[dict[str, Any]] = []
        self._fitted = False

    def _clip_settings(self) -> dict[str, Any]:
        return dict(self.settings.get("quantile_clip", {}) or {})

    def _stability_settings(self) -> dict[str, Any]:
        return dict(self.settings.get("stability_mask", {}) or {})

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Preprocessor frame is missing feature columns: {missing}")
        selected = frame.loc[:, self.feature_columns].astype(float)
        if not np.isfinite(selected.to_numpy()).all():
            raise ValueError("Preprocessor frame contains non-finite feature values")
        return selected

    def _fit_clip_bounds(self, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        settings = self._clip_settings()
        if not bool(settings.get("enabled", False)):
            return {}
        lower_q = float(settings.get("lower_quantile", 0.01))
        upper_q = float(settings.get("upper_quantile", 0.99))
        if not 0.0 <= lower_q < upper_q <= 1.0:
            raise ValueError("Quantile clipping requires 0 <= lower < upper <= 1")

        audit: dict[str, dict[str, Any]] = {}
        for column in _matching_columns(self.feature_columns, settings):
            lower = float(frame[column].quantile(lower_q))
            upper = float(frame[column].quantile(upper_q))
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError(f"Invalid train-only clip bounds for {column}")
            self.clip_bounds[column] = (lower, upper)
            outside = (frame[column] < lower) | (frame[column] > upper)
            audit[column] = {
                "clip_enabled": True,
                "clip_lower": lower,
                "clip_upper": upper,
                "train_clip_fraction": float(outside.mean()),
            }
        return audit

    def _feature_stability(
        self,
        feature: pd.Series,
        forward_returns: pd.Series,
        labels: pd.Series,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        block_rows = int(settings.get("block_rows", 720))
        min_blocks = int(settings.get("min_blocks", 5))
        recent_blocks = int(settings.get("recent_blocks", 2))
        min_abs_ic = float(settings.get("min_abs_rank_ic", 0.01))
        min_abs_gap = float(settings.get("min_abs_label_gap", 0.05))
        min_sign_agreement = float(settings.get("min_rank_ic_sign_agreement", 0.60))
        if block_rows <= 0 or recent_blocks <= 0:
            raise ValueError("Stability masking requires positive block_rows and recent_blocks")

        rank_ics: list[float] = []
        label_gaps: list[float] = []
        for start in range(0, len(feature), block_rows):
            stop = min(start + block_rows, len(feature))
            if stop - start < max(24, block_rows // 2):
                continue
            block = slice(start, stop)
            rank_ics.append(_rank_correlation(feature.iloc[block], forward_returns.iloc[block]))
            label_gaps.append(_standardized_label_gap(feature.iloc[block], labels.iloc[block]))

        finite_ics = np.asarray([value for value in rank_ics if np.isfinite(value)], dtype=float)
        finite_gaps = np.asarray([value for value in label_gaps if np.isfinite(value)], dtype=float)
        enough = len(rank_ics) >= min_blocks and len(finite_ics) >= min_blocks and len(finite_gaps) >= min_blocks
        if not enough or len(rank_ics) <= recent_blocks:
            return {
                "stability_checked": True,
                "block_count": len(rank_ics),
                "masked": False,
                "mask_reason": "insufficient_train_blocks",
                "block_rank_ics": _json_float_list(rank_ics),
                "block_label_gaps": _json_float_list(label_gaps),
            }

        prior_ics = np.asarray(rank_ics[:-recent_blocks], dtype=float)
        recent_ics = np.asarray(rank_ics[-recent_blocks:], dtype=float)
        prior_gaps = np.asarray(label_gaps[:-recent_blocks], dtype=float)
        recent_gaps = np.asarray(label_gaps[-recent_blocks:], dtype=float)
        prior_ic = float(np.nanmedian(prior_ics))
        recent_ic = float(np.nanmedian(recent_ics))
        prior_gap = float(np.nanmedian(prior_gaps))
        recent_gap = float(np.nanmedian(recent_gaps))
        meaningful = finite_ics[np.abs(finite_ics) >= min_abs_ic]
        sign_agreement = (
            float(max(np.mean(meaningful > 0), np.mean(meaningful < 0)))
            if meaningful.size
            else 0.0
        )
        ic_reversal = (
            prior_ic * recent_ic < 0
            and abs(prior_ic) >= min_abs_ic
            and abs(recent_ic) >= min_abs_ic
        )
        gap_reversal = (
            prior_gap * recent_gap < 0
            and abs(prior_gap) >= min_abs_gap
            and abs(recent_gap) >= min_abs_gap
        )
        low_agreement_with_recent_signal = (
            sign_agreement < min_sign_agreement
            and abs(recent_ic) >= min_abs_ic
            and abs(recent_gap) >= min_abs_gap
        )
        masked = bool((ic_reversal and gap_reversal) or low_agreement_with_recent_signal)
        if ic_reversal and gap_reversal:
            reason = "recent_return_and_label_reversal"
        elif low_agreement_with_recent_signal:
            reason = "unstable_block_signs_with_recent_signal"
        else:
            reason = "stable_enough"
        return {
            "stability_checked": True,
            "block_count": len(rank_ics),
            "prior_rank_ic": prior_ic,
            "recent_rank_ic": recent_ic,
            "prior_label_gap": prior_gap,
            "recent_label_gap": recent_gap,
            "rank_ic_sign_agreement": sign_agreement,
            "rank_ic_reversal": bool(ic_reversal),
            "label_gap_reversal": bool(gap_reversal),
            "masked": masked,
            "mask_reason": reason,
            "block_rank_ics": _json_float_list(rank_ics),
            "block_label_gaps": _json_float_list(label_gaps),
        }

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        forward_returns: pd.Series | None = None,
        labels: pd.Series | None = None,
    ) -> "CausalFoldPreprocessor":
        selected = self._validate_frame(frame)
        audit_by_feature = self._fit_clip_bounds(selected)
        stability = self._stability_settings()
        if bool(stability.get("enabled", False)):
            if forward_returns is None or labels is None:
                raise ValueError("Stability masking requires train-fold returns and labels")
            returns = pd.Series(forward_returns).reset_index(drop=True).astype(float)
            binary_labels = pd.Series(labels).reset_index(drop=True).astype(int)
            if len(returns) != len(selected) or len(binary_labels) != len(selected):
                raise ValueError("Stability targets must have the same length as train features")
            for column in _matching_columns(self.feature_columns, stability):
                record = self._feature_stability(
                    selected[column].reset_index(drop=True),
                    returns,
                    binary_labels,
                    stability,
                )
                audit_by_feature.setdefault(column, {}).update(record)
                if bool(record.get("masked", False)):
                    self.masked_columns.append(column)

        clipped = selected.copy()
        for column, (lower, upper) in self.clip_bounds.items():
            clipped[column] = clipped[column].clip(lower=lower, upper=upper)
        self.scaler.fit(clipped)
        self.audit_records = []
        for column in sorted(audit_by_feature):
            record = {key: None for key in self.AUDIT_COLUMNS}
            record.update({"feature": column, **audit_by_feature[column]})
            record["clip_enabled"] = bool(record.get("clip_enabled", False))
            record["stability_checked"] = bool(record.get("stability_checked", False))
            record["masked"] = bool(record.get("masked", False))
            self.audit_records.append(record)
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError("CausalFoldPreprocessor must be fitted before transform")
        selected = self._validate_frame(frame).copy()
        for column, (lower, upper) in self.clip_bounds.items():
            selected[column] = selected[column].clip(lower=lower, upper=upper)
        transformed = self.scaler.transform(selected)
        for column in self.masked_columns:
            transformed[:, self.feature_columns.index(column)] = 0.0
        return transformed

    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.audit_records, columns=self.AUDIT_COLUMNS)

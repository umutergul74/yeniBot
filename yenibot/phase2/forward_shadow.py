"""Hash-sealed, label-free inference primitives for forward-shadow v2."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from yenibot.experiment.configuration import profile_config
from yenibot.phase2.full_oof import file_sha256
from yenibot.phase2.net_utility import predict_ridge_payoff, utility_to_score
from yenibot.training.trainer import _add_regime_probs, _build_model, _device


SHADOW_PROCESS_ID = "block_prequential_forward_shadow_v2"
SHADOW_MANIFEST_VERSION = "forward_shadow_block_manifest_v2"
SHADOW_LEDGER_VERSION = "forward_shadow_ledger_v2"


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Timestamp is missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_shadow_manifest_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_hash", None)
    canonical.pop("integrity_audit", None)
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def seal_shadow_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = json.loads(json.dumps(payload))
    if sealed.get("manifest_hash"):
        raise ValueError("Unsealed manifest input cannot already contain a hash")
    sealed["manifest_hash"] = canonical_shadow_manifest_hash(sealed)
    return sealed


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Artifact escapes its manifest root: {relative!r}") from exc
    return candidate


def validate_shadow_manifest(
    payload: dict[str, Any], *, artifact_root: str | Path | None = None
) -> dict[str, Any]:
    if payload.get("manifest_version") != SHADOW_MANIFEST_VERSION:
        raise ValueError("Unsupported forward-shadow manifest version")
    if payload.get("process_id") != SHADOW_PROCESS_ID:
        raise ValueError("Forward-shadow process identity mismatch")
    configured = str(payload.get("manifest_hash", "")).lower()
    computed = canonical_shadow_manifest_hash(payload)
    if configured != computed:
        raise ValueError("Forward-shadow manifest hash mismatch")
    block = payload.get("block", {}) or {}
    required_block = (
        "block_id",
        "ordinal",
        "locked_at_utc",
        "evidence_start_inclusive",
        "evidence_end_inclusive",
    )
    if any(key not in block for key in required_block):
        raise ValueError("Forward-shadow block identity is incomplete")
    locked = _utc(block["locked_at_utc"])
    start = _utc(block["evidence_start_inclusive"])
    end = _utc(block["evidence_end_inclusive"])
    if start <= locked or end < start:
        raise ValueError("Evidence must begin strictly after the manifest lock")
    if int(block["ordinal"]) < 0:
        raise ValueError("Block ordinal must be non-negative")
    model = payload.get("model", {}) or {}
    features = list(model.get("feature_columns", []) or [])
    if not features or len(features) != len(set(features)):
        raise ValueError("Manifest needs unique model feature columns")
    expected_feature_hash = hashlib.sha256(
        _canonical_json(features)
    ).hexdigest()
    if str(model.get("feature_columns_sha256", "")) != expected_feature_hash:
        raise ValueError("Feature-column hash mismatch")
    _validate_ridge_fit((payload.get("payoff_layer") or {}).get("candidate_fit"))
    _validate_ridge_fit((payload.get("payoff_layer") or {}).get("atr_only_fit"))
    artifacts = payload.get("artifacts", {}) or {}
    required_artifacts = {"model", "scaler", "hmm", "validation_cdf"}
    if set(artifacts) != required_artifacts:
        raise ValueError("Manifest artifact set is incomplete or unexpected")
    audit: dict[str, Any] = {
        "manifest_hash_verified": True,
        "artifact_hashes_verified": artifact_root is not None,
        "verified_artifact_count": 0,
    }
    if artifact_root is not None:
        root = Path(artifact_root)
        for name, record in artifacts.items():
            relative = str((record or {}).get("path", ""))
            expected = str((record or {}).get("sha256", "")).lower()
            if not relative or len(expected) != 64:
                raise ValueError(f"Invalid {name} artifact identity")
            path = _safe_artifact_path(root, relative)
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"Changed or missing forward-shadow artifact: {name}")
            audit["verified_artifact_count"] += 1
    return {**payload, "integrity_audit": audit}


def load_shadow_manifest(
    path: str | Path, *, artifact_root: str | Path | None = None
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_shadow_manifest(
        payload,
        artifact_root=(artifact_root if artifact_root is not None else manifest_path.parent),
    )


def empirical_cdf_right(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if (
        reference.ndim != 1
        or len(reference) < 1000
        or scores.ndim != 1
        or not np.isfinite(reference).all()
        or not np.isfinite(scores).all()
        or ((reference < 0) | (reference > 1)).any()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("CDF needs >=1000 finite probability scores in [0, 1]")
    return np.searchsorted(np.sort(reference), scores, side="right") / len(reference)


def _validate_ridge_fit(fit: Any) -> dict[str, Any]:
    if not isinstance(fit, dict):
        raise ValueError("Missing frozen ridge fit")
    vectors = []
    for key in ("center", "scale", "coefficients"):
        values = np.asarray(fit.get(key), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid ridge {key}")
        vectors.append(values)
    if (vectors[1] <= 0).any():
        raise ValueError("Ridge scales must be positive")
    for key in ("alpha", "intercept"):
        if not np.isfinite(float(fit.get(key, np.nan))):
            raise ValueError(f"Invalid ridge {key}")
    if float(fit["alpha"]) <= 0:
        raise ValueError("Ridge alpha must be positive")
    if int(fit.get("fit_rows", 0)) < 1000:
        raise ValueError("Forward ridge needs at least 1000 source rows")
    return fit


class LabelFreeSequenceDataset(Dataset):
    """Contiguous sequence input that carries no outcome or label arrays."""

    def __init__(self, features: np.ndarray, *, seq_len: int, timestamps: Any):
        values = np.asarray(features, dtype=np.float32)
        times = pd.Series(pd.to_datetime(timestamps, utc=True, errors="raise"))
        if (
            values.ndim != 2
            or not len(values)
            or len(values) != len(times)
            or seq_len < 2
            or not np.isfinite(values).all()
            or times.isna().any()
            or not times.is_monotonic_increasing
            or times.duplicated().any()
        ):
            raise ValueError("Invalid label-free sequence inputs")
        self.features = values
        self.seq_len = int(seq_len)
        ends = np.arange(self.seq_len - 1, len(values))
        gaps = times.diff().ne(pd.Timedelta(hours=1)).astype(int)
        gaps.iloc[0] = 0
        runs = gaps.cumsum().to_numpy()
        self.end_positions = ends[
            runs[ends] == runs[ends - self.seq_len + 1]
        ]
        if not len(self.end_positions):
            raise ValueError("No contiguous label-free sequence is available")

    def __len__(self) -> int:
        return len(self.end_positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.end_positions[index])
        start = end - self.seq_len + 1
        return (
            torch.from_numpy(self.features[start : end + 1]),
            torch.tensor(end, dtype=torch.long),
        )


def predict_label_free_model(
    model: torch.nn.Module,
    dataset: LabelFreeSequenceDataset,
    source: pd.DataFrame,
    *,
    batch_size: int,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    torch_device = _device(device)
    model = model.to(torch_device)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    probabilities: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, row_positions in loader:
            logits, _ = model.forward_heads(features.to(torch_device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            positions.append(row_positions.numpy())
    row_positions = np.concatenate(positions)
    rows = source.iloc[row_positions].copy().reset_index(drop=True)
    rows["raw_score"] = np.concatenate(probabilities)
    rows["source_row_position"] = row_positions
    return rows


def predict_label_free_artifacts(
    frame: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    artifact_root: str | Path,
    config: dict[str, Any],
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Run the sealed model without requiring label or forward-return columns."""

    root = Path(artifact_root)
    checked = validate_shadow_manifest(manifest, artifact_root=root)
    model_identity = checked["model"]
    feature_columns = list(model_identity["feature_columns"])
    cfg = profile_config(config, str(model_identity["profile"]))
    hmm_features = list((cfg.get("hmm", {}) or {}).get("features", []) or [])
    required = list(
        dict.fromkeys(
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "atr_14",
                *feature_columns,
                *hmm_features,
            ]
        )
    )
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Label-free frame is missing columns: {missing}")
    original = frame[required].copy()
    original["timestamp"] = pd.to_datetime(original.timestamp, utc=True, errors="raise")
    original = original.sort_values("timestamp").reset_index(drop=True)
    numeric = list(dict.fromkeys([*feature_columns, *hmm_features, "open", "high", "low", "close", "atr_14"]))
    original[numeric] = original[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(original[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Label-free model inputs must be finite")
    artifacts = checked["artifacts"]
    scaler = joblib.load(_safe_artifact_path(root, artifacts["scaler"]["path"]))
    hmm = joblib.load(_safe_artifact_path(root, artifacts["hmm"]["path"]))
    transformed = original.copy()
    transformed.loc[:, feature_columns] = scaler.transform(
        transformed[feature_columns]
    )
    transformed = _add_regime_probs(transformed, hmm, cfg)
    sequence = LabelFreeSequenceDataset(
        transformed[feature_columns].to_numpy(dtype=np.float32),
        seq_len=int((cfg.get("model", {}) or {}).get("seq_len", 64)),
        timestamps=transformed.timestamp,
    )
    checkpoint_path = _safe_artifact_path(root, artifacts["model"]["path"])
    try:
        checkpoint = torch.load(checkpoint_path, map_location=_device(device), weights_only=False)
    except TypeError:  # pragma: no cover - old Torch compatibility
        checkpoint = torch.load(checkpoint_path, map_location=_device(device))
    if list(checkpoint.get("feature_columns", [])) != feature_columns:
        raise ValueError("Checkpoint feature columns differ from the manifest")
    model = _build_model(len(feature_columns), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    predicted = predict_label_free_model(
        model,
        sequence,
        original,
        batch_size=int((cfg.get("training", {}) or {}).get("batch_size", 256)),
        device=device,
    )
    decision = predicted.timestamp + pd.Timedelta(hours=1)
    block = checked["block"]
    mask = decision.between(
        _utc(block["evidence_start_inclusive"]),
        _utc(block["evidence_end_inclusive"]),
        inclusive="both",
    )
    return predicted.loc[mask].reset_index(drop=True)


def feature_row_fingerprints(
    frame: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    if not feature_columns or any(column not in frame for column in feature_columns):
        raise ValueError("Feature snapshot columns are incomplete")
    values = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Feature snapshots must be finite")
    fingerprints = []
    for timestamp, row in zip(frame.timestamp, values.itertuples(index=False, name=None)):
        record = {
            "timestamp": _utc(timestamp).isoformat(),
            "features": {
                column: format(float(value), ".17g")
                for column, value in zip(feature_columns, row)
            },
        }
        fingerprints.append(hashlib.sha256(_canonical_json(record)).hexdigest())
    return np.asarray(fingerprints, dtype=object)


def build_shadow_scores(
    raw_predictions: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    cdf_reference: np.ndarray,
    generated_at: Any,
) -> pd.DataFrame:
    """Apply only sealed CDF/ridge transforms; outcome columns are ignored."""

    checked = validate_shadow_manifest(manifest)
    required = ["timestamp", "raw_score", "open", "high", "low", "close", "atr_14"]
    missing = [column for column in required if column not in raw_predictions]
    if missing:
        raise ValueError(f"Raw shadow predictions are missing columns: {missing}")
    frame = raw_predictions.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    numeric = required[1:]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if (
        frame.timestamp.isna().any()
        or frame.timestamp.duplicated().any()
        or not np.isfinite(frame[numeric].to_numpy(dtype=float)).all()
        or not frame.raw_score.between(0, 1).all()
        or (frame.close <= 0).any()
        or (frame.atr_14 <= 0).any()
    ):
        raise ValueError("Invalid raw forward-shadow predictions")
    percentile = empirical_cdf_right(cdf_reference, frame.raw_score.to_numpy())
    atr_fraction = frame.atr_14.to_numpy(dtype=float) / frame.close.to_numpy(dtype=float)
    candidate_x = np.column_stack(
        [percentile, atr_fraction, percentile * atr_fraction]
    )
    atr_x = np.column_stack(
        [np.zeros(len(frame)), atr_fraction, np.zeros(len(frame))]
    )
    payoff = checked["payoff_layer"]
    candidate_utility = predict_ridge_payoff(payoff["candidate_fit"], candidate_x)
    atr_utility = predict_ridge_payoff(payoff["atr_only_fit"], atr_x)
    if not np.isfinite(candidate_utility).all() or not np.isfinite(atr_utility).all():
        raise ValueError("Frozen payoff layer produced invalid values")
    output = frame[["timestamp", "open", "high", "low", "close", "atr_14"]].copy()
    output["decision_time"] = output.timestamp + pd.Timedelta(hours=1)
    output["generated_at"] = _utc(generated_at)
    output["process_id"] = checked["process_id"]
    output["candidate_id"] = checked["candidate_id"]
    output["block_id"] = checked["block"]["block_id"]
    output["block_manifest_hash"] = checked["manifest_hash"]
    output["raw_score"] = frame.raw_score.to_numpy(dtype=float)
    output["score_percentile"] = percentile
    output["decision_atr_close_fraction"] = atr_fraction
    output["predicted_candidate_utility"] = candidate_utility
    output["candidate_score"] = utility_to_score(candidate_utility)
    output["candidate_action"] = candidate_utility > 0
    output["predicted_atr_utility"] = atr_utility
    output["atr_score"] = utility_to_score(atr_utility)
    output["atr_action"] = atr_utility > 0
    feature_columns = list(checked["model"]["feature_columns"])
    output["feature_snapshot_sha256"] = feature_row_fingerprints(
        frame, feature_columns
    )
    return output


SHADOW_LEDGER_COLUMNS = (
    "timestamp",
    "decision_time",
    "generated_at",
    "process_id",
    "candidate_id",
    "block_id",
    "block_manifest_hash",
    "raw_score",
    "score_percentile",
    "decision_atr_close_fraction",
    "predicted_candidate_utility",
    "candidate_score",
    "candidate_action",
    "predicted_atr_utility",
    "atr_score",
    "atr_action",
    "open",
    "high",
    "low",
    "close",
    "atr_14",
    "feature_snapshot_sha256",
    "evidence_role",
)


@contextmanager
def _ledger_lease(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - Windows CI is primary for this workspace
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _json_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _utc(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Ledger values must be finite")
        return result
    return value


def read_shadow_ledger(path: str | Path, *, with_hash: bool = False):
    ledger_path = Path(path)
    rows: list[dict[str, Any]] = []
    previous = ""
    if ledger_path.exists():
        for line_number, line in enumerate(
            ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            payload = json.loads(line)
            stored = str(payload.pop("hash", ""))
            actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
            if (
                payload.get("ledger_version") != SHADOW_LEDGER_VERSION
                or payload.get("previous_hash") != previous
                or stored != actual
            ):
                raise ValueError(f"Forward-shadow ledger hash chain failed at line {line_number}")
            rows.append(payload["prediction"])
            previous = stored
    frame = pd.DataFrame(rows, columns=SHADOW_LEDGER_COLUMNS)
    return (frame, previous) if with_hash else frame


def append_shadow_predictions(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    timely_delay_minutes: int = 15,
) -> dict[str, Any]:
    checked = validate_shadow_manifest(manifest)
    if timely_delay_minutes < 0:
        raise ValueError("Timely delay cannot be negative")
    missing = [column for column in SHADOW_LEDGER_COLUMNS if column != "evidence_role" and column not in frame]
    if missing:
        raise ValueError(f"Shadow ledger rows are missing columns: {missing}")
    incoming = frame[[column for column in SHADOW_LEDGER_COLUMNS if column != "evidence_role"]].copy()
    for column in ("timestamp", "decision_time", "generated_at"):
        incoming[column] = pd.to_datetime(incoming[column], utc=True, errors="raise")
    if (
        incoming.empty
        or incoming.timestamp.isna().any()
        or incoming.timestamp.duplicated().any()
        or not incoming.timestamp.is_monotonic_increasing
        or not incoming.decision_time.eq(incoming.timestamp + pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Shadow ledger rows need unique ordered hourly decisions")
    block = checked["block"]
    start = _utc(block["evidence_start_inclusive"])
    end = _utc(block["evidence_end_inclusive"])
    if not incoming.decision_time.between(start, end, inclusive="both").all():
        raise ValueError("Pre-lock, burn-in or out-of-block rows cannot enter evidence")
    identities = {
        "process_id": checked["process_id"],
        "candidate_id": checked["candidate_id"],
        "block_id": block["block_id"],
        "block_manifest_hash": checked["manifest_hash"],
    }
    if any(not incoming[key].astype(str).eq(str(value)).all() for key, value in identities.items()):
        raise ValueError("Shadow ledger model/block identity mismatch")
    numeric = [
        "raw_score",
        "score_percentile",
        "decision_atr_close_fraction",
        "predicted_candidate_utility",
        "candidate_score",
        "predicted_atr_utility",
        "atr_score",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
    ]
    incoming[numeric] = incoming[numeric].apply(pd.to_numeric, errors="raise")
    if (
        not np.isfinite(incoming[numeric].to_numpy(dtype=float)).all()
        or not incoming[["raw_score", "score_percentile", "candidate_score", "atr_score"]]
        .apply(lambda column: column.between(0, 1))
        .all()
        .all()
        or not incoming.candidate_action.astype(bool).eq(incoming.predicted_candidate_utility.gt(0)).all()
        or not incoming.atr_action.astype(bool).eq(incoming.predicted_atr_utility.gt(0)).all()
        or not incoming.feature_snapshot_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise ValueError("Invalid shadow ledger values/actions")
    incoming["candidate_action"] = incoming.candidate_action.astype(bool)
    incoming["atr_action"] = incoming.atr_action.astype(bool)
    timely = incoming.generated_at.between(
        incoming.decision_time,
        incoming.decision_time + pd.Timedelta(minutes=timely_delay_minutes),
        inclusive="both",
    )
    incoming["evidence_role"] = np.where(
        timely,
        "timely_shadow",
        "sealed_batch_replay",
    )
    for column in ("timestamp", "decision_time", "generated_at"):
        incoming[column] = incoming[column].map(lambda value: value.isoformat())
    records = [
        {column: _json_scalar(value) for column, value in row.items()}
        for row in incoming[list(SHADOW_LEDGER_COLUMNS)].to_dict("records")
    ]
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lease(ledger_path):
        old, previous = read_shadow_ledger(ledger_path, with_hash=True)
        if not old.empty and any(
            not old[key].astype(str).eq(str(value)).all()
            for key, value in identities.items()
        ):
            raise ValueError("A block ledger cannot change sealed identity")
        by_time = {str(row["timestamp"]): row for row in old.to_dict("records")}
        additions: list[dict[str, Any]] = []
        for row in records:
            timestamp = str(row["timestamp"])
            if timestamp in by_time:
                if row != by_time[timestamp]:
                    raise ValueError("Previously recorded shadow prediction changed")
                continue
            last = additions[-1]["timestamp"] if additions else (max(by_time) if by_time else None)
            if last is not None and _utc(timestamp) - _utc(last) != pd.Timedelta(hours=1):
                raise ValueError("Shadow ledger additions must be contiguous and later")
            additions.append(row)
        with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in additions:
                payload = {
                    "ledger_version": SHADOW_LEDGER_VERSION,
                    "previous_hash": previous,
                    "prediction": row,
                }
                previous = hashlib.sha256(_canonical_json(payload)).hexdigest()
                handle.write(
                    json.dumps(
                        {**payload, "hash": previous},
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "appended_rows": len(additions),
        "total_rows": len(old) + len(additions),
        "head_hash": previous,
        **identities,
        "fit_operations_performed": 0,
        "selection_operations_performed": 0,
    }

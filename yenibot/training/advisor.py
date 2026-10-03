"""Validation-only advisor experiments; atomic epoch checkpoints and strict splits.

python -m yenibot.training.advisor prepare --config configs/advisor_ablation.yaml
python -m yenibot.training.advisor run --config configs/advisor_ablation.yaml --experiment A
Same command resumes. No test prediction path exists in this development runner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader

from yenibot.features.builder import build_feature_matrix
from yenibot.data.advisor_validation import validate_advisor_klines
from yenibot.labeling.triple_barrier import add_long_only_labels
from yenibot.training.advisor_models import build_advisor_model
from yenibot.training.dataset import SequenceDataset
from yenibot.training.walk_forward import PurgedWalkForwardCV


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def exclusive_run(directory: Path):
    """OS lock releases on process death; no stale PID unlock guessing required."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_directory = directory
    if os.environ.get("ADVISOR_LOCK_ROOT"):
        # Drive mounts may not implement POSIX flock; use the runtime's disk.
        key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:24]
        lock_directory = Path(os.environ["ADVISOR_LOCK_ROOT"]) / key
        lock_directory.mkdir(parents=True, exist_ok=True)
    with (lock_directory / "run.lock").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def read_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def assert_hourly(frame: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if len(timestamps) < 2 or not timestamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Expected unique, sorted, contiguous hourly rows; never compress missing hours")


def audit_boundary(frame: pd.DataFrame, previous: np.ndarray, following: np.ndarray, horizon: int) -> dict:
    """Protect all target information, including fixed-horizon diagnostic returns."""
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    end_times = pd.to_datetime(frame.iloc[previous]["label_end_timestamp"], utc=True)
    if end_times.isna().any():
        raise ValueError("Missing label end time")
    expected = timestamps.iloc[previous] + pd.Timedelta(hours=horizon)
    if not np.array_equal(end_times.to_numpy(), expected.to_numpy()):
        raise ValueError("Label horizon metadata does not match configured horizon")
    if "exit_timestamp" in frame:
        exits = pd.to_datetime(frame.iloc[previous]["exit_timestamp"], utc=True)
        if exits.isna().any() or (exits > end_times).any():
            raise ValueError("Barrier exit occurs outside declared target interval")
    next_start = timestamps.iloc[following[0]]
    if end_times.max() >= next_start:
        raise ValueError("Label information overlaps the following section")
    return {"last_target_timestamp": end_times.max().isoformat(),
            "next_section_start": next_start.isoformat(), "passed": True}


def prepare(config: dict) -> None:
    """Use existing causal builders; no download, final test or parameter search."""
    cfg = copy.deepcopy(read_config(config["data"]["feature_config"]))
    cfg["features"]["wavelet"]["enabled"] = config["data"]["wavelet"]
    cfg["features"]["active_profile"] = "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility"
    snapshot = Path(config["data"]["snapshot"])
    manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
    raw = {}
    hashes = {}
    raw_audits = {}
    stop = pd.Timestamp(config["development"]["end"]) + pd.Timedelta(hours=config["labeling"]["max_holding_bars"])
    for interval in ("1h", "4h"):
        path = snapshot / f"btc_{interval}.parquet"
        hashes[interval] = sha256(path)
        if hashes[interval] != manifest["files"][interval]["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {interval}")
        part = pd.read_parquet(path)
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        part = part.loc[part["timestamp"] <= stop].copy()
        raw[interval], raw_audits[interval] = validate_advisor_klines(
            part, interval, zero_activity_policy=config["data"].get("zero_activity_policy", "error"))
        print(f"Raw audit {interval}: {raw_audits[interval]}", flush=True)
    frame = build_feature_matrix(raw["1h"], raw["4h"], cfg).frame
    frame = add_long_only_labels(frame, **config["labeling"])
    frame["label_end_timestamp"] = frame["timestamp"] + pd.Timedelta(hours=config["labeling"]["max_holding_bars"])
    frame = frame.loc[frame["timestamp"].between(pd.Timestamp(config["development"]["start"]), pd.Timestamp(config["development"]["end"]))].reset_index(drop=True)
    assert_hourly(frame)
    if frame[config["data"]["basic_features"]].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise ValueError("Basic feature values are unavailable")
    # Freeze the explicit full input list before its first experiment.
    feature_path = Path(config["data"]["feature_columns_file"])
    full_features = feature_path.read_text().splitlines() if feature_path.exists() else []
    missing = [column for column in full_features if column not in frame or frame[column].isna().any()]
    output = Path(config["data"]["frame"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    os.replace(temp, output)
    atomic_json(output.with_suffix(".manifest.json"), {
        "rows": len(frame), "start": frame.timestamp.iloc[0].isoformat(),
        "end": frame.timestamp.iloc[-1].isoformat(), "frame_sha256": sha256(output),
        "source_hashes": hashes, "feature_config": cfg, "protocol": config,
        "raw_quality_audits": raw_audits,
        "preparation_source_hashes": {str(path): sha256(path) for path in [
            Path(__file__).parents[1] / "data/advisor_validation.py",
            Path(__file__).parents[1] / "features/builder.py",
            Path(__file__).parents[1] / "features/wavelet.py",
            Path(__file__).parents[1] / "labeling/triple_barrier.py",
        ]},
        "basic_features": config["data"]["basic_features"],
        "full_features_missing": missing, "test_evaluations": 0,
        "role": "historical_development_not_unseen_test",
    })
    print(f"Prepared {len(frame)} development rows. Missing full inputs: {missing}", flush=True)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def metrics(labels: np.ndarray, probs: np.ndarray, returns: np.ndarray, threshold: float) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, probs >= threshold, average="binary", zero_division=0)
    correlation = float(spearmanr(probs, returns).statistic) if np.std(probs) > 0 and np.std(returns) > 0 else None
    return {"bce": float(log_loss(labels, probs, labels=[0, 1])),
            "average_precision": float(average_precision_score(labels, probs)) if np.any(labels == 1) else None,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "accuracy": float(accuracy_score(labels, probs >= threshold)),
            "rank_ic": correlation if correlation is not None and np.isfinite(correlation) else None,
            "prevalence": float(labels.mean()), "threshold": threshold, "samples": len(labels)}


def evaluate(model, loader, device, threshold: float) -> tuple[dict, pd.DataFrame]:
    model.eval()
    logits_all, labels, returns, positions = [], [], [], []
    with torch.no_grad():
        for x, y, fwd, position in loader:
            logits_all.append(model(x.to(device), return_logits=True).cpu())
            labels.append(y)
            returns.append(fwd)
            positions.append(position)
    logits = torch.cat(logits_all)
    y = torch.cat(labels)
    probs = torch.sigmoid(logits).numpy()
    result = metrics(y.numpy(), probs, torch.cat(returns).numpy(), threshold)
    # Stable logit BCE is the common epoch selection criterion for every model.
    result["bce"] = float(nn.functional.binary_cross_entropy_with_logits(logits, y))
    predictions = pd.DataFrame({"source_row_position": torch.cat(positions).numpy(),
                                "label": y.numpy(), "probability": probs,
                                "forward_return": torch.cat(returns).numpy()})
    return result, predictions


def run_fold(frame: pd.DataFrame, fold, config: dict, experiment: dict, features: list[str],
             seed: int, directory: Path, signature: str, device: torch.device) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "validation_metrics.json"
    checkpoint_path = directory / "last.pt"
    if result_path.exists():
        completed = json.loads(result_path.read_text())
        if completed["signature"] != signature:
            raise ValueError("Completed fold signature mismatch; use a new run directory")
        return completed
    horizon = config["labeling"]["max_holding_bars"]
    audit = {"train_validation": audit_boundary(frame, fold.train, fold.val, horizon),
             "validation_test_section": audit_boundary(frame, fold.val, fold.test, horizon)}
    train = frame.iloc[fold.train].copy()
    val = frame.iloc[fold.val].copy()
    scaler = RobustScaler().fit(train[features])
    train.loc[:, features] = scaler.transform(train[features])
    val.loc[:, features] = scaler.transform(val[features])
    seq_len = config["model"]["seq_len"]
    forward = f"fwd_return_{horizon}h"
    def dataset(part):
        return SequenceDataset(part[features].to_numpy(np.float32), part.label.to_numpy(np.float32),
                               part[forward].to_numpy(np.float32), seq_len=seq_len)
    batch = config["training"]["batch_size"]
    train_loader = DataLoader(dataset(train), batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset(val), batch_size=batch, shuffle=False, num_workers=0)
    seed_everything(seed + fold.fold, config["training"]["deterministic"])
    model = build_advisor_model(len(features), experiment["architecture"], config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    history, best_state = [], None
    best_loss, patience, start_epoch, best_epoch = float("inf"), 0, 0, 0
    if checkpoint_path.exists():
        # Only self-created, local checkpoints are loaded; never untrusted weights.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["device"] != str(device):
            raise ValueError("Checkpoint protocol/data/source/device mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        history, best_state = checkpoint["history"], checkpoint["best_state"]
        best_loss, patience = checkpoint["best_loss"], checkpoint["patience"]
        start_epoch, best_epoch = checkpoint["epoch"], checkpoint["best_epoch"]
        restore_rng(checkpoint["rng"])
        print(f"Resume fold {fold.fold}, seed {seed}, after epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, config["training"]["epochs"]):
        if patience >= config["training"]["patience"]:
            break
        model.train()
        total_loss, count = 0.0, 0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x, return_logits=True), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)
        validation, _ = evaluate(model, val_loader, device, config["training"]["threshold"])
        if not np.isfinite(validation["bce"]):
            raise ValueError("Non-finite validation BCE")
        if validation["bce"] < best_loss:
            best_loss, best_epoch, patience = validation["bce"], epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        history.append({"epoch": epoch + 1, "train_bce": total_loss / count, "validation": validation})
        atomic_checkpoint(checkpoint_path, {"signature": signature, "device": str(device), "epoch": epoch + 1,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "best_state": best_state, "best_loss": best_loss, "best_epoch": best_epoch,
                          "patience": patience, "history": history, "rng": rng_state(),
                          "scaler_center": scaler.center_, "scaler_scale": scaler.scale_, "features": features})
        atomic_json(directory / "progress.json", {"status": "training", "fold": fold.fold, "seed": seed,
                    "last_saved_epoch": epoch + 1, "best_epoch": best_epoch, "best_validation_bce": best_loss,
                    "updated_at": datetime.now(timezone.utc).isoformat(), "signature": signature})
        atomic_text(directory / "STATUS.md", f"# Eğitim durumu\n\nFold: {fold.fold}; seed: {seed}.\n\n"
                    f"Son kaydedilen epoch: {epoch + 1}; en iyi epoch: {best_epoch}.\n\n"
                    f"En iyi validation BCE: {best_loss:.6f}. Test değerlendirmesi: 0.\n\n"
                    "İşlem kesilirse aynı komutla son kaydedilen epoch'tan devam edin.\n")
        print(f"fold={fold.fold} seed={seed} epoch={epoch+1} train_BCE={total_loss/count:.6f} val_BCE={validation['bce']:.6f} best={best_epoch}", flush=True)
    model.load_state_dict(best_state)
    validation, predictions = evaluate(model, val_loader, device, config["training"]["threshold"])
    predictions["timestamp"] = val.iloc[predictions.source_row_position].timestamp.to_numpy()
    predictions.to_parquet(directory / "validation_predictions.parquet", index=False)
    result = {"status": "complete", "signature": signature, "fold": fold.fold, "seed": seed,
              "best_epoch": best_epoch, "epochs_trained": len(history), "validation": validation,
              "boundary_audit": audit, "parameters": sum(p.numel() for p in model.parameters()),
              "train_rows": len(train), "train_sequences": len(train_loader.dataset),
              "validation_rows": len(val), "features": features, "test_evaluations": 0}
    atomic_json(result_path, result)
    atomic_json(directory / "progress.json", result)
    atomic_text(directory / "STATUS.md", f"# Fold tamamlandı\n\nFold: {fold.fold}; seed: {seed}.\n\n"
                f"Eğitilen epoch: {len(history)}; seçilen epoch: {best_epoch}.\n\n"
                f"Validation BCE: {validation['bce']:.6f}. Test değerlendirmesi: 0.\n")
    pd.DataFrame([{"epoch": row["epoch"], "train_bce": row["train_bce"], **row["validation"]} for row in history]).to_csv(directory / "history.csv", index=False)
    return result


def run(config: dict, name: str, fold_ids: list[int] | None = None, seeds: list[int] | None = None) -> None:
    if config["training"]["loss"] != "bce" or config["training"]["selection_metric"] != "validation_bce":
        raise ValueError("This first-stage runner supports pure BCE and validation-BCE selection only")
    experiment = config["experiments"][name]
    features = (config["data"]["basic_features"] if experiment["features"] == "basic"
                else Path(config["data"]["feature_columns_file"]).read_text().splitlines())
    if not features or len(set(features)) != len(features):
        raise ValueError("Explicit unique feature list required")
    path = Path(config["data"]["frame"])
    metadata = json.loads(path.with_suffix(".manifest.json").read_text())
    data_hash = sha256(path)
    if metadata["frame_sha256"] != data_hash or any(
        metadata["protocol"][key] != config[key] for key in ("data", "development", "labeling")
    ):
        raise ValueError("Prepared data/protocol mismatch; prepare matching data first")
    expected_feature_cfg = read_config(config["data"]["feature_config"])
    expected_feature_cfg["features"]["wavelet"]["enabled"] = config["data"]["wavelet"]
    expected_feature_cfg["features"]["active_profile"] = "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility"
    if metadata["feature_config"] != expected_feature_cfg:
        raise ValueError("Feature configuration changed since preparation")
    frame = pd.read_parquet(path)
    assert_hourly(frame)
    required = [*features, "label", f"fwd_return_{config['labeling']['max_holding_bars']}h"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Required inputs missing; do not silently drop: {missing}")
    if frame[required].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise ValueError("Non-finite active inputs")
    if not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError("Binary labels required")
    cv = PurgedWalkForwardCV(**config["walk_forward"], label_horizon_bars=config["labeling"]["max_holding_bars"])
    all_folds = list(cv.split(len(frame)))
    selected = all_folds if fold_ids is None else [fold for fold in all_folds if fold.fold in fold_ids]
    if not selected or (fold_ids is not None and len(selected) != len(set(fold_ids))):
        raise ValueError("Invalid/empty fold selection")
    for fold in selected:
        for before, after in ((fold.train, fold.val), (fold.val, fold.test)):
            audit_boundary(frame, before, after, config["labeling"]["max_holding_bars"])
    selected_seeds = seeds if seeds is not None else config["training"]["seeds"]
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("Unique seeds required")
    torch.set_num_threads(config["training"]["torch_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    source_files = [Path(__file__), Path(__file__).with_name("advisor_models.py"), Path(__file__).with_name("dataset.py"),
                    Path(__file__).with_name("walk_forward.py"), Path(__file__).parents[1] / "models" / "tcn.py",
                    Path(__file__).parents[1] / "models" / "hybrid.py"]
    manifest = {"config": config, "experiment": name, "features": features, "data_sha256": data_hash,
                "source_hashes": {str(p): sha256(p) for p in source_files},
                "folds": [fold.fold for fold in selected], "seeds": selected_seeds,
                "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                "python": sys.version, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "selection": "validation_bce_only", "test_evaluations": 0}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = Path(config["output"]) / f"{name}_{signature[:12]}"
    with exclusive_run(directory):
        manifest_path = directory / "protocol.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run manifest mismatch")
        atomic_json(manifest_path, manifest)
        progress = {"status": "running", "experiment": name, "signature": signature,
                    "total_folds": len(selected), "seeds": selected_seeds, "completed": [], "test_evaluations": 0}
        try:
            for seed in selected_seeds:
                for fold in selected:
                    progress["current"] = {"seed": seed, "fold": fold.fold}
                    atomic_json(directory / "status.json", progress)
                    atomic_text(directory / "STATUS.md", f"# Deney {name}: çalışıyor\n\n"
                                f"Tamamlanan fold/seed: {len(progress['completed'])}/{len(selected)*len(selected_seeds)}.\n\n"
                                f"Mevcut seed: {seed}; fold: {fold.fold}.\n\n"
                                f"Epoch durumu: seed_{seed}/fold_{fold.fold:03d}/STATUS.md.\n\n"
                                "Seçim yalnızca validation BCE; test değerlendirmesi: 0.\n")
                    result = run_fold(frame, fold, config, experiment, features, seed,
                                      directory / f"seed_{seed}" / f"fold_{fold.fold:03d}", signature, device)
                    progress["completed"].append({"seed": seed, "fold": fold.fold,
                                                   "validation": result["validation"], "best_epoch": result["best_epoch"]})
                    atomic_json(directory / "status.json", progress)
                    pd.DataFrame([{**item["validation"], "seed": item["seed"], "fold": item["fold"], "best_epoch": item["best_epoch"]}
                                  for item in progress["completed"]]).to_csv(directory / "validation_summary.csv", index=False)
            progress["status"] = "complete"
            progress.pop("current", None)
            atomic_json(directory / "status.json", progress)
            atomic_text(directory / "STATUS.md", f"# Deney {name}: tamamlandı\n\n"
                        f"Tamamlanan fold/seed: {len(progress['completed'])}.\n\n"
                        "Sonuçlar: validation_summary.csv. Test değerlendirmesi: 0.\n")
        except BaseException as exc:
            progress["status"], progress["error"] = "interrupted_or_failed", repr(exc)
            atomic_json(directory / "status.json", progress)
            atomic_text(directory / "STATUS.md", f"# Deney {name}: kesildi veya hata oluştu\n\n"
                        f"Hata: {exc!r}\n\nMevcut fold/seed: {progress.get('current')}.\n\n"
                        "Son checkpoint korunuyor. Aynı komutla devam edin; önce hatayı inceleyin.\n")
            raise
    print(f"Run directory: {directory}; validation only, no test evaluation", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", choices=["A", "B", "C", "D"], default="A")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.action == "prepare":
        prepare(config)
    else:
        run(config, args.experiment, args.folds, args.seeds)


if __name__ == "__main__":
    main()

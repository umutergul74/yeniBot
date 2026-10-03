from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest
import torch

from yenibot.training import advisor
from yenibot.training.advisor_models import build_advisor_model
from yenibot.training.walk_forward import PurgedWalkForwardCV


def frame_fixture(n=100, horizon=3):
    timestamp = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamp, "label_end_timestamp": timestamp + pd.Timedelta(hours=horizon),
                         "exit_timestamp": timestamp + pd.Timedelta(hours=1),
                         "x": np.sin(np.arange(n)), "label": np.arange(n) % 2,
                         f"fwd_return_{horizon}h": np.cos(np.arange(n)) * 0.001})


def test_horizon_guard_and_exact_boundary():
    sizes = dict(train_bars=30, val_bars=15, test_bars=15, step_bars=15, purge_bars=10, embargo_bars=6)
    with pytest.raises(ValueError, match="cover the label horizon"):
        PurgedWalkForwardCV(**sizes, label_horizon_bars=10)
    frame = frame_fixture(horizon=10)
    old = next(PurgedWalkForwardCV(**sizes).split(len(frame)))
    with pytest.raises(ValueError, match="overlaps"):
        advisor.audit_boundary(frame, old.val, old.test, 10)
    sizes["embargo_bars"] = 10
    fold = next(PurgedWalkForwardCV(**sizes, label_horizon_bars=10).split(len(frame)))
    assert advisor.audit_boundary(frame, fold.val, fold.test, 10)["passed"]


def test_reject_missing_hour_and_false_event_metadata():
    frame = frame_fixture()
    with pytest.raises(ValueError, match="contiguous"):
        advisor.assert_hourly(frame.drop(index=20))
    frame.loc[19, "exit_timestamp"] = frame.loc[19, "label_end_timestamp"] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="outside"):
        advisor.audit_boundary(frame, np.arange(20), np.arange(25, 40), 3)


@pytest.mark.parametrize("architecture", ["gru", "tcn", "tcn_gru"])
def test_model_control_shapes_and_gradients(architecture):
    settings = dict(seq_len=4, tcn_channels=4, tcn_kernel_size=3, tcn_dilations=[1, 2],
                    gru_hidden=4, gru_layers=1, dropout=0.0, fusion_hidden=4)
    model = build_advisor_model(2, architecture, settings)
    logits = model(torch.randn(3, 4, 2), return_logits=True)
    assert logits.shape == (3,)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_interrupted_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch, device_name):
    """Resume after a saved epoch must preserve dropout/shuffle/Adam states."""
    torch.set_num_threads(2)
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    frame = frame_fixture()
    config = dict(labeling={"max_holding_bars": 3},
                  model=dict(seq_len=4, tcn_channels=4, tcn_kernel_size=3, tcn_dilations=[1],
                             gru_hidden=4, gru_layers=1, dropout=0.2, fusion_hidden=4),
                  training=dict(batch_size=8, deterministic=True, learning_rate=0.001, weight_decay=0.0001,
                                epochs=3, patience=10, threshold=0.5, grad_clip=1.0))
    fold = next(PurgedWalkForwardCV(train_bars=32, val_bars=12, test_bars=12, step_bars=12,
                                   purge_bars=3, embargo_bars=3, label_horizon_bars=3).split(len(frame)))
    kwargs = dict(frame=frame, fold=fold, config=config, experiment={"architecture": "gru"},
                  features=["x"], seed=42, signature="test-signature", device=torch.device(device_name))
    uninterrupted = advisor.run_fold(**kwargs, directory=tmp_path / "uninterrupted")
    real_evaluate = advisor.evaluate
    calls = 0
    def crash_after_first_epoch(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated credit/process interruption")
        return real_evaluate(*args, **kw)
    monkeypatch.setattr(advisor, "evaluate", crash_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulated"):
        advisor.run_fold(**kwargs, directory=tmp_path / "resumed")
    monkeypatch.setattr(advisor, "evaluate", real_evaluate)
    resumed = advisor.run_fold(**kwargs, directory=tmp_path / "resumed")
    assert resumed == uninterrupted
    left = torch.load(tmp_path / "uninterrupted" / "last.pt", weights_only=False)
    right = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert left["history"] == right["history"]
    assert all(torch.equal(value, right["model"][key]) for key, value in left["model"].items())
    changed = copy.deepcopy(kwargs)
    changed["signature"] = "changed-data-or-config"
    with pytest.raises(ValueError, match="signature mismatch"):
        advisor.run_fold(**changed, directory=tmp_path / "resumed")

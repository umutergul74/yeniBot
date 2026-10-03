import json

import numpy as np
import pandas as pd
import pytest

from yenibot.features.advisor_oi import append_oi_features, normalize_oi_source
from yenibot.training.advisor import sha256
from yenibot.training.advisor_colab import freeze_oi_inputs, pin_completed_a_reference


def source(start='2022-01-01', periods=25):
    return pd.DataFrame({'timestamp': pd.date_range(start, periods=periods, freq='5min', tz='UTC'),
                         'sum_open_interest': np.exp(np.arange(periods)*.001)*100,
                         'sum_open_interest_value': np.exp(np.arange(periods)*.002)*1000})


def test_backward_matching_log_difference_and_base_preservation():
    raw = source()
    frame = pd.DataFrame({'timestamp': raw.timestamp.iloc[[12, 24]].reset_index(drop=True), 'label': [1, 0]})
    # The subsequent source value must not enter the earlier prediction.
    raw.loc[13, 'sum_open_interest'] = 99999
    result, audit = append_oi_features(frame, raw, min_coverage=1)
    np.testing.assert_allclose(result.fut_oi_log_return, [.001, .001])
    np.testing.assert_allclose(result.fut_oi_value_log_return, [.002, .002])
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    assert audit['future_matches'] == 0 and audit['neutral_filled_rows'] == 0


def test_long_gap_not_treated_as_snapshot_return_and_low_coverage_rejected():
    raw = source().iloc[[0, 1, 24]]
    frame = pd.DataFrame({'timestamp': raw.timestamp.iloc[[1, 2]].reset_index(drop=True)})
    with pytest.raises(ValueError, match='coverage'):
        append_oi_features(frame, raw)
    result, audit = append_oi_features(frame, raw, min_coverage=.5)
    assert result.fut_oi_log_return.iloc[1] == 0
    assert audit['neutral_filled_rows'] == 1 and audit['long_source_gaps'] == 1


def test_invalid_or_conflicting_source_rejected():
    raw = source()
    conflict = raw.iloc[[0]].copy()
    conflict['sum_open_interest'] = 123
    with pytest.raises(ValueError, match='Conflicting'):
        normalize_oi_source(pd.concat([raw, conflict]))
    raw.loc[2, 'sum_open_interest'] = 0
    with pytest.raises(ValueError, match='non-positive'):
        normalize_oi_source(raw)


def test_monthly_download_resume_and_frozen_checksum(tmp_path):
    calls = []
    def download(lower, upper):
        calls.append(lower.month)
        if lower.month == 2 and len(calls) == 2:
            raise RuntimeError('simulated disconnect')
        return source(lower, periods=3)
    settings = dict(start='2022-01-01T00:00:00Z', end='2022-02-01T00:10:00Z', downloader=download)
    with pytest.raises(RuntimeError, match='disconnect'):
        freeze_oi_inputs(None, tmp_path, **settings)
    path = freeze_oi_inputs(None, tmp_path, **settings)
    assert calls == [1, 2, 2]  # January was persisted before interruption.
    assert freeze_oi_inputs(None, tmp_path, **settings) == path
    assert calls == [1, 2, 2]
    assert json.loads(path.with_suffix('.manifest.json').read_text())['sha256'] == sha256(path)
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        freeze_oi_inputs(None, tmp_path, **settings)


def test_a_reference_requires_completed_matching_scope(tmp_path):
    (tmp_path / 'data').mkdir()
    metadata = {'frame_sha256': 'verified_a', 'source_hashes': {'1h': 'one', '4h': 'four'}}
    (tmp_path / 'data/development_wavelet_off.manifest.json').write_text(json.dumps(metadata))
    run = tmp_path / 'runs/A_example'
    run.mkdir(parents=True)
    (run / 'protocol.json').write_text(json.dumps({'seeds': [42], 'folds': [0], 'data_sha256': 'verified_a'}))
    status = {'status': 'running', 'completed': [{'seed': 42, 'fold': 0}]}
    (run / 'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError, match='No completed A'):
        pin_completed_a_reference(tmp_path)
    status['status'] = 'complete'
    (run / 'status.json').write_text(json.dumps(status))
    reference = pin_completed_a_reference(tmp_path)
    assert json.loads(reference.read_text()) == metadata
    (tmp_path / 'data/development_wavelet_off.manifest.json').write_text('{}')
    assert pin_completed_a_reference(tmp_path) == reference  # Immutable original reference.

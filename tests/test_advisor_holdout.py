import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from yenibot.training.advisor_holdout import audit_reservation, assert_fitting_scope


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/advisor_final_test.yaml').read_text())


def test_reserved_month_and_label_maturity():
    audit = audit_reservation(config())
    assert audit['prediction_timestamps'] == 720
    assert audit['first_sequence_context_timestamp'] == '2026-08-29T09:00:00+00:00'
    assert audit['required_last_1h_label_source_bar'] == '2026-10-01T09:00:00+00:00'
    assert audit['required_1h_download_end_exclusive'] == '2026-10-01T10:00:00+00:00'
    assert not audit['ready_for_test_evaluation'] and not audit['test_data_read']


@pytest.mark.parametrize('change', ['short_gap','bad_count','test_selection'])
def test_invalid_reservation_rejected(change):
    cfg = copy.deepcopy(config())
    if change == 'short_gap':
        cfg['boundaries']['pre_test_gap_bars'] = 6
    elif change == 'bad_count':
        cfg['test']['expected_prediction_timestamps'] = 657
    else:
        cfg['selection']['test_may_change_methods_or_thresholds'] = True
    with pytest.raises(ValueError):
        audit_reservation(cfg)


def test_fitting_guard_rejects_reserved_dates_and_misleading_label_end():
    cfg = config()
    sample = pd.Timestamp(cfg['boundaries']['latest_fitting_sample'])
    frame = pd.DataFrame({'timestamp': [sample], 'label_end_timestamp': [sample+pd.Timedelta(hours=10)]})
    assert_fitting_scope(frame,cfg)
    frame['timestamp'] = sample+pd.Timedelta(hours=1)
    frame['label_end_timestamp'] = frame.timestamp+pd.Timedelta(hours=10)
    with pytest.raises(ValueError,match='reserved pre-test gap'):
        assert_fitting_scope(frame,cfg)
    frame['timestamp'] = sample
    frame['label_end_timestamp'] = sample+pd.Timedelta(hours=1)
    with pytest.raises(ValueError,match='full forward-return horizon'):
        assert_fitting_scope(frame,cfg)

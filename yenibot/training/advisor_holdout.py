"""Reservation/preflight only. Does not read test data, train or score a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def audit_reservation(config: dict) -> dict:
    test = config['test']
    start, end = pd.Timestamp(test['start']), pd.Timestamp(test['end'])
    if start.tzinfo is None or end.tzinfo is None or start.utcoffset().total_seconds() or end.utcoffset().total_seconds():
        raise ValueError('Holdout timestamps must be UTC')
    if start != start.floor('h') or end != end.floor('h') or end < start:
        raise ValueError('Holdout bounds must be ordered whole hours')
    horizon, sequence = test['label_horizon_bars'], test['sequence_length']
    gap = config['boundaries']['pre_test_gap_bars']
    if horizon <= 0 or sequence <= 0 or gap < horizon:
        raise ValueError('Boundary gap must cover the positive label horizon')
    cutoff = pd.Timestamp(config['boundaries']['latest_fitting_sample'])
    if cutoff != start-pd.Timedelta(hours=gap+1):
        raise ValueError('Fitting cutoff must preserve exactly the declared pre-test gap')
    if cutoff+pd.Timedelta(hours=horizon) >= start:
        raise ValueError('Fitting label horizon overlaps test')
    expected = int((end-start)/pd.Timedelta(hours=1))+1
    if expected != test['expected_prediction_timestamps']:
        raise ValueError('Declared prediction count does not match holdout dates')
    if not config['reservation']['user_confirmed_no_prior_training_backtest_or_performance_review']:
        raise ValueError('Untouched-period confirmation missing')
    selection, execution = config['selection'], config['execution']
    if selection['test_may_change_methods_or_thresholds'] or selection['seed_selection_using_test']:
        raise ValueError('Test must not select methods, thresholds or seeds')
    if execution['test_fit_operations_allowed'] != 0 or execution['update_weights_during_test']:
        raise ValueError('Reserved fixed holdout forbids test fitting/weight updates')
    required_last = end+pd.Timedelta(hours=horizon)
    return {'status': config['status'], 'test_start': start.isoformat(), 'test_end': end.isoformat(),
            'prediction_timestamps': expected, 'latest_fitting_sample': cutoff.isoformat(),
            'first_sequence_context_timestamp': (start-pd.Timedelta(hours=sequence-1)).isoformat(),
            'required_last_1h_label_source_bar': required_last.isoformat(),
            'required_1h_download_end_exclusive': (required_last+pd.Timedelta(hours=1)).isoformat(),
            'ready_for_test_evaluation': False, 'reason': 'reservation only; final training and artifacts not frozen',
            'test_data_read': False, 'fit_operations': 0, 'test_evaluations': 0}


def assert_fitting_scope(frame: pd.DataFrame, config: dict) -> None:
    """Reusable guard for a future final-training adapter, not wired into A-D."""
    audit_reservation(config)
    if frame.empty or 'timestamp' not in frame or 'label_end_timestamp' not in frame:
        raise ValueError('Fitting data require sample and full label-end timestamps')
    times = pd.to_datetime(frame.timestamp, utc=True, errors='coerce')
    ends = pd.to_datetime(frame.label_end_timestamp, utc=True, errors='coerce')
    if times.isna().any() or ends.isna().any() or (ends < times).any():
        raise ValueError('Invalid fitting sample/label-end timestamps')
    conservative_end = times+pd.Timedelta(hours=config['test']['label_horizon_bars'])
    if (ends < conservative_end).any():
        raise ValueError('Fitting metadata must cover full forward-return horizon')
    if (times > pd.Timestamp(config['boundaries']['latest_fitting_sample'])).any():
        raise ValueError('Fitting samples enter reserved pre-test gap or test period')
    if (ends >= pd.Timestamp(config['test']['start'])).any():
        raise ValueError('Fitting labels overlap reserved test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/advisor_final_test.yaml'))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    result = audit_reservation(config)
    result['reservation_config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))

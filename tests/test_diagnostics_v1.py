"""
Unit tests for v1 diagnostics outputs.
Verifies that diagnostic JSON artifacts exist, reload cleanly, and contain expected fields.
"""

import os
import json
import pytest

DIAG_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'v1_submitted', 'diagnostics')


def test_diagnostics_files_exist_and_reload():
    expected_files = [
        'missed_episodes.json',
        'auc_by_label_group.json',
        'time_to_onset_distribution.json',
        'locked_trials_history.json',
        'diagnostics_summary.json'
    ]
    
    for fname in expected_files:
        fpath = os.path.join(DIAG_DIR, fname)
        assert os.path.exists(fpath), f"Expected diagnostic file {fpath} to exist"
        
        with open(fpath, 'r') as f:
            data = json.load(f)
        assert data is not None, f"Failed to reload JSON from {fpath}"
        assert len(data) > 0, f"JSON from {fpath} is unexpectedly empty"


def test_diagnostics_content_integrity():
    missed_fpath = os.path.join(DIAG_DIR, 'missed_episodes.json')
    with open(missed_fpath, 'r') as f:
        missed_data = json.load(f)
    assert missed_data['total_episodes'] == 18
    assert missed_data['missed_count'] == 6
    assert len(missed_data['missed_episodes']) == 6

    auc_fpath = os.path.join(DIAG_DIR, 'auc_by_label_group.json')
    with open(auc_fpath, 'r') as f:
        auc_data = json.load(f)
    assert 'validation' in auc_data
    assert 'test' in auc_data

    onset_fpath = os.path.join(DIAG_DIR, 'time_to_onset_distribution.json')
    with open(onset_fpath, 'r') as f:
        onset_data = json.load(f)
    assert len(onset_data) == 10  # 10 bins from -3.0s to +2.0s

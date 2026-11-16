"""
Unit tests for denominator reconciliation and false positive breakdown.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from reconcile_denominators import reconcile_denominators


def test_reconcile_denominators_proof():
    res = reconcile_denominators()
    
    # 1. Total windows consistency
    assert res['total_windows'] == 1462, f"Expected 1462 total windows, got {res['total_windows']}"
    assert res['label0_windows'] == 1098, f"Expected 1098 label-0 windows, got {res['label0_windows']}"
    assert res['positive_windows'] == 364, f"Expected 364 positive windows, got {res['positive_windows']}"
    assert res['label0_windows'] + res['positive_windows'] == res['total_windows']
    
    # 2. Raw false positives at 0.49 on label-0 windows
    assert res['raw_fp_label0_at_049'] == 220, f"Expected 220 raw FP on label-0, got {res['raw_fp_label0_at_049']}"
    
    # 3. Tail-end freeze windows proof
    assert res['tail_windows_count'] == 37, f"Expected exactly 37 tail windows, got {res['tail_windows_count']}"
    assert res['harness_positives_count'] == 257, f"Expected 257 harness positive windows, got {res['harness_positives_count']}"
    assert res['raw_fp_label0_at_049'] + res['tail_windows_count'] == res['harness_positives_count']
    
    # 4. Normal time denominators
    assert round(res['old_normal_duration_s'], 1) == 620.6, f"Expected 620.6s old normal time, got {res['old_normal_duration_s']}"
    assert round(res['new_normal_duration_s'], 1) == 580.6, f"Expected 580.6s new normal time, got {res['new_normal_duration_s']}"
    assert round(res['old_normal_duration_s'] - res['new_normal_duration_s'], 1) == 40.0

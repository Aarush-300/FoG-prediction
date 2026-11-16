"""
Unit tests for trivial baseline models and cluster bootstrap.
"""

import pytest
import numpy as np
from src.baselines import extract_handcrafted_features, cluster_bootstrap_auc, run_baseline
from src.train_cv import TEST_SUBJECTS, DEV_FOLDS


def test_handcrafted_feature_extraction_properties():
    # Synthetic windows: 10 windows of 100 samples and 6 channels
    rng = np.random.RandomState(42)
    X = rng.randn(10, 100, 6).astype(np.float32)
    feats = extract_handcrafted_features(X)

    assert feats.shape == (10, 14), f"Expected shape (10, 14), got {feats.shape}"
    assert not np.isnan(feats).any(), "NaN values detected in extracted features"
    assert not np.isinf(feats).any(), "Inf values detected in extracted features"


def test_cluster_bootstrap_structure():
    rng = np.random.RandomState(42)
    subjects = np.repeat([1, 2, 3, 4, 5], 20)  # 5 subjects, 20 windows each
    y_true = np.array([0, 1] * 50)
    y_pred = y_true * 0.8 + rng.uniform(0, 0.2, size=100)

    res = cluster_bootstrap_auc(subjects, y_true, y_pred, n_boot=50, seed=42)
    assert 'roc_auc' in res
    assert 'roc_ci' in res
    assert 'pr_auc' in res
    assert 'pr_ci' in res

    roc_lower, roc_upper = res['roc_ci']
    assert 0.0 <= roc_lower <= roc_upper <= 1.0


def test_baseline_test_isolation_enforcement():
    # If test subject is in metadata, must raise AssertionError
    bad_meta = [{'multi_label': 0, 'subjectID': 20, 'taskID': 1, 'start_time_s': 0.0}]
    X_dummy = np.zeros((1, 100, 6), dtype=np.float32)
    with pytest.raises(AssertionError) as exc_info:
        run_baseline('task_id', X_dummy, [0], bad_meta)
    assert "CRITICAL SECURITY VIOLATION" in str(exc_info.value)

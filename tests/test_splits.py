"""
Unit Tests for Dataset Splits and Leakage Verification.
Guarantees strict subject-independence and trial isolation across train, val, and test sets.
"""

import json
import pytest
import numpy as np

def test_subject_disjointness():
    """Verify strictly zero subject overlap between Train, Validation, and Test sets."""
    with open("data/processed/train_meta.json") as f:
        train_meta = json.load(f)
    with open("data/processed/val_meta.json") as f:
        val_meta = json.load(f)
    with open("data/processed/test_meta.json") as f:
        test_meta = json.load(f)
        
    train_subjects = {int(m['subjectID']) for m in train_meta}
    val_subjects = {int(m['subjectID']) for m in val_meta}
    test_subjects = {int(m['subjectID']) for m in test_meta}
    
    assert len(train_subjects) > 0
    assert len(val_subjects) > 0
    assert len(test_subjects) > 0
    
    # Strictly zero overlap
    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)
    
    assert train_val_overlap == set(), f"Subject leakage between Train and Val: {train_val_overlap}"
    assert train_test_overlap == set(), f"Subject leakage between Train and Test: {train_test_overlap}"
    assert val_test_overlap == set(), f"Subject leakage between Val and Test: {val_test_overlap}"

def test_trial_isolation():
    """Verify that no trial (subjectID, sessionID, taskID) appears in more than one split partition."""
    with open("data/processed/train_meta.json") as f:
        train_meta = json.load(f)
    with open("data/processed/val_meta.json") as f:
        val_meta = json.load(f)
    with open("data/processed/test_meta.json") as f:
        test_meta = json.load(f)
        
    train_trials = {(m['subjectID'], m['sessionID'], m['taskID']) for m in train_meta}
    val_trials = {(m['subjectID'], m['sessionID'], m['taskID']) for m in val_meta}
    test_trials = {(m['subjectID'], m['sessionID'], m['taskID']) for m in test_meta}
    
    assert train_trials.intersection(val_trials) == set()
    assert train_trials.intersection(test_trials) == set()
    assert val_trials.intersection(test_trials) == set()

def test_dataset_arrays_consistency():
    """Verify data array shapes and alignment between features and labels in npz."""
    data = np.load("data/processed/fog_dataset.npz")
    
    assert 'X_train' in data and 'y_train_bin' in data
    assert 'X_val' in data and 'y_val_bin' in data
    assert 'X_test' in data and 'y_test_bin' in data
    
    X_train, y_train = data['X_train'], data['y_train_bin']
    X_val, y_val = data['X_val'], data['y_val_bin']
    X_test, y_test = data['X_test'], data['y_test_bin']
    
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    
    assert X_train.shape[1:] == (100, 6)
    assert X_val.shape[1:] == (100, 6)
    assert X_test.shape[1:] == (100, 6)
    
    # Verify non-trivial class balance in all splits
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pos_rate = np.mean(y == 1)
        assert 0.05 < pos_rate < 0.50, f"{name} positive rate out of bounds: {pos_rate:.3f}"

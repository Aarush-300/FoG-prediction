"""
Subject Splits and Invariants for Phase 2 Cross-Validation.

Strictly protects held-out test subjects {18, 20, 21, 22} from leakage,
and defines deterministic, balanced 6-fold dev splits and inner validation sets.
"""

TEST_SUBJECTS = {18, 20, 21, 22}

# Stratified 6-fold subject grouping across 18 dev subjects (3 subjects per fold)
DEV_FOLDS = {
    0: [2, 8, 16],
    1: [19, 6, 17],
    2: [1, 7, 12],
    3: [3, 10, 4],
    4: [15, 14, 13],
    5: [11, 9, 5]
}

# Deterministic 2-subject inner validation sets for early stopping (disjoint from outer fold)
INNER_VAL_FOLDS = {
    0: [19, 6],
    1: [1, 7],
    2: [3, 10],
    3: [15, 14],
    4: [11, 8],
    5: [2, 16]
}


def verify_test_isolation(meta_list):
    """Fails immediately if any test subject is present."""
    subjs = set(m['subjectID'] for m in meta_list)
    overlap = subjs & TEST_SUBJECTS
    assert not overlap, f"CRITICAL SECURITY VIOLATION: Test subjects {overlap} found in development cohort!"

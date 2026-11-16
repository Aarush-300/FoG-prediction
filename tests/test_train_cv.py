"""
Unit tests for Phase 2 6-fold CV training isolation and harness discipline.
"""

import pytest
from src.train_cv import verify_test_isolation, DEV_FOLDS, TEST_SUBJECTS


def test_test_subject_isolation_assertion():
    # Clean dev metadata should pass
    clean_meta = [{'subjectID': s} for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]]
    verify_test_isolation(clean_meta)

    # Any test subject appearing MUST raise an AssertionError
    for test_s in TEST_SUBJECTS:
        dirty_meta = [{'subjectID': 1}, {'subjectID': test_s}]
        with pytest.raises(AssertionError) as exc_info:
            verify_test_isolation(dirty_meta)
        assert "CRITICAL SECURITY VIOLATION" in str(exc_info.value)


def test_dev_folds_disjointness_and_completeness():
    all_fold_subjs = []
    for f_idx, subjs in DEV_FOLDS.items():
        assert len(subjs) == 3, f"Fold {f_idx} must have exactly 3 subjects"
        assert not (set(subjs) & TEST_SUBJECTS), f"Fold {f_idx} contains test subjects!"
        all_fold_subjs.extend(subjs)

    # All 18 dev subjects present with no duplicates
    assert len(all_fold_subjs) == 18
    assert len(set(all_fold_subjs)) == 18
    expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19}
    assert set(all_fold_subjs) == expected

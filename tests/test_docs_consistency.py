"""
Unit tests verifying documentation consistency against data split configuration.

Validates that docs/model_evaluation.md and README.md accurately reflect the
subject-independent train, validation, and test partition splits defined in
data/processed/preprocess_config.json.
"""

import os
import re
import json
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'preprocess_config.json')
DOCS_EVAL_PATH = os.path.join(ROOT_DIR, 'docs', 'model_evaluation.md')
README_PATH = os.path.join(ROOT_DIR, 'README.md')


@pytest.fixture(scope="module")
def split_config():
    assert os.path.exists(CONFIG_PATH), f"Config file missing: {CONFIG_PATH}"
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_docs_model_evaluation_splits_match_config(split_config):
    """Verifies docs/model_evaluation.md partition definitions match preprocess_config.json."""
    assert os.path.exists(DOCS_EVAL_PATH), f"File missing: {DOCS_EVAL_PATH}"
    with open(DOCS_EVAL_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    expected_train = set(split_config['train_subjects'])
    expected_val = set(split_config['val_subjects'])
    expected_test = set(split_config['test_subjects'])

    # 1. Parse validation subjects from docs/model_evaluation.md
    val_match = re.search(r'Validation Partition[^\n]*:\*\*[^S]*Subjects?\s+([^\(\.\n]+)', content, re.IGNORECASE)
    assert val_match is not None, "Failed to find Validation Partition in docs/model_evaluation.md"
    doc_val_subjs = set(int(s) for s in re.findall(r'\b\d+\b', val_match.group(1)))
    assert doc_val_subjs == expected_val, f"Validation subjects mismatch in docs: {doc_val_subjs} vs {expected_val}"

    # 2. Parse test subjects from docs/model_evaluation.md
    test_match = re.search(r'Held-Out Test Partition[^\n]*:\*\*[^S]*Subjects?\s+([^\(\.\n]+)', content, re.IGNORECASE)
    assert test_match is not None, "Failed to find Test Partition in docs/model_evaluation.md"
    doc_test_subjs = set(int(s) for s in re.findall(r'\b\d+\b', test_match.group(1)))
    assert doc_test_subjs == expected_test, f"Test subjects mismatch in docs: {doc_test_subjs} vs {expected_test}"

    # 3. Parse train subjects from docs/model_evaluation.md
    train_match = re.search(r'Train Partition[^\n]*:\*\*[^S]*Subjects?\s+([^\(\.\n]+)', content, re.IGNORECASE)
    assert train_match is not None, "Failed to find Train Partition in docs/model_evaluation.md"
    doc_train_subjs = set(int(s) for s in re.findall(r'\b\d+\b', train_match.group(1)))
    assert doc_train_subjs == expected_train, f"Train subjects mismatch in docs: {doc_train_subjs} vs {expected_train}"

    # 4. Critical check: Subject 4 must NEVER be in validation
    assert 4 not in doc_val_subjs, "Subject 4 was erroneously placed in validation partition!"
    assert 4 in doc_train_subjs, "Subject 4 must be in training partition."

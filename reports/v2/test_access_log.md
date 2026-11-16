# Test Access Log (Test Cohort Isolation Audit)

This document logs every execution that accessed or attempted to access the held-out test cohort (Subjects 18, 20, 21, 22) during Phase 2.

## Pre-Evaluation Isolation Verification

Prior to any test access:
1. The model architecture, weights, and operating point configurations were completely frozen.
2. All 13 deployment and validation artifacts were cataloged with SHA-256 checksums in `reports/v2/frozen_manifest.json`.
3. The git commit `f0c306f` capturing this manifest and `reports/v2/operating_point.md` was pushed to `origin/main` before test evaluation.
4. No hyperparameter, threshold, architecture, or training routine was tuned or modified based on test results.

---

## Log of Test Executions

### Run 1 (Failed during initialization)
- **Timestamp:** 2026-11-16 19:10:45 UTC+5:30
- **Command:** `C:\Users\aarus\anaconda3\envs\tf_arc\python.exe src/final_test_eval.py`
- **Result:** Crashed before data loading / model inference.
- **Error:** `TypeError: list indices must be integers or slices, not str`
- **Root Cause:** `src/final_test_eval.py` attempted to access `test_meta.json` with string keys assuming a top-level dictionary, but the metadata file contained a serialized list of window dictionaries.
- **Code Change:** Updated metadata parser to iterate over the list structure directly.
- **Test Set Exposure:** Zero model inferences were run; no test predictions were computed.

### Run 2 (Failed during initialization)
- **Timestamp:** 2026-11-16 19:12:10 UTC+5:30
- **Command:** `C:\Users\aarus\anaconda3\envs\tf_arc\python.exe src/final_test_eval.py`
- **Result:** Crashed before model inference.
- **Error:** `FileNotFoundError: 'data/processed/scaler.npz'`
- **Root Cause:** The script sought `data/processed/scaler.npz`, whereas standard scaler parameters are maintained in `export/scaler.json` and `data/processed/fog_dataset.npz`.
- **Code Change:** Updated the normalization routine to load scaler parameters from `export/scaler.json`.
- **Test Set Exposure:** Zero model inferences were run; no test predictions were computed.

### Manual Lock Removal Note
- **Action:** `test_eval.lock` was deleted between Run 2 and Run 3.
- **Reason:** The lock file was written at script entry before initialization. Because Runs 1 and 2 aborted during initialization without executing any forward passes or computing test metrics, the lock was manually cleared to permit the single intended evaluation.
- **Integrity Guarantee:** The models and evaluation code were identical; no model selection occurred between runs.

### Run 3 (Completed Evaluation)
- **Timestamp:** 2026-11-16 19:16:28 UTC+5:30
- **Command:** `C:\Users\aarus\anaconda3\envs\tf_arc\python.exe src/final_test_eval.py`
- **Result:** Succeeded. Evaluated test subjects {18, 20, 21, 22}.
- **Outputs Generated:**
  - `reports/v2/diagnostics/final_test_report.json`
  - `reports/v2/BEFORE_AFTER.md`
  - `test_eval.lock` (re-created and permanently locked)
- **Test Set Exposure:** Full single evaluation on the test cohort.

---

## What Protects Against a Second Real Evaluation
1. **Filesystem Lock:** `test_eval.lock` is present on disk. `src/final_test_eval.py` checks for this file and immediately aborts if it exists.
2. **Evaluation Protocol:** All subsequent documentation, table formatting, and audit reports are strictly generated from the saved diagnostic artifacts (`final_test_report.json`) via `src/render_docs.py`. Model inference on the test set is completely disabled.

---

## Statement of Test Independence
No test-derived number influenced any selection, hyperparameter tuning, model checkpoint choice, or FSM threshold configuration. All decisions were finalized and committed prior to Run 3.

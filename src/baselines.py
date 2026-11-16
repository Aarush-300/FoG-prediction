"""
Trivial Baselines for Phase 2 Experiment (g) Early Signal Check.

Evaluates three pre-registered baseline models under the exact same 6-fold
subject-grouped cross-validation and cluster bootstrap:
(a) Logistic regression on taskID only (one-hot encoded)
(b) Logistic regression on taskID + time since trial start (start_time_s)
(c) Logistic regression on 14 hand-crafted kinematic features
    (accel/gyro std, mean abs gyro, dominant frequency, 3-8 Hz / 0.5-3 Hz band ratio)
"""

import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from cv_splits import DEV_FOLDS, verify_test_isolation, TEST_SUBJECTS


def extract_handcrafted_features(X_raw, fs=50.0):
    """
    Extracts 14 hand-crafted kinematic features from IMU windows (N, 100, 6):
    - Accel std: 3 axes + magnitude (4)
    - Gyro std: 3 axes + magnitude (4)
    - Mean absolute gyro: 3 axes + magnitude (4)
    - Dominant frequency of acc magnitude (1)
    - Freeze Index: 3-8 Hz / (0.5-3 Hz + 1e-6) on acc magnitude (1)
    """
    acc = X_raw[:, :, 0:3]
    gyro = X_raw[:, :, 3:6]
    acc_mag = np.linalg.norm(acc, axis=-1)
    gyro_mag = np.linalg.norm(gyro, axis=-1)

    acc_std = np.std(acc, axis=1)  # (N, 3)
    acc_mag_std = np.std(acc_mag, axis=1, keepdims=True)  # (N, 1)

    gyro_std = np.std(gyro, axis=1)  # (N, 3)
    gyro_mag_std = np.std(gyro_mag, axis=1, keepdims=True)  # (N, 1)

    gyro_mean_abs = np.mean(np.abs(gyro), axis=1)  # (N, 3)
    gyro_mag_mean_abs = np.mean(np.abs(gyro_mag), axis=1, keepdims=True)  # (N, 1)

    W = X_raw.shape[1]
    acc_mag_detrend = acc_mag - np.mean(acc_mag, axis=1, keepdims=True)
    fft_vals = np.fft.rfft(acc_mag_detrend, axis=1)
    fft_power = (np.abs(fft_vals) ** 2) / W
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)

    loco_mask = (freqs >= 0.5) & (freqs <= 3.0)
    freeze_mask = (freqs > 3.0) & (freqs <= 8.0)
    valid_mask = (freqs >= 0.5) & (freqs <= 20.0)

    loco_power = np.sum(fft_power[:, loco_mask], axis=1, keepdims=True)
    freeze_power = np.sum(fft_power[:, freeze_mask], axis=1, keepdims=True)
    freeze_index = freeze_power / (loco_power + 1e-6)

    valid_freqs = freqs[valid_mask]
    valid_fft = fft_power[:, valid_mask]
    dom_freq_idx = np.argmax(valid_fft, axis=1)
    dom_freq = valid_freqs[dom_freq_idx][:, np.newaxis]

    feats = np.hstack([
        acc_std, acc_mag_std,
        gyro_std, gyro_mag_std,
        gyro_mean_abs, gyro_mag_mean_abs,
        dom_freq, freeze_index
    ])
    return feats.astype(np.float32)


def cluster_bootstrap_auc(subjects, y_true, y_pred, n_boot=1000, seed=42):
    """
    Computes ROC-AUC and PR-AUC with 95% cluster-bootstrap confidence intervals
    resampling whole subjects with replacement.
    """
    rng = np.random.RandomState(seed)
    unique_subjs = np.unique(subjects)
    subj_indices = {s: np.where(subjects == s)[0] for s in unique_subjs}

    boot_roc = []
    boot_pr = []
    for _ in range(n_boot):
        sampled_subjs = rng.choice(unique_subjs, size=len(unique_subjs), replace=True)
        idx_boot = np.concatenate([subj_indices[s] for s in sampled_subjs])
        y_t = y_true[idx_boot]
        y_p = y_pred[idx_boot]
        if len(np.unique(y_t)) < 2:
            continue
        boot_roc.append(roc_auc_score(y_t, y_p))
        boot_pr.append(average_precision_score(y_t, y_p))

    roc_ci = (float(np.percentile(boot_roc, 2.5)), float(np.percentile(boot_roc, 97.5)))
    pr_ci = (float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5)))
    return {
        'roc_auc': float(roc_auc_score(y_true, y_pred)),
        'roc_ci': roc_ci,
        'pr_auc': float(average_precision_score(y_true, y_pred)),
        'pr_ci': pr_ci
    }


def run_baseline(baseline_type, X_dev_raw, y_multi_dev, meta_dev, dev_folds=DEV_FOLDS):
    """
    Runs 6-fold CV for a given baseline model on Pure Pre-FoG (label 1) vs Normal (label 0).
    baseline_type: 'task_id', 'task_time', or 'handcrafted'
    """
    verify_test_isolation(meta_dev)

    # Filter to label 0 and label 1
    valid_mask = np.array([m['multi_label'] in [0, 1] for m in meta_dev])
    indices_valid = np.where(valid_mask)[0]

    subjs_valid = np.array([meta_dev[i]['subjectID'] for i in indices_valid])
    y_valid = np.array([1 if meta_dev[i]['multi_label'] == 1 else 0 for i in indices_valid], dtype=np.float32)
    tasks_valid = np.array([meta_dev[i]['taskID'] for i in indices_valid])
    times_valid = np.array([meta_dev[i]['start_time_s'] for i in indices_valid], dtype=np.float32)

    # Pre-extract features if handcrafted
    if baseline_type == 'handcrafted':
        X_feats_all = extract_handcrafted_features(X_dev_raw[indices_valid])
    elif baseline_type == 'task_id':
        ohe = OneHotEncoder(sparse_output=False, categories=[list(range(1, 8))])
        X_feats_all = ohe.fit_transform(tasks_valid.reshape(-1, 1))
    elif baseline_type == 'task_time':
        ohe = OneHotEncoder(sparse_output=False, categories=[list(range(1, 8))])
        task_ohe = ohe.fit_transform(tasks_valid.reshape(-1, 1))
        X_feats_all = np.hstack([task_ohe, times_valid[:, np.newaxis]])
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    oof_probs = np.zeros(len(indices_valid), dtype=np.float32)
    fold_metrics = {}

    for f_idx in range(6):
        outer_subjs = set(dev_folds[f_idx])
        train_mask = ~np.isin(subjs_valid, list(outer_subjs))
        outer_mask = np.isin(subjs_valid, list(outer_subjs))

        X_train = X_feats_all[train_mask]
        y_train = y_valid[train_mask]
        X_outer = X_feats_all[outer_mask]
        y_outer = y_valid[outer_mask]

        if baseline_type in ['handcrafted', 'task_time']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_outer = scaler.transform(X_outer)

        clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        probs_outer = clf.predict_proba(X_outer)[:, 1]
        oof_probs[outer_mask] = probs_outer

        f_roc = float(roc_auc_score(y_outer, probs_outer))
        f_pr = float(average_precision_score(y_outer, probs_outer))
        fold_metrics[f_idx] = {'roc_auc': f_roc, 'pr_auc': f_pr, 'n_windows': len(y_outer), 'n_pos': int(y_outer.sum())}

    # Pooled metrics with 1,000 cluster bootstrap
    boot_res = cluster_bootstrap_auc(subjs_valid, y_valid, oof_probs, n_boot=1000, seed=42)

    return {
        'baseline_type': baseline_type,
        'fold_metrics': fold_metrics,
        'pooled_metrics': boot_res,
        'oof_probs': oof_probs,
        'y_true': y_valid,
        'subjects': subjs_valid,
        'indices': indices_valid
    }


def evaluate_all_baselines(X_dev_raw, y_multi_dev, meta_dev, dev_folds=DEV_FOLDS):
    results = {}
    for b_type in ['task_id', 'task_time', 'handcrafted']:
        print(f"Running baseline: {b_type}...")
        res = run_baseline(b_type, X_dev_raw, y_multi_dev, meta_dev, dev_folds=dev_folds)
        results[b_type] = res
        p = res['pooled_metrics']
        print(f"  [{b_type}] Pooled ROC-AUC: {p['roc_auc']:.4f} (95% CI: [{p['roc_ci'][0]:.4f}, {p['roc_ci'][1]:.4f}]) | PR-AUC: {p['pr_auc']:.4f}")
    return results

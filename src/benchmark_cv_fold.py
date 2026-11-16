"""
Benchmark CV Fold Timing for Phase 2.

Measures the exact training time and seconds per epoch of Improved_ResCNN
on a single 6-fold CV fold with:
- 13 training subjects (scaled on training subjects only)
- 2 deterministic inner validation subjects (used strictly for EarlyStopping)
- 3 outer held-out subjects (used strictly for OOF evaluation)
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from cv_splits import TEST_SUBJECTS, DEV_FOLDS, INNER_VAL_FOLDS
from models_zoo import build_improved_res_cnn
from train_experiment import augment_training_windows_physical


def benchmark_single_fold(fold_idx=0, epochs=15, batch_size=64, seed=42):
    ds = np.load('data/processed/fog_dataset.npz')
    with open('data/processed/train_meta.json') as f:
        train_meta = json.load(f)
    with open('data/processed/val_meta.json') as f:
        val_meta = json.load(f)

    meta_dev = train_meta + val_meta
    subjs = set(m['subjectID'] for m in meta_dev)
    assert not (subjs & TEST_SUBJECTS), f"Test subject found: {subjs & TEST_SUBJECTS}"

    outer_held_out = set(DEV_FOLDS[fold_idx])
    inner_val = set(INNER_VAL_FOLDS[fold_idx])
    assert not (outer_held_out & inner_val), "CRITICAL: Inner val and outer held-out overlap!"

    # Training subjects: remaining 13 subjects
    train_subjs = subjs - outer_held_out - inner_val
    assert len(train_subjs) == 13
    assert len(inner_val) == 2
    assert len(outer_held_out) == 3

    X_dev_raw = np.concatenate([ds['X_train_raw'], ds['X_val_raw']], axis=0)
    y_bin_dev = np.concatenate([ds['y_train_bin'], ds['y_val_bin']], axis=0)

    train_mask = np.array([m['subjectID'] in train_subjs for m in meta_dev])
    inner_val_mask = np.array([m['subjectID'] in inner_val for m in meta_dev])
    outer_oof_mask = np.array([m['subjectID'] in outer_held_out for m in meta_dev])

    X_train_raw = X_dev_raw[train_mask]
    y_train = y_bin_dev[train_mask]
    X_inner_raw = X_dev_raw[inner_val_mask]
    y_inner = y_bin_dev[inner_val_mask]
    X_outer_raw = X_dev_raw[outer_oof_mask]
    y_outer = y_bin_dev[outer_oof_mask]

    # Scaler fit STRICTLY on training subjects only
    mean = np.mean(X_train_raw, axis=(0, 1), keepdims=True)
    std = np.std(X_train_raw, axis=(0, 1), keepdims=True) + 1e-7

    X_train_norm = (X_train_raw - mean) / std
    X_inner_norm = (X_inner_raw - mean) / std
    X_outer_norm = (X_outer_raw - mean) / std

    # Data augmentation on train
    t_aug_start = time.time()
    X_train_aug = augment_training_windows_physical(X_train_raw, y_train, mean, std, max_angle_deg=35.0)
    X_train_full = np.concatenate([X_train_norm, X_train_aug], axis=0)
    y_train_full = np.concatenate([y_train, y_train], axis=0)
    t_aug_end = time.time()

    np.random.seed(seed)
    perm = np.random.permutation(len(X_train_full))
    X_train_full, y_train_full = X_train_full[perm], y_train_full[perm]

    n_neg = int((y_train_full == 0).sum())
    n_pos = int((y_train_full == 1).sum())
    class_weights = {
        0: float((1 / n_neg) * (len(y_train_full) / 2.0)),
        1: float((1 / n_pos) * (len(y_train_full) / 2.0))
    }

    class TimeHistory(callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.epoch_times = []
        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_start = time.time()
        def on_epoch_end(self, epoch, logs={}):
            self.epoch_times.append(time.time() - self.epoch_start)

    time_cb = TimeHistory()

    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    with tf.device('/CPU:0'):
        model = build_improved_res_cnn(input_shape=(100, 6))
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                tf.keras.metrics.AUC(curve='ROC', name='roc_auc')
            ]
        )

        t0 = time.time()
        model.fit(
            X_train_full, y_train_full,
            validation_data=(X_inner_norm, y_inner),  # Inner val for EarlyStopping
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=5, restore_best_weights=True),
                time_cb
            ],
            verbose=1
        )
        t_total = time.time() - t0

    epoch_times = time_cb.epoch_times
    results = {
        'epochs_trained': len(epoch_times),
        'seconds_per_epoch_mean': float(np.mean(epoch_times)),
        'seconds_per_epoch_min': float(np.min(epoch_times)),
        'seconds_per_epoch_max': float(np.max(epoch_times)),
        'train_time_s': float(t_total),
        'aug_time_s': float(t_aug_end - t_aug_start),
        'total_fold_time_s': float(t_total + (t_aug_end - t_aug_start)),
        'train_windows': len(X_train_full),
        'inner_val_windows': len(X_inner_norm),
        'outer_oof_windows': len(X_outer_norm)
    }

    print("\n" + "=" * 60)
    print("NEW INNER-VALIDATION SETUP TIMING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Train subjects (13): {sorted(list(train_subjs))} ({results['train_windows']} windows)")
    print(f"Inner val subjects (2, EarlyStopping): {sorted(list(inner_val))} ({results['inner_val_windows']} windows)")
    print(f"Outer OOF subjects (3, Held-out): {sorted(list(outer_held_out))} ({results['outer_oof_windows']} windows)")
    print(f"Epochs trained: {results['epochs_trained']}")
    print(f"Seconds per epoch: mean = {results['seconds_per_epoch_mean']:.2f}s (min = {results['seconds_per_epoch_min']:.2f}s, max = {results['seconds_per_epoch_max']:.2f}s)")
    print(f"Fold 0 training time: {results['train_time_s']:.2f}s ({results['train_time_s']/60:.2f} min)")
    print(f"Total Fold time (with augmentation): {results['total_fold_time_s']:.2f}s ({results['total_fold_time_s']/60:.2f} min)")
    return results


if __name__ == '__main__':
    benchmark_single_fold()

"""
6-Fold Subject-Grouped Cross-Validation Training Harness for Phase 2.

Implements all Pre-Registration Rules & Protocols:
1. Strict test isolation: Asserts subjects {18, 20, 21, 22} never appear.
2. Disjoint 6-fold subject-grouped splits on 18 pooled dev subjects.
3. Inner-validation early-stopping: 2 deterministic inner validation subjects
   used for EarlyStopping/ModelCheckpointing; outer 3 subjects used strictly for OOF evaluation.
4. Scaler fit strictly on training fold subjects (13 subjects) only.
5. Resumable checkpointing: Saves fold models to models/v2_dev/ and
   out-of-fold (OOF) predictions to reports/v2/oof/ immediately upon completion.
6. Experiment configurations:
   - 'g_signal_check': Pure Pre-FoG (label 1) vs Normal (label 0) only.
   - 'b_rotation': 3D Euler rotation range {35, 50, 70}.
   - 'e_prefog_weights': Pre-FoG window loss weighting {1x, 2x, 4x}.
   - 'f_labels': Label definition variants (v1, drop -1, clean positives).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from cv_splits import TEST_SUBJECTS, DEV_FOLDS, INNER_VAL_FOLDS, verify_test_isolation
from models_zoo import build_improved_res_cnn
from train_experiment import augment_training_windows_physical
from baselines import extract_handcrafted_features, cluster_bootstrap_auc, evaluate_all_baselines


def load_dev_data(dataset_npz="data/processed/fog_dataset.npz",
                  train_meta_json="data/processed/train_meta.json",
                  val_meta_json="data/processed/val_meta.json"):
    ds = np.load(dataset_npz)
    with open(train_meta_json) as f:
        train_meta = json.load(f)
    with open(val_meta_json) as f:
        val_meta = json.load(f)

    meta_dev = train_meta + val_meta
    verify_test_isolation(meta_dev)

    X_dev_raw = np.concatenate([ds['X_train_raw'], ds['X_val_raw']], axis=0)
    y_bin_dev = np.concatenate([ds['y_train_bin'], ds['y_val_bin']], axis=0)
    y_multi_dev = np.concatenate([ds['y_train_multi'], ds['y_val_multi']], axis=0)

    assert len(X_dev_raw) == len(meta_dev)
    return X_dev_raw, y_bin_dev, y_multi_dev, meta_dev


def run_experiment_fold(exp_id, seed, fold_idx,
                        X_dev_raw, y_bin_dev, y_multi_dev, meta_dev,
                        models_dir="models/v2_dev",
                        oof_dir="reports/v2/oof",
                        epochs=40, batch_size=64,
                        max_angle_deg=35.0, prefog_weight=1.0):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{exp_id}_seed{seed}_fold{fold_idx}.h5")
    oof_path = os.path.join(oof_dir, f"{exp_id}_seed{seed}_fold{fold_idx}.npz")

    # Resumability check: if already trained, skip
    if os.path.exists(model_path) and os.path.exists(oof_path):
        print(f"[RESUME] Fold {fold_idx} (Seed {seed}, Exp '{exp_id}') already exists. Loading OOF...")
        data = np.load(oof_path)
        return data['val_indices'], data['val_probs'], data['val_y_true'], float(data['train_time_s'])

    print(f"\n[TRAIN] Starting Fold {fold_idx}/5 | Seed {seed} | Exp '{exp_id}'...")
    outer_subjs = set(DEV_FOLDS[fold_idx])
    inner_subjs = set(INNER_VAL_FOLDS[fold_idx])
    all_dev_subjs = set(m['subjectID'] for m in meta_dev)
    train_subjs = all_dev_subjs - outer_subjs - inner_subjs

    assert len(outer_subjs) == 3, f"Fold {fold_idx}: outer_subjs must have 3 subjects"
    assert len(inner_subjs) == 2, f"Fold {fold_idx}: inner_subjs must have 2 subjects"
    assert len(train_subjs) == 13, f"Fold {fold_idx}: train_subjs must have 13 subjects"
    assert not (outer_subjs & inner_subjs), f"Fold {fold_idx}: outer and inner overlap!"
    assert not (outer_subjs & TEST_SUBJECTS), f"Fold {fold_idx}: test subject in outer!"
    assert not (inner_subjs & TEST_SUBJECTS), f"Fold {fold_idx}: test subject in inner!"

    # Subject disjoint splits
    train_indices = [i for i, m in enumerate(meta_dev) if m['subjectID'] in train_subjs]
    inner_indices = [i for i, m in enumerate(meta_dev) if m['subjectID'] in inner_subjs]
    outer_indices = [i for i, m in enumerate(meta_dev) if m['subjectID'] in outer_subjs]

    # Filter data based on experiment definition
    if exp_id.startswith('g_signal_check'):
        # Pure Pre-FoG (label 1) vs Normal (label 0) ONLY. Drop -1 and 2.
        train_indices = [i for i in train_indices if y_multi_dev[i] in [0, 1]]
        inner_indices = [i for i in inner_indices if y_multi_dev[i] in [0, 1]]
        outer_indices = [i for i in outer_indices if y_multi_dev[i] in [0, 1]]
        y_train = (y_multi_dev[train_indices] == 1).astype(np.float32)
        y_inner = (y_multi_dev[inner_indices] == 1).astype(np.float32)
        y_outer = (y_multi_dev[outer_indices] == 1).astype(np.float32)
    elif exp_id == 'f_drop_transition':
        # Drop transition (-1) from training
        train_indices = [i for i in train_indices if y_multi_dev[i] != -1]
        inner_indices = [i for i in inner_indices if y_multi_dev[i] != -1]
        outer_indices = [i for i in outer_indices if y_multi_dev[i] != -1]
        y_train = y_bin_dev[train_indices].astype(np.float32)
        y_inner = y_bin_dev[inner_indices].astype(np.float32)
        y_outer = y_bin_dev[outer_indices].astype(np.float32)
    elif exp_id == 'f_clean_positives':
        # Positives = pure pre-FoG (1) + active FoG (2) only. Drop -1.
        train_indices = [i for i in train_indices if y_multi_dev[i] in [0, 1, 2]]
        inner_indices = [i for i in inner_indices if y_multi_dev[i] in [0, 1, 2]]
        outer_indices = [i for i in outer_indices if y_multi_dev[i] in [0, 1, 2]]
        y_train = (np.isin(y_multi_dev[train_indices], [1, 2])).astype(np.float32)
        y_inner = (np.isin(y_multi_dev[inner_indices], [1, 2])).astype(np.float32)
        y_outer = (np.isin(y_multi_dev[outer_indices], [1, 2])).astype(np.float32)
    else:
        # Standard v1 binary task
        y_train = y_bin_dev[train_indices].astype(np.float32)
        y_inner = y_bin_dev[inner_indices].astype(np.float32)
        y_outer = y_bin_dev[outer_indices].astype(np.float32)

    X_train_raw = X_dev_raw[train_indices]
    X_inner_raw = X_dev_raw[inner_indices]
    X_outer_raw = X_dev_raw[outer_indices]

    # Scaler fit STRICTLY on 13 training subjects only
    mean = np.mean(X_train_raw, axis=(0, 1), keepdims=True)
    std = np.std(X_train_raw, axis=(0, 1), keepdims=True) + 1e-7

    X_train_norm = (X_train_raw - mean) / std
    X_inner_norm = (X_inner_raw - mean) / std
    X_outer_norm = (X_outer_raw - mean) / std

    # Physical data augmentation on raw training fold
    X_train_aug = augment_training_windows_physical(X_train_raw, y_train, mean, std, max_angle_deg=max_angle_deg)
    X_train_full = np.concatenate([X_train_norm, X_train_aug], axis=0)
    y_train_full = np.concatenate([y_train, y_train], axis=0)

    # Shuffling
    np.random.seed(seed + fold_idx * 100)
    perm = np.random.permutation(len(X_train_full))
    X_train_full, y_train_full = X_train_full[perm], y_train_full[perm]

    # Class weights on augmented train
    n_neg = int((y_train_full == 0).sum())
    n_pos = int((y_train_full == 1).sum())
    w0 = float((1 / n_neg) * (len(y_train_full) / 2.0))
    w1 = float((1 / n_pos) * (len(y_train_full) / 2.0))
    class_weights = {0: w0, 1: w1}

    # Sample weights for pre-FoG weighting experiment
    sample_weights = None
    if prefog_weight != 1.0 and not exp_id.startswith('g_signal_check'):
        train_multi_full = np.concatenate([y_multi_dev[train_indices], y_multi_dev[train_indices]])[perm]
        sample_weights = np.ones(len(y_train_full), dtype=np.float32)
        sample_weights[y_train_full == 0] = w0
        sample_weights[y_train_full == 1] = w1
        sample_weights[train_multi_full == 1] *= float(prefog_weight)
        class_weights = None

    # Build and compile model
    tf.keras.backend.clear_session()
    np.random.seed(seed + fold_idx)
    tf.random.set_seed(seed + fold_idx)

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

        cb = [
            callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=6, restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=3, min_lr=1e-5, verbose=0)
        ]

        t_start = time.time()
        model.fit(
            X_train_full, y_train_full,
            validation_data=(X_inner_norm, y_inner),  # Inner val for EarlyStopping / Model selection
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            sample_weight=sample_weights,
            callbacks=cb,
            verbose=0
        )
        t_elapsed = time.time() - t_start

        # Pure Out-of-fold inference on outer held-out fold
        val_probs = model.predict(X_outer_norm, verbose=0).ravel()

        # Save trained fold model & scaler
        model.save(model_path)
        np.savez_compressed(
            oof_path,
            val_indices=np.array(outer_indices, dtype=int),
            val_probs=val_probs,
            val_y_true=y_outer,
            train_time_s=t_elapsed,
            scaler_mean=mean,
            scaler_std=std
        )

    outer_roc = float(roc_auc_score(y_outer, val_probs))
    outer_pr = float(average_precision_score(y_outer, val_probs))
    print(f"[DONE] Fold {fold_idx} completed in {t_elapsed:.1f} s | Outer OOF ROC: {outer_roc:.4f} | PR: {outer_pr:.4f}")
    return np.array(outer_indices, dtype=int), val_probs, y_outer, t_elapsed


def run_experiment(exp_id, seeds=[42], max_angle_deg=35.0, prefog_weight=1.0, epochs=40):
    print("=" * 70)
    print(f"RUNNING EXPERIMENT: {exp_id} across seeds {seeds}")
    print("=" * 70)
    X_dev_raw, y_bin_dev, y_multi_dev, meta_dev = load_dev_data()

    all_seed_results = []

    for seed in seeds:
        print(f"\n>>> SEED {seed} <<<")
        seed_oof_probs = {}
        seed_oof_true = {}
        fold_metrics = {}
        total_time = 0.0

        for fold_idx in range(6):
            val_idx, val_probs, val_y, t_sec = run_experiment_fold(
                exp_id=exp_id,
                seed=seed,
                fold_idx=fold_idx,
                X_dev_raw=X_dev_raw,
                y_bin_dev=y_bin_dev,
                y_multi_dev=y_multi_dev,
                meta_dev=meta_dev,
                epochs=epochs,
                max_angle_deg=max_angle_deg,
                prefog_weight=prefog_weight
            )
            total_time += t_sec
            for vi, p, y in zip(val_idx, val_probs, val_y):
                seed_oof_probs[vi] = p
                seed_oof_true[vi] = y

            f_roc = float(roc_auc_score(val_y, val_probs))
            f_pr = float(average_precision_score(val_y, val_probs))
            fold_metrics[fold_idx] = {'roc_auc': f_roc, 'pr_auc': f_pr, 'n_windows': len(val_y), 'n_pos': int(val_y.sum())}

        sorted_indices = sorted(seed_oof_probs.keys())
        y_oof_true = np.array([seed_oof_true[i] for i in sorted_indices])
        y_oof_prob = np.array([seed_oof_probs[i] for i in sorted_indices])
        subjs_oof = np.array([meta_dev[i]['subjectID'] for i in sorted_indices])

        boot_res = cluster_bootstrap_auc(subjs_oof, y_oof_true, y_oof_prob, n_boot=1000, seed=seed)
        roc = boot_res['roc_auc']
        roc_ci = boot_res['roc_ci']
        pr = boot_res['pr_auc']
        pr_ci = boot_res['pr_ci']

        print(f"\n[SUMMARY] Seed {seed} Pooled OOF: ROC-AUC = {roc:.4f} (95% CI: [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]) | PR-AUC = {pr:.4f} (95% CI: [{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]) | Total Time: {total_time/60:.2f} min")
        all_seed_results.append({
            'seed': seed,
            'roc_auc': roc,
            'roc_ci': roc_ci,
            'pr_auc': pr,
            'pr_ci': pr_ci,
            'fold_metrics': fold_metrics,
            'total_time_s': total_time,
            'sorted_indices': sorted_indices,
            'y_oof_prob': y_oof_prob,
            'y_oof_true': y_oof_true
        })

    # If this is the signal check experiment, evaluate baselines and early signal diagnostics
    if exp_id.startswith('g_signal_check'):
        print("\n" + "=" * 70)
        print("EVALUATING TRIVIAL BASELINES FOR EXPERIMENT (g)...")
        print("=" * 70)
        baseline_results = evaluate_all_baselines(X_dev_raw, y_multi_dev, meta_dev, dev_folds=DEV_FOLDS)

        # Save summary JSON
        summary_out = {
            'rescnn_seeds': [
                {
                    'seed': r['seed'],
                    'roc_auc': r['roc_auc'],
                    'roc_ci': r['roc_ci'],
                    'pr_auc': r['pr_auc'],
                    'pr_ci': r['pr_ci'],
                    'fold_metrics': r['fold_metrics']
                } for r in all_seed_results
            ],
            'baselines': {
                b_name: {
                    'pooled_metrics': b_data['pooled_metrics'],
                    'fold_metrics': b_data['fold_metrics']
                } for b_name, b_data in baseline_results.items()
            }
        }
        os.makedirs('reports/v2/diagnostics', exist_ok=True)
        with open('reports/v2/diagnostics/experiment_g_results.json', 'w') as f:
            json.dump(summary_out, f, indent=2)
        print("\nSaved Experiment (g) results to reports/v2/diagnostics/experiment_g_results.json")

    return all_seed_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True, choices=[
        'g_signal_check', 'b_rotation_35', 'b_rotation_50', 'b_rotation_70',
        'e_prefog_1x', 'e_prefog_2x', 'e_prefog_4x',
        'f_labels_v1', 'f_drop_transition', 'f_clean_positives'
    ])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--max_angle_deg', type=float, default=35.0)
    parser.add_argument('--prefog_weight', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    run_experiment(
        exp_id=args.exp_id,
        seeds=args.seeds,
        max_angle_deg=args.max_angle_deg,
        prefog_weight=args.prefog_weight,
        epochs=args.epochs
    )

"""
Execution and Full Evaluation of Phase 2 Experiment (g): Early Signal Check.

Trains Improved_ResCNN across 6-fold CV on 3 seeds (42, 43, 44) strictly on
Pure Pre-FoG (label 1) vs Normal Gait (label 0), and compares against three
trivial baselines:
(a) Logistic regression on taskID only
(b) Logistic regression on taskID + start_time_s
(c) Logistic regression on 14 hand-crafted kinematic features

Computes:
1. Per-fold and pooled ROC-AUC and PR-AUC with 1,000 cluster-bootstrap 95% CIs
2. Mean OOF probability vs time-to-onset bins ([-3.0, -2.5) to [1.5, 2.0))
3. Per-episode early-signal table (max EMA in [onset-5.0s, onset-0.5s])
4. Pre-registered GO / RE-SCOPE decision check.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cv_splits import DEV_FOLDS, INNER_VAL_FOLDS, TEST_SUBJECTS, verify_test_isolation
from train_cv import load_dev_data, run_experiment_fold
from baselines import evaluate_all_baselines, cluster_bootstrap_auc
from eval_pipeline import extract_trial_records


def run_experiment_g_full(seeds=[42, 43, 44], epochs=30, batch_size=64):
    print("=" * 80)
    print("STARTING EXPERIMENT (g): EARLY SIGNAL CHECK (SEEDS: {})".format(seeds))
    print("=" * 80)

    X_dev_raw, y_bin_dev, y_multi_dev, meta_dev = load_dev_data()
    verify_test_isolation(meta_dev)

    # 1. Train Improved_ResCNN across 6 folds for each seed (or load existing checkpoints)
    rescnn_seed_runs = []
    for seed in seeds:
        print(f"\n==================== RESCNN SEED {seed} ====================")
        seed_oof_probs = {}
        seed_oof_true = {}
        seed_fold_metrics = {}
        total_time = 0.0

        for f_idx in range(6):
            val_idx, val_probs, val_y, t_sec = run_experiment_fold(
                exp_id='g_signal_check',
                seed=seed,
                fold_idx=f_idx,
                X_dev_raw=X_dev_raw,
                y_bin_dev=y_bin_dev,
                y_multi_dev=y_multi_dev,
                meta_dev=meta_dev,
                epochs=epochs,
                batch_size=batch_size
            )
            total_time += t_sec
            for vi, p, y in zip(val_idx, val_probs, val_y):
                seed_oof_probs[vi] = p
                seed_oof_true[vi] = y

            f_roc = float(roc_auc_score(val_y, val_probs))
            f_pr = float(average_precision_score(val_y, val_probs))
            seed_fold_metrics[f_idx] = {
                'roc_auc': f_roc,
                'pr_auc': f_pr,
                'n_windows': len(val_y),
                'n_pos': int(val_y.sum())
            }

        sorted_idx = sorted(seed_oof_probs.keys())
        y_true_sorted = np.array([seed_oof_true[i] for i in sorted_idx])
        y_prob_sorted = np.array([seed_oof_probs[i] for i in sorted_idx])
        subjs_sorted = np.array([meta_dev[i]['subjectID'] for i in sorted_idx])

        # 1,000 cluster bootstrap resampling whole subjects
        boot_res = cluster_bootstrap_auc(subjs_sorted, y_true_sorted, y_prob_sorted, n_boot=1000, seed=seed)
        print(f"\n[RESCNN Seed {seed}] Pooled ROC-AUC: {boot_res['roc_auc']:.4f} (95% CI: [{boot_res['roc_ci'][0]:.4f}, {boot_res['roc_ci'][1]:.4f}]) | PR-AUC: {boot_res['pr_auc']:.4f} (95% CI: [{boot_res['pr_ci'][0]:.4f}, {boot_res['pr_ci'][1]:.4f}]) | Time: {total_time/60:.2f} min")

        rescnn_seed_runs.append({
            'seed': seed,
            'pooled_metrics': boot_res,
            'fold_metrics': seed_fold_metrics,
            'sorted_indices': sorted_idx,
            'y_prob': y_prob_sorted,
            'y_true': y_true_sorted,
            'subjects': subjs_sorted,
            'train_time_s': total_time
        })

    # Average ResCNN predictions across the 3 seeds
    avg_oof_prob = np.mean([r['y_prob'] for r in rescnn_seed_runs], axis=0)
    primary_indices = rescnn_seed_runs[0]['sorted_indices']
    primary_true = rescnn_seed_runs[0]['y_true']
    primary_subjs = rescnn_seed_runs[0]['subjects']

    ensemble_boot = cluster_bootstrap_auc(primary_subjs, primary_true, avg_oof_prob, n_boot=1000, seed=42)
    print(f"\n==================== RESCNN 3-SEED ENSEMBLE ====================")
    print(f"Pooled ROC-AUC: {ensemble_boot['roc_auc']:.4f} (95% CI: [{ensemble_boot['roc_ci'][0]:.4f}, {ensemble_boot['roc_ci'][1]:.4f}]) | PR-AUC: {ensemble_boot['pr_auc']:.4f}")

    # 2. Evaluate all three trivial baselines
    print("\n==================== EVALUATING TRIVIAL BASELINES ====================")
    baseline_results = evaluate_all_baselines(X_dev_raw, y_multi_dev, meta_dev, dev_folds=DEV_FOLDS)

    # 3. Predict full dev cohort out-of-fold across all windows (including transitions and FoG)
    # This enables continuous trial timeline reconstruction and time-to-onset binning
    print("\n==================== GENERATING FULL DEV OOF PREDICTIONS ====================")
    full_dev_probs_per_seed = []
    for seed in seeds:
        seed_full_probs = np.zeros(len(meta_dev), dtype=np.float32)
        for f_idx in range(6):
            model_path = f"models/v2_dev/g_signal_check_seed{seed}_fold{f_idx}.h5"
            oof_path = f"reports/v2/oof/g_signal_check_seed{seed}_fold{f_idx}.npz"
            data = np.load(oof_path)
            mean = data['scaler_mean']
            std = data['scaler_std']

            outer_subjs = set(DEV_FOLDS[f_idx])
            outer_all_idx = [i for i, m in enumerate(meta_dev) if m['subjectID'] in outer_subjs]

            model = tf.keras.models.load_model(model_path)
            X_outer_norm = (X_dev_raw[outer_all_idx] - mean) / std
            probs_outer = model.predict(X_outer_norm, verbose=0).ravel()
            seed_full_probs[outer_all_idx] = probs_outer
        full_dev_probs_per_seed.append(seed_full_probs)

    full_dev_oof_probs = np.mean(full_dev_probs_per_seed, axis=0)

    # 4. Extract continuous trial records
    raw_csv = 'data/raw/sensor_data.csv'
    channels = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z', 'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']
    dev_trials = extract_trial_records(raw_csv, meta_dev, full_dev_oof_probs, channels)

    # 5. Time-to-Onset Probability Binning
    print("\n==================== TIME-TO-ONSET PROBABILITY DISTRIBUTION ====================")
    bins = [
        (-3.0, -2.5), (-2.5, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5),
        (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)
    ]
    bin_values = {b: [] for b in bins}

    for tr in dev_trials:
        if tr['subjectID'] in TEST_SUBJECTS:
            continue
        for ep in tr['episodes']:
            t_on = ep['onset']
            for t_end, p in zip(tr['end_times'], tr['probs']):
                dt = t_end - t_on
                for b in bins:
                    if b[0] <= dt < b[1]:
                        bin_values[b].append(p)

    bin_summary = []
    print(f"{'Time-to-Onset Bin (s)':<25} | {'Count':<8} | {'Mean Prob':<12} | {'Std Dev':<10}")
    print("-" * 60)
    for b in bins:
        vals = bin_values[b]
        m_prob = float(np.mean(vals)) if len(vals) > 0 else 0.0
        s_prob = float(np.std(vals)) if len(vals) > 0 else 0.0
        bin_label = f"[{b[0]:+.1f}, {b[1]:+.1f}) s"
        print(f"{bin_label:<25} | {len(vals):<8} | {m_prob:<12.4f} | {s_prob:<10.4f}")
        bin_summary.append({
            'bin_start_s': b[0],
            'bin_end_s': b[1],
            'bin_label': bin_label,
            'count': len(vals),
            'mean_prob': m_prob,
            'std_prob': s_prob
        })

    # 6. Per-Episode Early-Signal Table (Max EMA in [onset-5.0s, onset-0.5s])
    print("\n==================== PER-EPISODE EARLY-SIGNAL ANALYSIS ====================")
    episode_early_signals = []
    for tr in dev_trials:
        if tr['subjectID'] in TEST_SUBJECTS:
            continue
        # Compute EMA trajectory across all steps in trial
        ema = 0.0
        step_emas = []
        for t_end, p in zip(tr['end_times'], tr['probs']):
            ema = 0.8 * ema + 0.2 * p
            step_emas.append((t_end, ema, p))

        for ep_idx, ep in enumerate(tr['episodes']):
            t_on = ep['onset']
            t_off = ep['offset']
            dur = t_off - t_on

            # Windows in [t_on - 5.0, t_on - 0.5]
            early_emas = [e for t, e, _ in step_emas if (t_on - 5.0) <= t <= (t_on - 0.5)]
            early_raws = [p for t, _, p in step_emas if (t_on - 5.0) <= t <= (t_on - 0.5)]

            max_ema = float(np.max(early_emas)) if early_emas else 0.0
            max_raw = float(np.max(early_raws)) if early_raws else 0.0

            episode_early_signals.append({
                'subjectID': tr['subjectID'],
                'sessionID': tr['sessionID'],
                'taskID': tr['taskID'],
                'episode_idx': ep_idx,
                'onset_s': round(t_on, 2),
                'offset_s': round(t_off, 2),
                'duration_s': round(dur, 2),
                'max_ema_early': round(max_ema, 4),
                'max_raw_early': round(max_raw, 4),
                'crosses_threshold_0_49': bool(max_ema >= 0.49)
            })

    print(f"Total dev freeze episodes analyzed: {len(episode_early_signals)}")
    eps_crossing = [e for e in episode_early_signals if e['crosses_threshold_0_49']]
    print(f"Episodes with early EMA >= 0.49 in [onset-5.0s, onset-0.5s]: {len(eps_crossing)} / {len(episode_early_signals)} ({len(eps_crossing)/len(episode_early_signals)*100:.1f}%)")

    # 7. Pre-Registered Decision Rule Check
    print("\n==================== PRE-REGISTERED DECISION EVALUATION ====================")
    best_rescnn_auc = ensemble_boot['roc_auc']
    rescnn_ci_low = ensemble_boot['roc_ci'][0]
    rescnn_ci_high = ensemble_boot['roc_ci'][1]

    # Best baseline
    best_b_name = None
    best_b_auc = -1.0
    for b_name, b_res in baseline_results.items():
        b_auc = b_res['pooled_metrics']['roc_auc']
        if b_auc > best_b_auc:
            best_b_auc = b_auc
            best_b_name = b_name

    margin = best_rescnn_auc - best_b_auc
    ci_excludes_05 = rescnn_ci_low > 0.50
    auc_ge_065 = best_rescnn_auc >= 0.65
    margin_ge_005 = margin >= 0.05

    print(f"1. ResCNN Pooled ROC-AUC >= 0.65?        {best_rescnn_auc:.4f} >= 0.65 -> {auc_ge_065}")
    print(f"2. 95% Cluster-Bootstrap CI Excludes 0.5? [{rescnn_ci_low:.4f}, {rescnn_ci_high:.4f}] -> {ci_excludes_05}")
    print(f"3. ResCNN Margin over Best Baseline >= 0.05?")
    print(f"   Best Baseline: '{best_b_name}' (AUC = {best_b_auc:.4f})")
    print(f"   Margin: {margin:+.4f} >= 0.05 -> {margin_ge_005}")

    all_passed = auc_ge_065 and ci_excludes_05 and margin_ge_005
    if all_passed:
        decision = "PREDICTION_CLAIM_SUPPORTED"
        decision_text = "PASS: Signal check passed. Proceed with predictive early warning formulation."
    else:
        decision = "FORMAL_RESCOPE_TO_FAST_DETECTION"
        decision_text = (
            "FAIL / RE-SCOPE: The empirical evidence does not support anticipatory prediction 0.5-3.0s ahead. "
            "Per pre-registered protocol, primary project objective is formally RE-SCOPED to FAST DETECTION "
            "(rapid reaction upon freeze inception, minimizing detection delay)."
        )

    print("\n" + "#" * 80)
    print(f"FORMAL PROTOCOL DECISION: {decision}")
    print(decision_text)
    print("#" * 80)

    # Save comprehensive results JSON
    report_dict = {
        'decision': decision,
        'decision_text': decision_text,
        'criteria': {
            'rescnn_pooled_roc_auc': best_rescnn_auc,
            'rescnn_ci': [rescnn_ci_low, rescnn_ci_high],
            'best_baseline_name': best_b_name,
            'best_baseline_auc': best_b_auc,
            'margin': margin,
            'auc_ge_065': auc_ge_065,
            'ci_excludes_05': ci_excludes_05,
            'margin_ge_005': margin_ge_005
        },
        'rescnn_ensemble': ensemble_boot,
        'rescnn_seeds': [
            {
                'seed': r['seed'],
                'pooled_metrics': r['pooled_metrics'],
                'fold_metrics': r['fold_metrics'],
                'train_time_s': r['train_time_s']
            } for r in rescnn_seed_runs
        ],
        'baselines': {
            b_name: {
                'pooled_metrics': b_data['pooled_metrics'],
                'fold_metrics': b_data['fold_metrics']
            } for b_name, b_data in baseline_results.items()
        },
        'time_to_onset_distribution': bin_summary,
        'episode_early_signals': episode_early_signals
    }

    os.makedirs('reports/v2/diagnostics', exist_ok=True)
    out_file = 'reports/v2/diagnostics/experiment_g_full_report.json'
    with open(out_file, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nSaved full report to {out_file}")

    return report_dict


if __name__ == '__main__':
    run_experiment_g_full()

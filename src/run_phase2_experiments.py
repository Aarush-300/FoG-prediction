"""
Phase 2 Experiments Pipeline: Baseline Stability, (b) Rotation, (e) Loss Weighting, (f) Labels.

Executes all deployable candidate models under 6-fold subject-grouped CV with
inner-validation early stopping, full continuous trial evaluation, FSM grid search,
optimism gap estimation via Leave-One-Fold-Out (LOFO), and formal Winner Rule evaluation.
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cv_splits import DEV_FOLDS, INNER_VAL_FOLDS, TEST_SUBJECTS, verify_test_isolation
from train_cv import load_dev_data, run_experiment_fold
from eval_pipeline import extract_trial_records, FoGFSM


def train_and_get_full_oof(exp_id, seed, max_angle_deg, prefog_weight, epochs=40, batch_size=64):
    """Trains 6 folds for a given experiment configuration and computes full dev cohort OOF predictions."""
    X_dev_raw, y_bin_dev, y_multi_dev, meta_dev = load_dev_data()
    verify_test_isolation(meta_dev)

    # Train all 6 folds (or resume if checkpoints exist)
    t_start = time.time()
    for f_idx in range(6):
        run_experiment_fold(
            exp_id=exp_id,
            seed=seed,
            fold_idx=f_idx,
            X_dev_raw=X_dev_raw,
            y_bin_dev=y_bin_dev,
            y_multi_dev=y_multi_dev,
            meta_dev=meta_dev,
            epochs=epochs,
            batch_size=batch_size,
            max_angle_deg=max_angle_deg,
            prefog_weight=prefog_weight
        )
    total_train_time = time.time() - t_start

    # Generate full dev cohort out-of-fold predictions across all 8,814 windows
    full_dev_probs = np.zeros(len(meta_dev), dtype=np.float32)
    for f_idx in range(6):
        model_path = f"models/v2_dev/{exp_id}_seed{seed}_fold{f_idx}.h5"
        oof_path = f"reports/v2/oof/{exp_id}_seed{seed}_fold{f_idx}.npz"
        data = np.load(oof_path)
        mean = data['scaler_mean']
        std = data['scaler_std']

        outer_subjs = set(DEV_FOLDS[f_idx])
        outer_all_idx = [i for i, m in enumerate(meta_dev) if m['subjectID'] in outer_subjs]

        model = tf.keras.models.load_model(model_path)
        X_outer_norm = (X_dev_raw[outer_all_idx] - mean) / std
        probs_outer = model.predict(X_outer_norm, verbose=0).ravel()
        full_dev_probs[outer_all_idx] = probs_outer

    # Compute window-level metrics
    if exp_id == 'f_drop_transition':
        eval_mask = np.array([m['multi_label'] != -1 for m in meta_dev])
        y_eval = y_bin_dev[eval_mask]
        p_eval = full_dev_probs[eval_mask]
    elif exp_id == 'f_clean_positives':
        eval_mask = np.array([m['multi_label'] in [0, 1, 2] for m in meta_dev])
        y_eval = np.array([1 if m['multi_label'] in [1, 2] else 0 for i, m in enumerate(meta_dev) if eval_mask[i]])
        p_eval = full_dev_probs[eval_mask]
    else:
        y_eval = y_bin_dev
        p_eval = full_dev_probs

    window_roc = float(roc_auc_score(y_eval, p_eval))
    window_pr = float(average_precision_score(y_eval, p_eval))

    return full_dev_probs, window_roc, window_pr, total_train_time


def build_dev_trials(full_dev_probs, meta_dev):
    raw_csv = 'data/raw/sensor_data.csv'
    channels = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z', 'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']
    trials = extract_trial_records(raw_csv, meta_dev, full_dev_probs, channels)

    # Filter out test subjects and annotate fold index
    dev_trials = []
    subj_to_fold = {}
    for f_idx, subjs in DEV_FOLDS.items():
        for s in subjs:
            subj_to_fold[s] = f_idx

    for tr in trials:
        if tr['subjectID'] in TEST_SUBJECTS:
            continue
        tr['fold_idx'] = subj_to_fold[tr['subjectID']]
        dev_trials.append(tr)

    return dev_trials


def eval_trials_fsm(trials, config):
    """Evaluates trial records through the FSM with a given configuration."""
    fsm = FoGFSM(
        high=config['high'],
        low=config['low'],
        confirm=config['confirm'],
        cooldown_ms=config['cooldown_ms'],
        alpha=config['alpha']
    )

    total_false_alerts = 0
    total_normal_s = 0.0
    episodes_eval = []

    for tr in trials:
        fsm.reset()
        alerts = []
        for p, t_end in zip(tr['probs'], tr['end_times']):
            now_ms = int(round(t_end * 1000))
            _, fired = fsm.update(p, now_ms)
            if fired:
                alerts.append(t_end)

        # Count false alerts
        for a in alerts:
            is_true = any(ep['onset'] - 3.0 <= a <= ep['offset'] for ep in tr['episodes'])
            if not is_true:
                total_false_alerts += 1
        total_normal_s += tr['normal_duration_s']

        # Episode-level performance
        for ep in tr['episodes']:
            ep_alerts = [a for a in alerts if ep['onset'] - 3.0 <= a <= ep['offset']]
            strict_alerts = [a for a in alerts if ep['onset'] - 3.0 <= a <= ep['onset'] - 0.5]
            relaxed_alerts = [a for a in alerts if ep['onset'] - 3.0 <= a <= ep['onset']]

            lenient_caught = len(ep_alerts) > 0
            strict_caught = len(strict_alerts) > 0
            relaxed_caught = len(relaxed_alerts) > 0

            delay = float(ep_alerts[0] - ep['onset']) if ep_alerts else None
            alerted_within_1s = any(a <= ep['onset'] + 1.0 for a in ep_alerts)

            episodes_eval.append({
                'lenient_caught': lenient_caught,
                'strict_caught': strict_caught,
                'relaxed_caught': relaxed_caught,
                'delay': delay,
                'alerted_within_1s': alerted_within_1s
            })

    norm_hr = total_normal_s / 3600.0
    fa_hr = float(total_false_alerts / norm_hr) if norm_hr > 0 else 0.0

    n_eps = len(episodes_eval)
    lenient_sens = float(sum(e['lenient_caught'] for e in episodes_eval) / n_eps) if n_eps > 0 else 0.0
    strict_sens = float(sum(e['strict_caught'] for e in episodes_eval) / n_eps) if n_eps > 0 else 0.0
    relaxed_sens = float(sum(e['relaxed_caught'] for e in episodes_eval) / n_eps) if n_eps > 0 else 0.0

    delays = [e['delay'] for e in episodes_eval if e['delay'] is not None]
    median_delay = float(np.median(delays)) if delays else 0.0
    pct_within_1s = float(sum(e['alerted_within_1s'] for e in episodes_eval) / n_eps * 100.0) if n_eps > 0 else 0.0

    return {
        'fa_hr': fa_hr,
        'lenient_sensitivity': lenient_sens,
        'strict_sensitivity': strict_sens,
        'relaxed_sensitivity': relaxed_sens,
        'median_delay_s': median_delay,
        'pct_alerted_within_1s': pct_within_1s,
        'total_episodes': n_eps,
        'caught_episodes': sum(e['lenient_caught'] for e in episodes_eval),
        'total_false_alerts': total_false_alerts,
        'total_normal_hours': norm_hr
    }


def grid_search_fsm(trials, max_fa_hr=30.0):
    """Searches over FSM grid to maximize lenient sensitivity subject to FA/hr <= max_fa_hr."""
    grid = []
    for confirm in [1, 2]:
        for alpha in [0.60, 0.70, 0.80, 0.90, 1.00]:
            for high in [0.35, 0.40, 0.45, 0.49, 0.55, 0.60, 0.65]:
                for low in [0.15, 0.20, 0.25, 0.29, 0.35]:
                    if low >= high:
                        continue
                    for cd in [2000, 3000, 4000, 5000]:
                        grid.append({
                            'confirm': confirm,
                            'alpha': alpha,
                            'high': high,
                            'low': low,
                            'cooldown_ms': cd
                        })

    best_config = None
    best_metrics = None
    best_lenient = -1.0
    best_fa = 999.0

    for cfg in grid:
        res = eval_trials_fsm(trials, cfg)
        if res['fa_hr'] <= max_fa_hr:
            if (res['lenient_sensitivity'] > best_lenient) or (
                res['lenient_sensitivity'] == best_lenient and res['fa_hr'] < best_fa
            ):
                best_lenient = res['lenient_sensitivity']
                best_fa = res['fa_hr']
                best_config = cfg
                best_metrics = res

    # Fall-through if no config meets FA budget
    if best_config is None:
        min_fa = 9999.0
        for cfg in grid:
            res = eval_trials_fsm(trials, cfg)
            if res['lenient_sensitivity'] >= 0.70 and res['fa_hr'] < min_fa:
                min_fa = res['fa_hr']
                best_config = cfg
                best_metrics = res
        if best_config is None:
            best_config = grid[0]
            best_metrics = eval_trials_fsm(trials, best_config)

    return best_config, best_metrics


def eval_in_sample_and_lofo(dev_trials, max_fa_hr=30.0):
    """Computes both in-sample pooled OOF performance and out-of-sample Leave-One-Fold-Out (LOFO) performance."""
    # 1. In-sample optimization on all pooled dev trials
    in_sample_cfg, in_sample_metrics = grid_search_fsm(dev_trials, max_fa_hr=max_fa_hr)

    # 2. Leave-One-Fold-Out (LOFO) evaluation
    lofo_fold_results = {}
    lofo_configs = {}
    test_eps_all = []
    total_lofo_fa = 0
    total_lofo_normal_s = 0.0

    for test_fold in range(6):
        train_trials = [tr for tr in dev_trials if tr['fold_idx'] != test_fold]
        test_trials = [tr for tr in dev_trials if tr['fold_idx'] == test_fold]

        # Choose best FSM on 5 training folds
        fsm_opt, _ = grid_search_fsm(train_trials, max_fa_hr=max_fa_hr)
        lofo_configs[test_fold] = fsm_opt

        # Evaluate on held-out 6th fold
        test_eval = eval_trials_fsm(test_trials, fsm_opt)
        lofo_fold_results[test_fold] = test_eval

        total_lofo_fa += test_eval['total_false_alerts']
        total_lofo_normal_s += test_eval['total_normal_hours'] * 3600.0

    # Aggregate LOFO across all 6 test folds
    total_eps = sum(lofo_fold_results[f]['total_episodes'] for f in range(6))
    total_caught = sum(lofo_fold_results[f]['caught_episodes'] for f in range(6))
    pooled_lofo_normal_hr = total_lofo_normal_s / 3600.0

    pooled_lofo_metrics = {
        'lenient_sensitivity': float(total_caught / total_eps) if total_eps > 0 else 0.0,
        'strict_sensitivity': float(np.mean([lofo_fold_results[f]['strict_sensitivity'] for f in range(6)])),
        'relaxed_sensitivity': float(np.mean([lofo_fold_results[f]['relaxed_sensitivity'] for f in range(6)])),
        'fa_hr': float(total_lofo_fa / pooled_lofo_normal_hr) if pooled_lofo_normal_hr > 0 else 0.0,
        'median_delay_s': float(np.median([lofo_fold_results[f]['median_delay_s'] for f in range(6)])),
        'pct_alerted_within_1s': float(np.mean([lofo_fold_results[f]['pct_alerted_within_1s'] for f in range(6)]))
    }

    # Fold-level lenient sensitivity per fold (for paired winner test)
    fold_lenient_sens = [lofo_fold_results[f]['lenient_sensitivity'] for f in range(6)]

    optimism_gap = {
        'gap_lenient_sens': in_sample_metrics['lenient_sensitivity'] - pooled_lofo_metrics['lenient_sensitivity'],
        'gap_fa_hr': pooled_lofo_metrics['fa_hr'] - in_sample_metrics['fa_hr']
    }

    return {
        'in_sample_config': in_sample_cfg,
        'in_sample_metrics': in_sample_metrics,
        'lofo_metrics': pooled_lofo_metrics,
        'lofo_fold_metrics': lofo_fold_results,
        'fold_lenient_sens': fold_lenient_sens,
        'optimism_gap': optimism_gap
    }


def run_all_phase2_experiments():
    print("=" * 80)
    print("STARTING FULL PHASE 2 EXPERIMENTS: BASELINE SD, (b), (e), (f)")
    print("=" * 80)

    _, _, _, meta_dev = load_dev_data()
    all_results = {}

    # 1. Baseline Recipe across 3 Seeds: [42, 43, 44] to measure seed-to-seed SD
    baseline_seeds = [42, 43, 44]
    baseline_seed_runs = {}

    for sd in baseline_seeds:
        print(f"\n--- Running Baseline Recipe (b_rotation_35) Seed {sd} ---")
        probs, w_roc, w_pr, t_train = train_and_get_full_oof(
            exp_id='b_rotation_35',
            seed=sd,
            max_angle_deg=35.0,
            prefog_weight=1.0,
            epochs=35
        )
        trials = build_dev_trials(probs, meta_dev)
        eval_res = eval_in_sample_and_lofo(trials, max_fa_hr=30.0)
        baseline_seed_runs[sd] = {
            'window_roc': w_roc,
            'window_pr': w_pr,
            'train_time_s': t_train,
            'eval_res': eval_res
        }
        print(f"  [Baseline Seed {sd}] Window ROC: {w_roc:.4f} | PR: {w_pr:.4f} | LOFO Lenient: {eval_res['lofo_metrics']['lenient_sensitivity']*100:.1f}% | FA/hr: {eval_res['lofo_metrics']['fa_hr']:.1f}")

    # Compute seed-to-seed SD of lenient sensitivity across the 3 baseline seeds
    base_lenient_sens_seeds = [baseline_seed_runs[sd]['eval_res']['lofo_metrics']['lenient_sensitivity'] for sd in baseline_seeds]
    sigma_seed = float(np.std(base_lenient_sens_seeds, ddof=1))
    base_s42_folds = baseline_seed_runs[42]['eval_res']['fold_lenient_sens']

    print("\n" + "=" * 80)
    print(f"BASELINE STABILITY: Seed-to-seed SD (sigma_seed) = {sigma_seed:.4f} ({sigma_seed*100:.2f} percentage points)")
    print(f"Baseline Seeds LOFO Lenient: {[round(x, 4) for x in base_lenient_sens_seeds]}")
    print("=" * 80)

    all_results['baseline_stability'] = {
        'seeds': baseline_seeds,
        'lenient_sens_per_seed': base_lenient_sens_seeds,
        'sigma_seed': sigma_seed,
        'seed_runs': baseline_seed_runs
    }

    # Define Candidate Variants to evaluate (all seed 42)
    candidate_variants = [
        ('b_rotation_50', 50.0, 1.0, 'v1'),
        ('b_rotation_70', 70.0, 1.0, 'v1'),
        ('e_prefog_2x', 35.0, 2.0, 'v1'),
        ('e_prefog_4x', 35.0, 4.0, 'v1'),
        ('f_drop_transition', 35.0, 1.0, 'drop_trans'),
        ('f_clean_positives', 35.0, 1.0, 'clean_pos')
    ]

    variant_results = {}
    for exp_id, angle, weight, label_mode in candidate_variants:
        print(f"\n--- Running Candidate Variant: {exp_id} ---")
        probs, w_roc, w_pr, t_train = train_and_get_full_oof(
            exp_id=exp_id,
            seed=42,
            max_angle_deg=angle,
            prefog_weight=weight,
            epochs=35
        )
        trials = build_dev_trials(probs, meta_dev)
        eval_res = eval_in_sample_and_lofo(trials, max_fa_hr=30.0)

        # Fold-level paired difference vs baseline seed 42
        cand_folds = eval_res['fold_lenient_sens']
        paired_diffs = [cand_folds[k] - base_s42_folds[k] for k in range(6)]
        delta_paired = float(np.mean(paired_diffs))

        # Winner rule check: delta_paired > sigma_seed
        beats_threshold = delta_paired > sigma_seed

        variant_results[exp_id] = {
            'exp_id': exp_id,
            'max_angle_deg': angle,
            'prefog_weight': weight,
            'label_mode': label_mode,
            'window_roc': w_roc,
            'window_pr': w_pr,
            'train_time_s': t_train,
            'eval_res': eval_res,
            'fold_lenient_sens': cand_folds,
            'paired_diffs_vs_base42': paired_diffs,
            'delta_paired': delta_paired,
            'sigma_seed_threshold': sigma_seed,
            'beats_threshold': beats_threshold
        }

        print(f"  [{exp_id}] LOFO Lenient: {eval_res['lofo_metrics']['lenient_sensitivity']*100:.1f}% | FA/hr: {eval_res['lofo_metrics']['fa_hr']:.1f} | Paired Delta vs Base: {delta_paired*100:+.2f}% | Beats SD ({sigma_seed*100:.2f}%): {beats_threshold}")

    all_results['variants'] = variant_results

    # Summary and Winner Declaration
    print("\n" + "#" * 80)
    print("PHASE 2 SUMMARY & WINNER RULE EVALUATION")
    print("#" * 80)
    print(f"{'Variant':<20} | {'Window ROC':<10} | {'LOFO Lenient':<12} | {'LOFO Strict':<11} | {'LOFO FA/hr':<10} | {'Paired Delta':<12} | {'Exceeds SD?':<12}")
    print("-" * 95)

    base_s42_lofo = baseline_seed_runs[42]['eval_res']['lofo_metrics']
    print(f"{'Baseline (Seed 42)':<20} | {baseline_seed_runs[42]['window_roc']:<10.4f} | {base_s42_lofo['lenient_sensitivity']*100:<11.1f}% | {base_s42_lofo['strict_sensitivity']*100:<10.1f}% | {base_s42_lofo['fa_hr']:<10.1f} | {'[REF]':<12} | {'N/A':<12}")

    winners = []
    for exp_id, v in variant_results.items():
        m = v['eval_res']['lofo_metrics']
        exceeds_str = "YES (WINNER)" if v['beats_threshold'] else "NO"
        if v['beats_threshold']:
            winners.append(exp_id)
        print(f"{exp_id:<20} | {v['window_roc']:<10.4f} | {m['lenient_sensitivity']*100:<11.1f}% | {m['strict_sensitivity']*100:<10.1f}% | {m['fa_hr']:<10.1f} | {v['delta_paired']*100:+11.2f}% | {exceeds_str:<12}")

    if len(winners) == 0:
        overall_decision = "RETAIN_BASELINE_V1"
        decision_text = "No candidate variant produced a fold-paired improvement exceeding baseline seed-to-seed standard deviation. Retaining v1 baseline recipe."
    else:
        best_winner = max(winners, key=lambda w: variant_results[w]['delta_paired'])
        overall_decision = f"ADOPT_{best_winner.upper()}"
        decision_text = f"Candidate variant '{best_winner}' achieved a fold-paired gain of {variant_results[best_winner]['delta_paired']*100:+.2f}%, exceeding the seed SD ({sigma_seed*100:.2f}%). Selected as new production recipe."

    all_results['overall_decision'] = overall_decision
    all_results['decision_text'] = decision_text
    print("\n" + "=" * 80)
    print(f"DECISION: {overall_decision}")
    print(decision_text)
    print("=" * 80)

    # Save to JSON
    os.makedirs('reports/v2/diagnostics', exist_ok=True)
    out_file = 'reports/v2/diagnostics/phase2_experiments_full_report.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full report to {out_file}")

    return all_results


if __name__ == '__main__':
    run_all_phase2_experiments()

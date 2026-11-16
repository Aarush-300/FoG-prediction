"""
Experiment (g) Controls and Supplemental Diagnostics.

Computes:
1. Early-signal control: max EMA (alpha 0.8, 4.5s) on matched normal stretches (>8s from onset)
   sampled to equal count per subject, reporting fraction crossing 0.49 for episodes vs normal,
   per subject and pooled, with a subject-level bootstrap CI (1,000 resamples).
2. Extended time-to-onset table with a 'far from onset' reference row (>8s away).
3. Per-subject ROC-AUC for subjects with >= 3 episodes and count with AUC > 0.6.
4. Headline single-seed performance metrics (ESP32 runs a single model).
5. PR-AUC lift over empirical base rate (5.43%) for each seed.

Saves full results to reports/v2/diagnostics/experiment_g_controls.json.
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_splits import TEST_SUBJECTS, DEV_FOLDS, verify_test_isolation
from train_cv import load_dev_data
from eval_pipeline import extract_trial_records


def compute_controls():
    print("=" * 80)
    print("COMPUTING EXPERIMENT (g) CONTROLS & DIAGNOSTICS")
    print("=" * 80)

    X_dev_raw, y_bin_dev, y_multi_dev, meta_dev = load_dev_data()
    verify_test_isolation(meta_dev)

    # 1. Load OOF predictions for pure pre-FoG (label 1) vs normal (label 0)
    valid_idx = [i for i, m in enumerate(meta_dev) if m['multi_label'] in [0, 1]]
    valid_subjs = np.array([meta_dev[i]['subjectID'] for i in valid_idx])
    valid_y = np.array([1 if meta_dev[i]['multi_label'] == 1 else 0 for i in valid_idx])

    probs_by_seed = {}
    for seed in [42, 43, 44]:
        seed_probs = np.zeros(len(valid_idx), dtype=np.float32)
        idx_map = {orig_i: local_i for local_i, orig_i in enumerate(valid_idx)}
        for f in range(6):
            data = np.load(f'reports/v2/oof/g_signal_check_seed{seed}_fold{f}.npz')
            for vi, p in zip(data['val_indices'], data['val_probs']):
                if vi in idx_map:
                    seed_probs[idx_map[vi]] = p
        probs_by_seed[seed] = seed_probs

    # Empirical base rate
    n_pos = int(valid_y.sum())
    n_total = len(valid_y)
    base_rate = float(n_pos / n_total)

    # 2(d) & 2(e): Single seed headline and lift over base rate
    headline_seeds = {}
    for seed in [42, 43, 44]:
        p = probs_by_seed[seed]
        roc = float(roc_auc_score(valid_y, p))
        pr = float(average_precision_score(valid_y, p))
        lift = float(pr / base_rate)
        headline_seeds[f"seed_{seed}"] = {
            'roc_auc': roc,
            'pr_auc': pr,
            'pr_lift_over_base': lift
        }
        print(f"Seed {seed}: ROC-AUC = {roc:.4f} | PR-AUC = {pr:.4f} | Lift = {lift:.2f}x (Base rate = {base_rate*100:.2f}%)")

    # 2(c): Per-subject ROC-AUC for subjects with at least 3 episodes
    # Subjects with >= 3 episodes: 1, 2, 3, 9, 11, 12, 13, 15, 19
    subjs_ge3 = [1, 2, 3, 9, 11, 12, 13, 15, 19]
    per_subj_roc = {}
    cnt_above_06 = 0
    print("\n--- Per-Subject ROC-AUC (Subjects with >= 3 episodes) ---")
    for s in subjs_ge3:
        mask = (valid_subjs == s)
        y_s = valid_y[mask]
        s_rocs = [float(roc_auc_score(y_s, probs_by_seed[seed][mask])) for seed in [42, 43, 44]]
        mean_roc = float(np.mean(s_rocs))
        if mean_roc > 0.60:
            cnt_above_06 += 1
        per_subj_roc[str(s)] = {
            'subjectID': s,
            'n_pos_windows': int(y_s.sum()),
            'n_neg_windows': int((y_s == 0).sum()),
            'seed_rocs': s_rocs,
            'mean_roc_auc': mean_roc,
            'auc_above_0_6': bool(mean_roc > 0.60)
        }
        print(f"Subj {s:2d}: Mean ROC = {mean_roc:.4f} (Seed 42={s_rocs[0]:.4f}, 43={s_rocs[1]:.4f}, 44={s_rocs[2]:.4f}) | > 0.6: {mean_roc > 0.60}")
    print(f"Count of subjects with >=3 episodes having AUC > 0.6: {cnt_above_06} / {len(subjs_ge3)}")

    # Load continuous trial records for 2(a) and 2(b)
    # Average probabilities over 3 seeds
    full_dev_probs_per_seed = []
    for seed in [42, 43, 44]:
        seed_probs = np.zeros(len(meta_dev), dtype=np.float32)
        for f in range(6):
            data = np.load(f'reports/v2/oof/g_signal_check_seed{seed}_fold{f}.npz')
            for vi, p in zip(data['val_indices'], data['val_probs']):
                seed_probs[vi] = p
        full_dev_probs_per_seed.append(seed_probs)
    avg_probs = np.mean(full_dev_probs_per_seed, axis=0)

    raw_csv = 'data/raw/sensor_data.csv'
    channels = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z', 'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']
    dev_trials = extract_trial_records(raw_csv, meta_dev, avg_probs, channels)

    # 2(b): Far from onset reference row (> 8.0s away from any onset)
    far_from_onset_probs = []
    for tr in dev_trials:
        if tr['subjectID'] in TEST_SUBJECTS:
            continue
        onsets = [ep['onset'] for ep in tr['episodes']]
        offsets = [ep['offset'] for ep in tr['episodes']]
        for t_end, p, lbl in zip(tr['end_times'], tr['probs'], tr['binary_labels']):
            if lbl == 0:
                future_onsets = [on for on in onsets if on > t_end]
                # Distance to next onset > 8.0s (or no future onset in trial)
                if len(future_onsets) == 0 or min(future_onsets) - t_end > 8.0:
                    # Also outside post-freeze recovery
                    if not any(off <= t_end <= off + 3.0 for off in offsets):
                        far_from_onset_probs.append(float(p))

    far_ref = {
        'row_label': 'Far from onset (> 8.0 s)',
        'count': len(far_from_onset_probs),
        'mean_prob': float(np.mean(far_from_onset_probs)),
        'std_prob': float(np.std(far_from_onset_probs))
    }
    print(f"\n--- Far from Onset Reference Row ---")
    print(f"Windows count: {far_ref['count']} | Mean Prob: {far_ref['mean_prob']:.4f} +/- {far_ref['std_prob']:.4f}")

    # 2(a): Matched normal stretches control (4.5s length, max EMA >= 0.49)
    subject_episodes = {}
    subject_normal_stretches = {}

    for tr in dev_trials:
        subj = tr['subjectID']
        if subj in TEST_SUBJECTS:
            continue
        if subj not in subject_episodes:
            subject_episodes[subj] = []
            subject_normal_stretches[subj] = []

        onsets = [ep['onset'] for ep in tr['episodes']]
        offsets = [ep['offset'] for ep in tr['episodes']]

        ema = 0.0
        step_emas = []
        for t_end, p in zip(tr['end_times'], tr['probs']):
            ema = 0.8 * ema + 0.2 * p
            step_emas.append((t_end, ema, p))

        for ep in tr['episodes']:
            t_on = ep['onset']
            early_emas = [e for t, e, _ in step_emas if (t_on - 5.0) <= t <= (t_on - 0.5)]
            max_ema = float(np.max(early_emas)) if early_emas else 0.0
            subject_episodes[subj].append({
                'max_ema': max_ema,
                'crosses_0_49': bool(max_ema >= 0.49)
            })

        # Find eligible 4.5s normal stretches (10 windows with 0.5s stride = 4.5s)
        # where all windows have distance >= 8.0s to any freeze onset
        for i in range(len(step_emas) - 9):
            stretch = step_emas[i : i + 10]
            t_start = stretch[0][0] - 2.0
            t_end = stretch[-1][0]

            far_from_onsets = True
            for on in onsets:
                if not (t_end <= on - 8.0 or t_start >= on + 8.0):
                    far_from_onsets = False
                    break
            if far_from_onsets:
                not_in_offset_tail = True
                for off in offsets:
                    if not (t_end <= off or t_start >= off + 3.0):
                        not_in_offset_tail = False
                        break
                if not_in_offset_tail:
                    max_ema_stretch = float(np.max([e for _, e, _ in stretch]))
                    subject_normal_stretches[subj].append({
                        't_start': t_start,
                        't_end': t_end,
                        'max_ema': max_ema_stretch,
                        'crosses_0_49': bool(max_ema_stretch >= 0.49)
                    })

    rng = np.random.RandomState(42)
    matched_normals = {}
    control_comparison = {}
    ep_cross_list = []
    norm_cross_list = []
    subj_list = []

    print("\n--- Early-Signal Control: Episodes vs Matched Normal Stretches ---")
    print(f"{'Subject':<8} | {'N_eps':<6} | {'Episode Cross':<15} | {'Normal Cross':<15} | {'Difference':<10}")
    print("-" * 65)

    for s in sorted(subject_episodes.keys()):
        n_eps = len(subject_episodes[s])
        if n_eps == 0:
            continue
        avail = subject_normal_stretches[s]
        assert len(avail) >= n_eps, f"Subject {s} has only {len(avail)} normal stretches for {n_eps} episodes!"

        chosen_idx = rng.choice(len(avail), size=n_eps, replace=False)
        sampled = [avail[i] for i in chosen_idx]
        matched_normals[s] = sampled

        ep_cross = sum(e['crosses_0_49'] for e in subject_episodes[s])
        norm_cross = sum(n['crosses_0_49'] for n in sampled)
        frac_ep = ep_cross / n_eps
        frac_norm = norm_cross / n_eps
        diff = frac_ep - frac_norm

        control_comparison[str(s)] = {
            'subjectID': s,
            'n_episodes': n_eps,
            'episodes_crossing_0_49': ep_cross,
            'fraction_episodes_crossing': frac_ep,
            'normal_stretches_crossing_0_49': norm_cross,
            'fraction_normal_crossing': frac_norm,
            'difference': diff
        }

        for e in subject_episodes[s]:
            ep_cross_list.append(int(e['crosses_0_49']))
            subj_list.append(s)
        for n in sampled:
            norm_cross_list.append(int(n['crosses_0_49']))

        print(f"Subj {s:<3d} | {n_eps:<6d} | {ep_cross}/{n_eps} ({frac_ep*100:5.1f}%)   | {norm_cross}/{n_eps} ({frac_norm*100:5.1f}%)   | {diff:+6.3f}")

    total_eps = len(ep_cross_list)
    total_ep_cross = sum(ep_cross_list)
    total_norm_cross = sum(norm_cross_list)
    pooled_frac_ep = total_ep_cross / total_eps
    pooled_frac_norm = total_norm_cross / total_eps
    pooled_diff = pooled_frac_ep - pooled_frac_norm

    print("-" * 65)
    print(f"POOLED   | {total_eps:<6d} | {total_ep_cross}/{total_eps} ({pooled_frac_ep*100:5.1f}%) | {total_norm_cross}/{total_eps} ({pooled_frac_norm*100:5.1f}%) | {pooled_diff:+6.3f}")

    # Subject-level cluster bootstrap CI of the difference (1,000 resamples)
    boot_rng = np.random.RandomState(42)
    dev_subjs_with_eps = np.array(sorted(list(subject_episodes.keys())))
    dev_subjs_with_eps = [s for s in dev_subjs_with_eps if len(subject_episodes[s]) > 0]

    boot_diffs = []
    for _ in range(1000):
        sampled_subjs = boot_rng.choice(dev_subjs_with_eps, size=len(dev_subjs_with_eps), replace=True)
        boot_ep_cross = sum(control_comparison[str(s)]['episodes_crossing_0_49'] for s in sampled_subjs)
        boot_norm_cross = sum(control_comparison[str(s)]['normal_stretches_crossing_0_49'] for s in sampled_subjs)
        boot_n = sum(control_comparison[str(s)]['n_episodes'] for s in sampled_subjs)
        boot_diffs.append((boot_ep_cross - boot_norm_cross) / boot_n)

    diff_ci = (float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5)))
    print(f"Subject-Level Bootstrap 95% CI of Difference: [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")

    # Assemble and write reports/v2/diagnostics/experiment_g_controls.json
    output_report = {
        'headline_single_seed_metrics': headline_seeds,
        'base_rate': base_rate,
        'per_subject_roc_ge3_episodes': {
            'count_above_0_6': cnt_above_06,
            'total_subjects_ge3_eps': len(subjs_ge3),
            'subjects': per_subj_roc
        },
        'far_from_onset_reference': far_ref,
        'matched_normal_control': {
            'pooled_episodes_crossing': f"{total_ep_cross}/{total_eps} ({pooled_frac_ep*100:.1f}%)",
            'pooled_normals_crossing': f"{total_norm_cross}/{total_eps} ({pooled_frac_norm*100:.1f}%)",
            'pooled_difference': pooled_diff,
            'bootstrap_95_ci_of_difference': diff_ci,
            'per_subject': control_comparison
        }
    }

    out_path = 'reports/v2/diagnostics/experiment_g_controls.json'
    with open(out_path, 'w') as f:
        json.dump(output_report, f, indent=2)
    print(f"\nSaved controls report to {out_path}")
    return output_report


if __name__ == '__main__':
    compute_controls()

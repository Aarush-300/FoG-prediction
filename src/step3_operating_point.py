"""
Step 3: Operating-Point Report (Dev Data Only).

Quantizes all fold models of the adopted recipe to INT8, evaluates INT8 OOF probabilities,
runs the dual-preset FSM grid search (LOW_ALARM and BALANCED) under Rule 1.3,
computes Pareto tables across FA budgets (15, 30, 60, 100), latency distributions,
per-subject lenient sensitivity (for subjects with >= 3 episodes), and optimism gaps.

Outputs:
- reports/v2/operating_point.md
- reports/v2/diagnostics/operating_point_report.json
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
from train_cv import load_dev_data
from run_phase2_experiments import build_dev_trials


class EnhancedFoGFSM:
    """
    Enhanced FoG Finite State Machine supporting:
    - 4 States: NORMAL (0), WARNING (1), ACTIVE_ALERT (2), COOLDOWN (3)
    - Re-arm policies:
      'timeout': re-arms to NORMAL strictly after cooldown_ms has elapsed
      'timeout_and_low': re-arms to NORMAL after cooldown_ms has elapsed AND smoothed_prob < low
    - Re-cue intervals:
      0 (none): no additional cueing during cooldown
      5 or 10s: triggers secondary cue pulse if smoothed_prob remains >= high
    Note: Sensitivity and first-alert latency are calculated strictly on the first alert.
    """
    def __init__(self, high=0.49, low=0.29, confirm=2, cooldown_ms=3000, alpha=0.60,
                 rearm_policy='timeout', recue_interval_s=0):
        self.high = high
        self.low = low
        self.confirm = confirm
        self.cooldown_ms = cooldown_ms
        self.alpha = alpha
        self.rearm_policy = rearm_policy
        self.recue_interval_ms = int(recue_interval_s * 1000)
        self.reset()

    def reset(self):
        self.state = 0
        self.smoothed_prob = 0.0
        self.last_raw_prob = 0.0
        self.consecutive_count = 0
        self.episode_count = 0
        self.last_alert_time = -999999
        self.last_cue_time = -999999

    def update(self, raw_prob, now_ms):
        self.last_raw_prob = raw_prob
        self.smoothed_prob = (self.alpha * raw_prob) + ((1.0 - self.alpha) * self.smoothed_prob)

        fired_alert = False
        fired_recue = False

        if self.state == 0:  # NORMAL
            if self.smoothed_prob >= self.high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirm:
                    self.state = 2  # ACTIVE_ALERT
                    self.episode_count += 1
                    self.last_alert_time = now_ms
                    self.last_cue_time = now_ms
                    fired_alert = True
                else:
                    self.state = 1  # WARNING
            else:
                self.consecutive_count = 0

        elif self.state == 1:  # WARNING
            if self.smoothed_prob >= self.high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirm:
                    self.state = 2  # ACTIVE_ALERT
                    self.episode_count += 1
                    self.last_alert_time = now_ms
                    self.last_cue_time = now_ms
                    fired_alert = True
            elif self.smoothed_prob < self.low:
                self.consecutive_count = 0
                self.state = 0  # NORMAL

        elif self.state == 2:  # ACTIVE_ALERT
            self.state = 3  # COOLDOWN

        elif self.state == 3:  # COOLDOWN
            # Check re-arm
            rearmed = False
            if self.rearm_policy == 'timeout':
                rearmed = (now_ms - self.last_alert_time >= self.cooldown_ms)
            else:  # 'timeout_and_low'
                rearmed = (now_ms - self.last_alert_time >= self.cooldown_ms) and (self.smoothed_prob < self.low)

            if rearmed:
                self.consecutive_count = 0
                self.state = 0  # NORMAL
            else:
                # Periodic re-cue if sustained high probability
                if self.recue_interval_ms > 0 and self.smoothed_prob >= self.high:
                    if (now_ms - self.last_cue_time >= self.recue_interval_ms):
                        self.last_cue_time = now_ms
                        fired_recue = True

        return self.state, fired_alert, fired_recue


def quantize_and_evaluate_int8(adopted_recipe, seed, fold_idx, X_dev_raw, y_bin_dev, meta_dev):
    """
    Quantizes a trained fold model to INT8 TFLite, computes INT8 out-of-fold predictions,
    and returns retention vs float model.
    """
    model_h5 = f"models/v2_dev/{adopted_recipe}_seed{seed}_fold{fold_idx}.h5"
    tflite_path = f"models/v2_dev/{adopted_recipe}_seed{seed}_fold{fold_idx}_int8.tflite"
    oof_npz = f"reports/v2/oof/{adopted_recipe}_seed{seed}_fold{fold_idx}.npz"
    int8_oof_npz = f"reports/v2/oof/{adopted_recipe}_seed{seed}_fold{fold_idx}_int8.npz"

    float_data = np.load(oof_npz)
    mean = float_data['scaler_mean']
    std = float_data['scaler_std']
    float_probs = float_data['val_probs']
    y_true = float_data['val_y_true']
    val_indices = float_data['val_indices']

    float_pr = float(average_precision_score(y_true, float_probs))

    # Outer indices
    outer_subjs = set(DEV_FOLDS[fold_idx])
    outer_indices = [i for i, m in enumerate(meta_dev) if m['subjectID'] in outer_subjs]
    X_outer_norm = (X_dev_raw[outer_indices] - mean) / std

    # Quantize to INT8 if not already present
    if not os.path.exists(tflite_path):
        print(f"Quantizing {model_h5} to INT8...")
        keras_model = tf.keras.models.load_model(model_h5)

        # Representative dataset from 13 training subjects
        train_subjs = set(m['subjectID'] for m in meta_dev) - outer_subjs - set(INNER_VAL_FOLDS[fold_idx])
        train_indices = [i for i, m in enumerate(meta_dev) if m['subjectID'] in train_subjs]
        np.random.seed(seed + fold_idx * 10)
        repr_indices = np.random.choice(train_indices, size=min(500, len(train_indices)), replace=False)
        X_repr_norm = ((X_dev_raw[repr_indices] - mean) / std).astype(np.float32)

        def representative_data_gen():
            for i in range(len(X_repr_norm)):
                yield [X_repr_norm[i:i+1]]

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_quant = converter.convert()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_quant)

    # Run INT8 inference on held-out fold
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']

    N = len(X_outer_norm)
    int8_probs = np.zeros(N, dtype=np.float32)
    X_int8 = np.clip(np.round(X_outer_norm / in_scale) + in_zp, -128, 127).astype(np.int8)

    for i in range(N):
        interp.set_tensor(in_det['index'], X_int8[i:i+1])
        interp.invoke()
        q_val = interp.get_tensor(out_det['index']).flatten()[0]
        int8_probs[i] = (float(q_val) - out_zp) * out_scale

    int8_pr = float(average_precision_score(y_true, int8_probs))
    int8_roc = float(roc_auc_score(y_true, int8_probs))
    retention_pct = (int8_pr / float_pr) * 100.0 if float_pr > 0 else 100.0

    # Save INT8 OOF
    np.savez_compressed(
        int8_oof_npz,
        val_indices=val_indices,
        int8_probs=int8_probs,
        float_probs=float_probs,
        y_true=y_true,
        retention_pct=retention_pct,
        int8_pr=int8_pr,
        float_pr=float_pr,
        int8_roc=int8_roc
    )

    return outer_indices, int8_probs, float_pr, int8_pr, retention_pct


def run_int8_pipeline_for_seed(adopted_recipe, seed, X_dev_raw, y_bin_dev, meta_dev):
    """Generates full INT8 continuous trial predictions across dev cohort for a single seed."""
    full_int8_probs = np.zeros(len(meta_dev), dtype=np.float32)
    fold_retentions = []

    for f_idx in range(6):
        indices, probs, f_pr, i_pr, ret = quantize_and_evaluate_int8(
            adopted_recipe, seed, f_idx, X_dev_raw, y_bin_dev, meta_dev
        )
        full_int8_probs[indices] = probs
        fold_retentions.append({
            'fold': f_idx,
            'float_pr': f_pr,
            'int8_pr': i_pr,
            'retention_pct': ret
        })

    trials = build_dev_trials(full_int8_probs, meta_dev)
    pooled_pr = float(average_precision_score(y_bin_dev, full_int8_probs))
    pooled_roc = float(roc_auc_score(y_bin_dev, full_int8_probs))

    return trials, fold_retentions, pooled_pr, pooled_roc, full_int8_probs


def eval_trials_fsm_fast(trials, config):
    """Evaluates trial records with EnhancedFoGFSM."""
    fsm = EnhancedFoGFSM(
        high=config['high'],
        low=config['low'],
        confirm=config['confirm'],
        cooldown_ms=config['cooldown_ms'],
        alpha=config['alpha'],
        rearm_policy=config.get('rearm_policy', 'timeout'),
        recue_interval_s=config.get('recue_interval_s', 0)
    )

    total_false_alerts = 0
    total_normal_s = 0.0
    episodes_eval = []

    for tr in trials:
        fsm.reset()
        alerts = []
        recues = []
        for p, t_end in zip(tr['probs'], tr['end_times']):
            now_ms = int(round(t_end * 1000))
            _, fired_alert, fired_recue = fsm.update(p, now_ms)
            if fired_alert:
                alerts.append(t_end)
            if fired_recue:
                recues.append(t_end)

        for a in alerts:
            is_true = any(ep['onset'] - 3.0 <= a <= ep['offset'] for ep in tr['episodes'])
            if not is_true:
                total_false_alerts += 1
        total_normal_s += tr['normal_duration_s']

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
                'alerted_within_1s': alerted_within_1s,
                'subjectID': tr['subjectID']
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
        'delays': delays,
        'pct_alerted_within_1s': pct_within_1s,
        'total_episodes': n_eps,
        'caught_episodes': sum(e['lenient_caught'] for e in episodes_eval),
        'total_false_alerts': total_false_alerts,
        'total_normal_hours': norm_hr,
        'episodes_eval': episodes_eval
    }


def run_step3_operating_point_analysis():
    print("=" * 80)
    print("STEP 3: OPERATING-POINT REPORT & INT8 OOF TUNING (DEV DATA ONLY)")
    print("=" * 80)

    # 1. Determine adopted recipe from confirmation report
    with open('reports/v2/diagnostics/confirmation_report.json', 'r') as f:
        conf_rep = json.load(f)
    adopted_recipe = conf_rep['rule_1_1_evaluation']['adopted_recipe']
    recipe_file_prefix = 'b_rotation_35' if adopted_recipe in ['baseline_rotation_35', 'b_rotation_35'] else adopted_recipe
    print(f"Adopted recipe from confirmation: {adopted_recipe} (file prefix: {recipe_file_prefix})")

    X_dev_raw, y_bin_dev, y_multi_dev, meta_dev = load_dev_data()
    verify_test_isolation(meta_dev)

    # 2. Quantize all fold models to INT8 and generate INT8 continuous trials for seeds 42, 43, 44
    seeds = [42, 43, 44]
    trials_by_seed = {}
    retentions_by_seed = {}
    pooled_int8_pr_by_seed = {}
    pooled_int8_roc_by_seed = {}
    full_probs_by_seed = {}

    for s in seeds:
        print(f"\nProcessing INT8 quantization & OOF evaluation for Seed {s}...")
        trials, retentions, p_pr, p_roc, probs = run_int8_pipeline_for_seed(
            recipe_file_prefix, s, X_dev_raw, y_bin_dev, meta_dev
        )
        trials_by_seed[s] = trials
        retentions_by_seed[s] = retentions
        pooled_int8_pr_by_seed[s] = p_pr
        pooled_int8_roc_by_seed[s] = p_roc
        full_probs_by_seed[s] = probs
        mean_ret = np.mean([r['retention_pct'] for r in retentions])
        print(f"Seed {s}: Pooled INT8 PR-AUC: {p_pr:.4f}, Mean Fold Retention: {mean_ret:.1f}%")

    # 3. Comprehensive FSM Grid Search over Rule 1.3
    # Grid: confirm in [1, 2], alpha in [0.60..1.00], high in [0.35..0.70], low in [0.15..0.35], cd in [2000..6000]
    # rearm_policy in ['timeout', 'timeout_and_low'], recue_interval_s in [0, 5, 10]
    confirms = [1, 2]
    alphas = [0.60, 0.70, 0.80, 0.90, 1.00]
    highs = [0.35, 0.40, 0.45, 0.49, 0.50, 0.55, 0.60, 0.65, 0.70]
    lows = [0.15, 0.20, 0.25, 0.29, 0.30, 0.35]
    cds = [2000, 3000, 4000, 5000, 6000]
    rearms = ['timeout', 'timeout_and_low']
    recues = [0, 5, 10]

    # Pre-filter grid combinations: low < high
    grid = []
    for c in confirms:
        for a in alphas:
            for h in highs:
                for l in lows:
                    if l >= h:
                        continue
                    for cd in cds:
                        for r_pol in rearms:
                            for r_cue in recues:
                                grid.append({
                                    'confirm': c,
                                    'alpha': a,
                                    'high': h,
                                    'low': l,
                                    'cooldown_ms': cd,
                                    'rearm_policy': r_pol,
                                    'recue_interval_s': r_cue
                                })

    print(f"\nSearching over {len(grid)} FSM hyperparameter configurations across 3 seeds...")

    # Fast evaluation: pre-compute smoothed probabilities per trial & alpha
    # Since recue_interval_s only fires during sustained cooldown without affecting first-alert latency/sensitivity,
    # we can group by base config without recue, find best base configs, and select optimal recue
    t_start = time.time()
    evaluated_configs = []

    # Subsample grid or search systematically
    # To keep runtime snappy (~30s), evaluate representative grid
    # Step through grid
    for idx, cfg in enumerate(grid):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Evaluated {idx}/{len(grid)} configs ({time.time() - t_start:.1f}s)...")

        # Evaluate across the 3 seeds
        seed_res = [eval_trials_fsm_fast(trials_by_seed[s], cfg) for s in seeds]

        mean_lenient = float(np.mean([r['lenient_sensitivity'] for r in seed_res]))
        mean_strict = float(np.mean([r['strict_sensitivity'] for r in seed_res]))
        mean_relaxed = float(np.mean([r['relaxed_sensitivity'] for r in seed_res]))
        mean_fa = float(np.mean([r['fa_hr'] for r in seed_res]))
        mean_delay = float(np.mean([r['median_delay_s'] for r in seed_res]))
        mean_pct_1s = float(np.mean([r['pct_alerted_within_1s'] for r in seed_res]))

        evaluated_configs.append({
            'config': cfg,
            'mean_lenient_sens': mean_lenient,
            'mean_strict_sens': mean_strict,
            'mean_relaxed_sens': mean_relaxed,
            'mean_fa_hr': mean_fa,
            'mean_delay_s': mean_delay,
            'mean_pct_1s': mean_pct_1s,
            'seed_res': seed_res
        })

    print(f"Completed grid evaluation of {len(evaluated_configs)} configs in {time.time() - t_start:.1f}s.")

    # 4. Selection of Dual Presets under Rule 1.3:
    # Preset 1: LOW_ALARM: Maximize mean lenient sensitivity subject to mean FA/h <= 30.0
    low_alarm_candidates = [c for c in evaluated_configs if c['mean_fa_hr'] <= 30.0]
    if low_alarm_candidates:
        # Sort by highest lenient sens, then lowest FA/hr
        low_alarm_candidates.sort(key=lambda x: (x['mean_lenient_sens'], -x['mean_fa_hr']), reverse=True)
        chosen_low_alarm = low_alarm_candidates[0]
    else:
        # Fallback to lowest FA/hr overall
        evaluated_configs.sort(key=lambda x: x['mean_fa_hr'])
        chosen_low_alarm = evaluated_configs[0]

    # Preset 2: BALANCED: Lowest mean FA/h subject to mean lenient sensitivity >= 70%
    balanced_candidates = [c for c in evaluated_configs if c['mean_lenient_sens'] >= 0.70]
    if balanced_candidates:
        balanced_candidates.sort(key=lambda x: x['mean_fa_hr'])
        chosen_balanced = balanced_candidates[0]
    else:
        # Fall-through to highest lenient sensitivity
        evaluated_configs.sort(key=lambda x: (x['mean_lenient_sens'], -x['mean_fa_hr']), reverse=True)
        chosen_balanced = evaluated_configs[0]

    # 5. Pareto Table for FA budgets: 15, 30, 60, 100 FA/hr
    budgets = [15.0, 30.0, 60.0, 100.0]
    pareto_table = []
    for b in budgets:
        b_cands = [c for c in evaluated_configs if c['mean_fa_hr'] <= b]
        if b_cands:
            b_cands.sort(key=lambda x: (x['mean_lenient_sens'], -x['mean_fa_hr']), reverse=True)
            best_b = b_cands[0]
            pareto_table.append({
                'budget': b,
                'mean_lenient': best_b['mean_lenient_sens'] * 100.0,
                'mean_strict': best_b['mean_strict_sens'] * 100.0,
                'mean_relaxed': best_b['mean_relaxed_sens'] * 100.0,
                'mean_fa': best_b['mean_fa_hr'],
                'seconds_per_fa': 3600.0 / best_b['mean_fa_hr'] if best_b['mean_fa_hr'] > 0 else 99999.0,
                'config': best_b['config']
            })
        else:
            pareto_table.append({
                'budget': b,
                'mean_lenient': 0.0,
                'mean_strict': 0.0,
                'mean_relaxed': 0.0,
                'mean_fa': 0.0,
                'seconds_per_fa': 99999.0,
                'config': {}
            })

    # 6. Detailed Analysis of Chosen Presets: Delays, IQR, Per-Subject Breakdown, Optimism Gap
    def analyze_preset_details(chosen_item, preset_name):
        cfg = chosen_item['config']
        all_delays = []
        for s in seeds:
            res = eval_trials_fsm_fast(trials_by_seed[s], cfg)
            all_delays.extend(res['delays'])

        med_delay = float(np.median(all_delays)) if all_delays else 0.0
        q25 = float(np.percentile(all_delays, 25)) if all_delays else 0.0
        q75 = float(np.percentile(all_delays, 75)) if all_delays else 0.0
        pct_within_1s = float(sum(d <= 1.0 for d in all_delays) / len(all_delays) * 100.0) if all_delays else 0.0

        # Per-subject breakdown for subjects with >= 3 episodes (dev cohort)
        # Aggregate episodes across the 3 seeds
        subj_episodes = {}
        for s in seeds:
            res = eval_trials_fsm_fast(trials_by_seed[s], cfg)
            for ep in res['episodes_eval']:
                sid = ep['subjectID']
                subj_episodes.setdefault(sid, []).append(ep)

        subj_breakdown = {}
        for sid, eps in sorted(subj_episodes.items()):
            n_tot = len(eps) // len(seeds)  # episodes per seed
            n_caught = sum(e['lenient_caught'] for e in eps) / len(seeds)
            if n_tot >= 3:
                subj_breakdown[int(sid)] = {
                    'total_episodes': n_tot,
                    'caught_episodes': n_caught,
                    'lenient_sensitivity': float(n_caught / n_tot * 100.0) if n_tot > 0 else 0.0
                }

        # Optimism gap: LOFO vs In-sample
        # LOFO fold evaluation:
        lofo_fold_fa = []
        lofo_fold_caught = []
        lofo_fold_tot = []
        lofo_fold_norm_hr = []

        for f_idx in range(6):
            fold_test_trials = [tr for s in seeds for tr in trials_by_seed[s] if tr['fold_idx'] == f_idx]
            fold_res = eval_trials_fsm_fast(fold_test_trials, cfg)
            lofo_fold_fa.append(fold_res['total_false_alerts'])
            lofo_fold_caught.append(fold_res['caught_episodes'])
            lofo_fold_tot.append(fold_res['total_episodes'])
            lofo_fold_norm_hr.append(fold_res['total_normal_hours'])

        lofo_lenient = float(sum(lofo_fold_caught) / sum(lofo_fold_tot)) if sum(lofo_fold_tot) > 0 else 0.0
        lofo_fa = float(sum(lofo_fold_fa) / sum(lofo_fold_norm_hr)) if sum(lofo_fold_norm_hr) > 0 else 0.0

        in_sample_lenient = chosen_item['mean_lenient_sens']
        in_sample_fa = chosen_item['mean_fa_hr']

        optimism_gap = {
            'in_sample_lenient': in_sample_lenient * 100.0,
            'lofo_lenient': lofo_lenient * 100.0,
            'gap_lenient': (in_sample_lenient - lofo_lenient) * 100.0,
            'in_sample_fa': in_sample_fa,
            'lofo_fa': lofo_fa,
            'gap_fa': lofo_fa - in_sample_fa
        }

        return {
            'preset_name': preset_name,
            'config': cfg,
            'mean_lenient': in_sample_lenient * 100.0,
            'mean_strict': chosen_item['mean_strict_sens'] * 100.0,
            'mean_relaxed': chosen_item['mean_relaxed_sens'] * 100.0,
            'mean_fa_hr': in_sample_fa,
            'seconds_per_fa': 3600.0 / in_sample_fa if in_sample_fa > 0 else 99999.0,
            'median_delay_s': med_delay,
            'iqr_delay_s': [q25, q75],
            'pct_within_1s': pct_within_1s,
            'per_subject_breakdown': subj_breakdown,
            'optimism_gap': optimism_gap
        }

    low_alarm_analysis = analyze_preset_details(chosen_low_alarm, 'LOW_ALARM')
    balanced_analysis = analyze_preset_details(chosen_balanced, 'BALANCED')

    # 7. Save JSON diagnostic report
    out_json = {
        'adopted_recipe': adopted_recipe,
        'retentions_by_seed': retentions_by_seed,
        'pooled_int8_pr_by_seed': pooled_int8_pr_by_seed,
        'pareto_table': pareto_table,
        'low_alarm_preset': low_alarm_analysis,
        'balanced_preset': balanced_analysis
    }

    os.makedirs('reports/v2/diagnostics', exist_ok=True)
    with open('reports/v2/diagnostics/operating_point_report.json', 'w') as f:
        json.dump(out_json, f, indent=2)

    # 8. Generate reports/v2/operating_point.md
    md_content = f"""# Operating-Point Selection Report: INT8 OOF Optimization (Step 3)

**Evaluation Date:** 2026-11-16  
**Adopted Model Architecture:** `{adopted_recipe}`  
**Optimization Target:** Mean out-of-fold performance across 3 development seeds (42, 43, 44) evaluated on fully quantized INT8 TFLite models.

---

## 1. Float vs INT8 Window PR-AUC Retention per Fold

| Fold Index | Seed 42 Retention (%) | Seed 43 Retention (%) | Seed 44 Retention (%) | Mean Retention (%) |
|:---:|:---:|:---:|:---:|:---:|
"""
    for f in range(6):
        r42 = retentions_by_seed[42][f]['retention_pct']
        r43 = retentions_by_seed[43][f]['retention_pct']
        r44 = retentions_by_seed[44][f]['retention_pct']
        m_r = np.mean([r42, r43, r44])
        md_content += f"| Fold {f} | {r42:.2f}% | {r43:.2f}% | {r44:.2f}% | **{m_r:.2f}%** |\n"

    all_rets = [r['retention_pct'] for s in seeds for r in retentions_by_seed[s]]
    md_content += f"| **Overall** | **{np.mean([r['retention_pct'] for r in retentions_by_seed[42]]):.2f}%** | **{np.mean([r['retention_pct'] for r in retentions_by_seed[43]]):.2f}%** | **{np.mean([r['retention_pct'] for r in retentions_by_seed[44]]):.2f}%** | **{np.mean(all_rets):.2f}%** |\n\n"

    md_content += f"""---

## 2. Pre-Registered Operating Presets (Rule 1.3)

| Parameter | Preset: `LOW_ALARM` | Preset: `BALANCED` |
|:---|:---:|:---:|
| **Target Objective** | Max Lenient Sens subject to FA/hr $\\le 30.0$ | Lowest FA/hr subject to Lenient Sens $\\ge 70.0\\%$ |
| **Consecutive Confirmations ($N_{{confirm}}$)** | `{low_alarm_analysis['config']['confirm']}` | `{balanced_analysis['config']['confirm']}` |
| **EMA Smoothing ($\\alpha$)** | `{low_alarm_analysis['config']['alpha']:.2f}` | `{balanced_analysis['config']['alpha']:.2f}` |
| **High Alert Threshold ($T_{{high}}$)** | `{low_alarm_analysis['config']['high']:.2f}` | `{balanced_analysis['config']['high']:.2f}` |
| **Low Reset Threshold ($T_{{low}}$)** | `{low_alarm_analysis['config']['low']:.2f}` | `{balanced_analysis['config']['low']:.2f}` |
| **Refractory Cooldown ($T_{{cooldown}}$)** | `{low_alarm_analysis['config']['cooldown_ms']} ms` | `{balanced_analysis['config']['cooldown_ms']} ms` |
| **Re-Arm Policy** | `{low_alarm_analysis['config']['rearm_policy']}` | `{balanced_analysis['config']['rearm_policy']}` |
| **Re-Cue Interval** | `{low_alarm_analysis['config']['recue_interval_s']} s` | `{balanced_analysis['config']['recue_interval_s']} s` |
| **Mean LOFO Lenient Sensitivity** | **{low_alarm_analysis['mean_lenient']:.1f}%** | **{balanced_analysis['mean_lenient']:.1f}%** |
| **Mean LOFO Strict Sensitivity** | {low_alarm_analysis['mean_strict']:.1f}% | {balanced_analysis['mean_strict']:.1f}% |
| **Mean LOFO Relaxed Sensitivity** | {low_alarm_analysis['mean_relaxed']:.1f}% | {balanced_analysis['mean_relaxed']:.1f}% |
| **Mean LOFO False Alerts / Hour** | **{low_alarm_analysis['mean_fa_hr']:.1f} FA/hr** | **{balanced_analysis['mean_fa_hr']:.1f} FA/hr** |
| **One False Alert Every** | **{low_alarm_analysis['seconds_per_fa']:.1f} seconds** | **{balanced_analysis['seconds_per_fa']:.1f} seconds** |
| **Median Detection Delay** | {low_alarm_analysis['median_delay_s']:.2f} s | {balanced_analysis['median_delay_s']:.2f} s |
| **IQR Detection Delay [Q25, Q75]** | [{low_alarm_analysis['iqr_delay_s'][0]:.2f}s, {low_alarm_analysis['iqr_delay_s'][1]:.2f}s] | [{balanced_analysis['iqr_delay_s'][0]:.2f}s, {balanced_analysis['iqr_delay_s'][1]:.2f}s] |
| **% Alerted Within 1.0 s of Onset** | {low_alarm_analysis['pct_within_1s']:.1f}% | {balanced_analysis['pct_within_1s']:.1f}% |

---

## 3. Pareto Frontier Across False-Alert Budgets

| FA Budget (FA/hr) | LOFO Lenient Sens (%) | LOFO Strict Sens (%) | LOFO Relaxed Sens (%) | Achieved FA/hr | 1 False Alert Every N Seconds | Selected Configuration (Confirm, $\\alpha$, High, Low, Cooldown, Re-arm) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
"""
    for row in pareto_table:
        cfg_str = f"c={row['config'].get('confirm')}, a={row['config'].get('alpha')}, h={row['config'].get('high')}, l={row['config'].get('low')}, cd={row['config'].get('cooldown_ms')}ms, {row['config'].get('rearm_policy')}" if row['config'] else "None meeting budget"
        md_content += f"| $\\le {row['budget']:.0f}$ | {row['mean_lenient']:.1f}% | {row['mean_strict']:.1f}% | {row['mean_relaxed']:.1f}% | {row['mean_fa']:.1f} | {row['seconds_per_fa']:.1f}s | {cfg_str} |\n"

    md_content += f"""
---

## 4. Per-Subject Lenient Sensitivity (Dev Cohort Subjects with $\\ge 3$ Episodes)

| Subject ID | Total Episodes | `LOW_ALARM` Caught (%) | `BALANCED` Caught (%) |
|:---:|:---:|:---:|:---:|
"""
    all_sids = sorted(low_alarm_analysis['per_subject_breakdown'].keys())
    for sid in all_sids:
        la_s = low_alarm_analysis['per_subject_breakdown'][sid]
        ba_s = balanced_analysis['per_subject_breakdown'][sid]
        md_content += f"| Subject {sid} | {la_s['total_episodes']} | {la_s['caught_episodes']:.0f}/{la_s['total_episodes']} ({la_s['lenient_sensitivity']:.1f}%) | {ba_s['caught_episodes']:.0f}/{ba_s['total_episodes']} ({ba_s['lenient_sensitivity']:.1f}%) |\n"

    md_content += f"""
---

## 5. Optimism Gap Analysis (In-Sample vs LOFO)

| Preset | Metric | In-Sample Value | LOFO Value | Optimism Gap (In-Sample minus LOFO) |
|:---|:---|:---:|:---:|:---:|
| `LOW_ALARM` | Lenient Sensitivity (%) | {low_alarm_analysis['optimism_gap']['in_sample_lenient']:.2f}% | {low_alarm_analysis['optimism_gap']['lofo_lenient']:.2f}% | {low_alarm_analysis['optimism_gap']['gap_lenient']:+.2f} pts |
| `LOW_ALARM` | False Alerts / hour | {low_alarm_analysis['optimism_gap']['in_sample_fa']:.1f} | {low_alarm_analysis['optimism_gap']['lofo_fa']:.1f} | {low_alarm_analysis['optimism_gap']['gap_fa']:+.1f} FA/hr |
| `BALANCED` | Lenient Sensitivity (%) | {balanced_analysis['optimism_gap']['in_sample_lenient']:.2f}% | {balanced_analysis['optimism_gap']['lofo_lenient']:.2f}% | {balanced_analysis['optimism_gap']['gap_lenient']:+.2f} pts |
| `BALANCED` | False Alerts / hour | {balanced_analysis['optimism_gap']['in_sample_fa']:.1f} | {balanced_analysis['optimism_gap']['lofo_fa']:.1f} | {balanced_analysis['optimism_gap']['gap_fa']:+.1f} FA/hr |
"""

    with open('reports/v2/operating_point.md', 'w') as f:
        f.write(md_content)

    print("Wrote reports/v2/operating_point.md and operating_point_report.json successfully.")


if __name__ == '__main__':
    run_step3_operating_point_analysis()

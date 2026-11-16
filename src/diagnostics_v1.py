"""
Diagnostics for v1-submitted models on validation and held-out test cohorts.

Performs:
1. Missed-episode analysis & root cause classification on held-out test cohort.
2. Time-to-onset probability trajectories in 0.5s bins ([-3.0s, +2.0s]).
3. AUC (ROC and PR) by clinical label groups:
   (i)   Pure Pre-FoG (1) vs Normal (0)
   (ii)  Immediate Transition (-1) vs Normal (0)
   (iii) Active Freeze (2) vs Normal (0)
4. Full alert and FSM state histories for locked trials (Sub 20 Task 6, Sub 21 Task 7).

Saves all outputs to JSON in reports/v1_submitted/diagnostics/.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from eval_pipeline import FoGFSM, extract_trial_records, predict_tflite_int8


def compute_auc_pair(y_multi, probs, target_label, normal_label=0):
    mask = np.isin(y_multi, [normal_label, target_label])
    y_sub = (y_multi[mask] == target_label).astype(int)
    p_sub = probs[mask]
    if len(np.unique(y_sub)) < 2:
        return 0.0, 0.0
    roc = float(roc_auc_score(y_sub, p_sub))
    prec, rec, _ = precision_recall_curve(y_sub, p_sub)
    pr = float(auc(rec, prec))
    return roc, pr


def run_all_v1_diagnostics(dataset_npz="data/processed/fog_dataset.npz",
                           test_meta_json="data/processed/test_meta.json",
                           val_meta_json="data/processed/val_meta.json",
                           raw_csv_path="data/raw/sensor_data.csv",
                           keras_model_path="models/best_model/selected_best_model.h5",
                           tflite_model_path="models/esp32/best_model_int8.tflite",
                           output_dir="reports/v1_submitted/diagnostics"):
    os.makedirs(output_dir, exist_ok=True)

    ds = np.load(dataset_npz)
    X_val = ds['X_val']
    y_val_bin = ds['y_val_bin']
    y_val_multi = ds['y_val_multi']
    X_test = ds['X_test']
    y_test_bin = ds['y_test_bin']
    y_test_multi = ds['y_test_multi']

    with open(val_meta_json, 'r') as f:
        val_meta = json.load(f)
    with open(test_meta_json, 'r') as f:
        test_meta = json.load(f)

    # 1. Model Inferences
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    tflite_val_probs = predict_tflite_int8(interpreter, X_val)
    tflite_test_probs = predict_tflite_int8(interpreter, X_test)

    k_model = tf.keras.models.load_model(keras_model_path, compile=False)
    k_val_probs = k_model.predict(X_val, verbose=0).ravel()
    k_test_probs = k_model.predict(X_test, verbose=0).ravel()

    channels = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z',
                'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']
    trials = extract_trial_records(raw_csv_path, test_meta, tflite_test_probs, channels)
    trials_k = extract_trial_records(raw_csv_path, test_meta, k_test_probs, channels)

    fsm_cfg = {'high': 0.49, 'low': 0.29, 'confirm': 2, 'cooldown_ms': 3000, 'alpha': 0.60}
    st_names = ['NORMAL', 'WARNING', 'ACTIVE_ALERT', 'COOLDOWN']

    # ==========================================
    # 1. Missed Episode Analysis
    # ==========================================
    all_ep_records = []
    trial_step_logs = {}

    for tr in trials:
        fsm = FoGFSM(**fsm_cfg)
        step_log = []
        alerts = []
        for p, t_end in zip(tr['probs'], tr['end_times']):
            now_ms = int(round(t_end * 1000))
            sb = fsm.state
            sa, fired = fsm.update(p, now_ms)
            step_log.append({
                't_end': float(t_end),
                'now_ms': int(now_ms),
                'raw_p': float(p),
                'ema_p': float(fsm.smoothed_prob),
                'st_before': int(sb),
                'st_after': int(sa),
                'count': int(fsm.consecutive_count),
                'fired': bool(fired)
            })
            if fired:
                alerts.append(float(t_end))

        trial_step_logs[tr['key']] = step_log

        for ep in tr['episodes']:
            onset = float(ep['onset'])
            offset = float(ep['offset'])
            dur = float(offset - onset)
            hazard_steps = [s for s in step_log if (onset - 3.0) <= s['t_end'] <= offset]
            raw_vals = [s['raw_p'] for s in hazard_steps]
            ema_vals = [s['ema_p'] for s in hazard_steps]
            max_raw = float(max(raw_vals)) if raw_vals else 0.0
            max_ema = float(max(ema_vals)) if ema_vals else 0.0

            ep_alerts = [a for a in alerts if (onset - 3.0) <= a <= offset]
            lenient = len(ep_alerts) > 0

            # Cause classification
            cooldown_blocking = any(s['st_before'] == 3 and (s['raw_p'] >= 0.49 or s['ema_p'] >= 0.49) for s in hazard_steps)
            crossed_ema = any(s['ema_p'] >= 0.49 for s in hazard_steps)
            crossed_raw = any(s['raw_p'] >= 0.49 for s in hazard_steps)

            if cooldown_blocking:
                cause = "(c) FSM was in COOLDOWN from an earlier alert"
            elif not crossed_raw and not crossed_ema:
                cause = "(a) never crossed the high threshold (raw prob peaked below 0.49)"
            elif crossed_raw and not crossed_ema:
                cause = "(a) never crossed the high threshold (raw crossed 0.49 but EMA kept it below 0.49)"
            elif crossed_ema:
                cause = "(b) crossed but freeze ended before the confirmations completed"
            else:
                cause = "(d) other"

            all_ep_records.append({
                'subjectID': int(ep['subjectID']),
                'sessionID': int(ep['sessionID']),
                'taskID': int(ep['taskID']),
                'onset': onset,
                'offset': offset,
                'duration': dur,
                'max_raw_prob': max_raw,
                'max_ema_prob': max_ema,
                'lenient_caught': lenient,
                'root_cause': cause,
                'hazard_steps': hazard_steps
            })

    durs = [e['duration'] for e in all_ep_records]
    missed_eps = [e for e in all_ep_records if not e['lenient_caught']]

    missed_episodes_data = {
        'total_episodes': len(all_ep_records),
        'duration_distribution_s': {
            'min': float(np.min(durs)),
            'p25': float(np.percentile(durs, 25)),
            'median': float(np.median(durs)),
            'p75': float(np.percentile(durs, 75)),
            'max': float(np.max(durs)),
            'mean': float(np.mean(durs))
        },
        'missed_count': len(missed_eps),
        'missed_episodes': [
            {
                'subjectID': m['subjectID'],
                'sessionID': m['sessionID'],
                'taskID': m['taskID'],
                'onset': m['onset'],
                'offset': m['offset'],
                'duration': m['duration'],
                'max_raw_prob': m['max_raw_prob'],
                'max_ema_prob': m['max_ema_prob'],
                'root_cause': m['root_cause'],
                'hazard_timeline': m['hazard_steps']
            } for m in missed_eps
        ]
    }

    with open(os.path.join(output_dir, 'missed_episodes.json'), 'w') as f:
        json.dump(missed_episodes_data, f, indent=2)

    # ==========================================
    # 2. AUC by Label Group
    # ==========================================
    auc_data = {}
    for s_name, ym, kp, ip in [
        ("validation", y_val_multi, k_val_probs, tflite_val_probs),
        ("test", y_test_multi, k_test_probs, tflite_test_probs)
    ]:
        k_r1, k_p1 = compute_auc_pair(ym, kp, 1, 0)
        i_r1, i_p1 = compute_auc_pair(ym, ip, 1, 0)
        k_rt, k_pt = compute_auc_pair(ym, kp, -1, 0)
        i_rt, i_pt = compute_auc_pair(ym, ip, -1, 0)
        k_r2, k_p2 = compute_auc_pair(ym, kp, 2, 0)
        i_r2, i_p2 = compute_auc_pair(ym, ip, 2, 0)

        auc_data[s_name] = {
            'window_counts': {
                'total': len(ym),
                'normal_0': int(np.sum(ym == 0)),
                'pre_fog_1': int(np.sum(ym == 1)),
                'transition_minus1': int(np.sum(ym == -1)),
                'active_fog_2': int(np.sum(ym == 2))
            },
            'pre_fog_1_vs_normal_0': {
                'keras_float': {'roc_auc': k_r1, 'pr_auc': k_p1},
                'int8_tflite': {'roc_auc': i_r1, 'pr_auc': i_p1}
            },
            'transition_minus1_vs_normal_0': {
                'keras_float': {'roc_auc': k_rt, 'pr_auc': k_pt},
                'int8_tflite': {'roc_auc': i_rt, 'pr_auc': i_pt}
            },
            'active_fog_2_vs_normal_0': {
                'keras_float': {'roc_auc': k_r2, 'pr_auc': k_p2},
                'int8_tflite': {'roc_auc': i_r2, 'pr_auc': i_p2}
            }
        }

    with open(os.path.join(output_dir, 'auc_by_label_group.json'), 'w') as f:
        json.dump(auc_data, f, indent=2)

    # ==========================================
    # 3. Time-to-Onset Distribution
    # ==========================================
    bins = [(-3.0, -2.5), (-2.5, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5),
            (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    bin_data_keras = {b: [] for b in bins}
    bin_data_int8 = {b: [] for b in bins}

    for tr_k, tr_i in zip(trials_k, trials):
        for ep in tr_k['episodes']:
            onset = ep['onset']
            for p_k, p_i, t_end in zip(tr_k['probs'], tr_i['probs'], tr_k['end_times']):
                dt = t_end - onset
                for b in bins:
                    if b[0] <= dt < b[1]:
                        bin_data_keras[b].append(float(p_k))
                        bin_data_int8[b].append(float(p_i))

    time_to_onset_data = []
    for b in bins:
        k_vals = bin_data_keras[b]
        i_vals = bin_data_int8[b]
        k_mean = float(np.mean(k_vals)) if k_vals else 0.0
        i_mean = float(np.mean(i_vals)) if i_vals else 0.0
        phase = "Active Freeze" if b[0] >= 0.0 else ("Immediate Transition (<0.5s)" if b[0] >= -0.5 else "Pure Pre-FoG (>0.5s)")
        time_to_onset_data.append({
            'bin_start_s': b[0],
            'bin_end_s': b[1],
            'window_count': len(k_vals),
            'mean_prob_keras_float': k_mean,
            'mean_prob_int8_tflite': i_mean,
            'clinical_phase': phase
        })

    with open(os.path.join(output_dir, 'time_to_onset_distribution.json'), 'w') as f:
        json.dump(time_to_onset_data, f, indent=2)

    # ==========================================
    # 4. Locked Trials History
    # ==========================================
    locked_trials_info = []
    for subj, task in [(20, 6), (21, 7)]:
        tr = [t for t in trials if t['subjectID'] == subj and t['taskID'] == task][0]
        step_log = trial_step_logs[tr['key']]

        alerts = []
        for s in step_log:
            if s['fired']:
                in_haz = any(ep['onset'] - 3.0 <= s['t_end'] <= ep['offset'] for ep in tr['episodes'])
                alerts.append({
                    'time_s': s['t_end'],
                    'raw_prob': s['raw_p'],
                    'ema_prob': s['ema_p'],
                    'is_true_alert': bool(in_haz)
                })

        locked_trials_info.append({
            'subjectID': subj,
            'sessionID': tr['sessionID'],
            'taskID': task,
            'episodes': [
                {
                    'onset': ep['onset'],
                    'offset': ep['offset'],
                    'duration': ep['offset'] - ep['onset'],
                    'hazard_start': ep['onset'] - 3.0
                } for ep in tr['episodes']
            ],
            'alerts_fired': alerts,
            'step_log': step_log
        })

    with open(os.path.join(output_dir, 'locked_trials_history.json'), 'w') as f:
        json.dump(locked_trials_info, f, indent=2)

    # Consolidate top-level summary
    summary_data = {
        'missed_episodes_count': len(missed_eps),
        'total_episodes': len(all_ep_records),
        'auc_by_label_group': auc_data,
        'time_to_onset_distribution': time_to_onset_data
    }
    with open(os.path.join(output_dir, 'diagnostics_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"[DIAGNOSTICS v1] Complete diagnostics written to {output_dir}")
    return summary_data


if __name__ == '__main__':
    run_all_v1_diagnostics()

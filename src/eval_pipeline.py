"""
End-to-End Pipeline Evaluation for ESP32 Freezing-of-Gait Prediction.

Evaluates trained models (Keras .h5 and INT8 .tflite) combined with an exact
Python port of the firmware finite state machine (FSM) across continuous trials.
Computes both window-level and clinical episode-level metrics:
- False alerts per hour of normal gait
- STRICT episode sensitivity (alert in [onset - 3.0s, onset - 0.5s])
- LENIENT episode sensitivity (alert in [onset - 3.0s, offset])
- Lead-time distribution (mean, median, % >= 0.5s early)
- Per-subject breakdown with episode counts
- Window-level PR-AUC, ROC-AUC, F1, sensitivity, specificity
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from preprocess import resample_trial


class FoGFSM:
    """
    Exact Python port of firmware/ESP32_FoG/fog_detection.cpp
    4-State Machine: NORMAL (0) -> WARNING (1) -> ACTIVE_ALERT (2) -> COOLDOWN (3) -> NORMAL (0).
    """
    def __init__(self, high=0.49, low=0.29, confirm=2, cooldown_ms=3000, alpha=0.60):
        self.high = high
        self.low = low
        self.confirm = confirm
        self.cooldown_ms = cooldown_ms
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.state = 0  # 0: NORMAL, 1: WARNING, 2: ACTIVE_ALERT, 3: COOLDOWN
        self.smoothed_prob = 0.0
        self.last_raw_prob = 0.0
        self.consecutive_count = 0
        self.episode_count = 0
        self.last_alert_time = 0

    def update(self, raw_prob, now_ms):
        self.last_raw_prob = raw_prob
        # 1. EMA smoothing
        self.smoothed_prob = (self.alpha * raw_prob) + ((1.0 - self.alpha) * self.smoothed_prob)

        fired_alert = False
        # 2. State transitions
        if self.state == 0:  # NORMAL
            if self.smoothed_prob >= self.high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirm:
                    self.state = 2  # ACTIVE_ALERT
                    self.episode_count += 1
                    self.last_alert_time = now_ms
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
                    fired_alert = True
            elif self.smoothed_prob < self.low:
                self.consecutive_count = 0
                self.state = 0  # NORMAL

        elif self.state == 2:  # ACTIVE_ALERT
            self.state = 3  # Transition to COOLDOWN on the subsequent step

        elif self.state == 3:  # COOLDOWN
            if (now_ms - self.last_alert_time >= self.cooldown_ms) and (self.smoothed_prob < self.low):
                self.consecutive_count = 0
                self.state = 0  # NORMAL

        return self.state, fired_alert


def extract_trial_records(raw_csv_path, meta_list, probs, channels):
    """
    Groups window inferences into continuous recording trials (subjectID, sessionID, taskID),
    ordered chronologically by start_sample, and extracts ground-truth freeze episodes.
    """
    df = pd.read_csv(raw_csv_path)
    
    trials_dict = {}
    for idx, m in enumerate(meta_list):
        key = (int(m['subjectID']), int(m['sessionID']), int(m['taskID']))
        trials_dict.setdefault(key, []).append((idx, m))
        
    trial_records = []
    
    for key, indexed_wins in sorted(trials_dict.items()):
        subj, sess, task = key
        # Order windows chronologically by start_sample
        indexed_wins.sort(key=lambda x: x[1]['start_sample'])
        
        t_df = df[(df['subjectID'] == subj) & (df['sessionID'] == sess) & (df['taskID'] == task)]
        sig, fog, sev = resample_trial(t_df, channels)
        if fog is None or len(sig) == 0:
            continue
            
        fog_padded = np.pad(fog, (1, 1), 'constant', constant_values=0)
        diffs = np.diff(fog_padded)
        raw_onsets = np.where(diffs == 1)[0] / 50.0
        raw_offsets = np.where(diffs == -1)[0] / 50.0
        trial_dur_s = len(sig) / 50.0
        
        # Cluster window metadata to identify freeze episodes
        episodes_meta = []
        curr = []
        for idx, m in indexed_wins:
            if m['multi_label'] != 0:
                curr.append((idx, m))
            else:
                if curr and any(w[1]['multi_label'] == 2 for w in curr):
                    episodes_meta.append(curr)
                curr = []
        if curr and any(w[1]['multi_label'] == 2 for w in curr):
            episodes_meta.append(curr)
            
        ep_ranges = []
        for ep in episodes_meta:
            ep_start = ep[0][1]['start_time_s']
            ep_end = ep[-1][1]['end_time_s']
            matched = [(on, off) for on, off in zip(raw_onsets, raw_offsets) if on <= ep_end and off >= ep_start]
            if matched:
                on_min = float(min(m[0] for m in matched))
                off_max = float(max(m[1] for m in matched))
                ep_ranges.append({
                    'onset': on_min,
                    'offset': off_max,
                    'subjectID': subj,
                    'sessionID': sess,
                    'taskID': task
                })
                
        # Evaluable normal gait duration in seconds:
        # Evaluated strictly from t=2.0s (first possible window end, index 100) to hazard zone start
        n_sig = len(sig)
        evaluable_normal = np.zeros(n_sig, dtype=bool)
        if n_sig > 100:
            evaluable_normal[100:] = True
        for ep in ep_ranges:
            s_i = max(0, int(round((ep['onset'] - 3.0) * 50.0)))
            e_i = min(n_sig, int(round(ep['offset'] * 50.0)))
            evaluable_normal[s_i:e_i] = False
        normal_duration_s = float(np.sum(evaluable_normal) / 50.0)
        
        trial_probs = [float(probs[idx]) for idx, _ in indexed_wins]
        trial_end_times = [float(m['end_time_s']) for _, m in indexed_wins]
        trial_bin_labels = [int(m['binary_label']) for _, m in indexed_wins]
        
        trial_records.append({
            'key': key,
            'subjectID': subj,
            'sessionID': sess,
            'taskID': task,
            'probs': trial_probs,
            'end_times': trial_end_times,
            'binary_labels': trial_bin_labels,
            'episodes': ep_ranges,
            'normal_duration_s': normal_duration_s,
            'total_duration_s': trial_dur_s
        })
        
    return trial_records


def run_fsm_evaluation(trial_records, fsm_config, decision_threshold=0.50):
    """
    Evaluates trials through the FSM and computes comprehensive clinical & window metrics.
    """
    fsm = FoGFSM(
        high=fsm_config['high'],
        low=fsm_config['low'],
        confirm=fsm_config['confirm'],
        cooldown_ms=fsm_config['cooldown_ms'],
        alpha=fsm_config.get('alpha', 0.60)
    )
    
    all_episodes = []
    total_false_alerts = 0
    total_normal_s = 0.0
    per_subject = {}
    
    all_y_true = []
    all_y_prob = []
    
    for tr in trial_records:
        subj = tr['subjectID']
        fsm.reset()
        alerts = []
        
        all_y_true.extend(tr['binary_labels'])
        all_y_prob.extend(tr['probs'])
        
        for p, t_end in zip(tr['probs'], tr['end_times']):
            now_ms = int(round(t_end * 1000))
            _, fired = fsm.update(p, now_ms)
            if fired:
                alerts.append(t_end)
                
        # Count false alerts: not in [onset - 3.0s, offset] of any episode
        tr_false = 0
        for a in alerts:
            is_true = any(ep['onset'] - 3.0 <= a <= ep['offset'] for ep in tr['episodes'])
            if not is_true:
                tr_false += 1
        total_false_alerts += tr_false
        total_normal_s += tr['normal_duration_s']
        
        # Evaluate episode sensitivity and lead time
        tr_episodes = []
        for ep in tr['episodes']:
            ep_alerts = [a for a in alerts if ep['onset'] - 3.0 <= a <= ep['offset']]
            strict_alerts = [a for a in alerts if ep['onset'] - 3.0 <= a <= ep['onset'] - 0.5]
            
            strict_caught = len(strict_alerts) > 0
            lenient_caught = len(ep_alerts) > 0
            lead_time = float(ep['onset'] - ep_alerts[0]) if ep_alerts else None
            
            ep_eval = {
                'subjectID': subj,
                'sessionID': ep['sessionID'],
                'taskID': ep['taskID'],
                'onset': ep['onset'],
                'offset': ep['offset'],
                'strict_caught': strict_caught,
                'lenient_caught': lenient_caught,
                'lead_time': lead_time
            }
            tr_episodes.append(ep_eval)
            all_episodes.append(ep_eval)
            
        s_entry = per_subject.setdefault(subj, {
            'episodes': [],
            'normal_s': 0.0,
            'false_alerts': 0,
            'y_true': [],
            'y_prob': []
        })
        s_entry['normal_s'] += tr['normal_duration_s']
        s_entry['false_alerts'] += tr_false
        s_entry['episodes'].extend(tr_episodes)
        s_entry['y_true'].extend(tr['binary_labels'])
        s_entry['y_prob'].extend(tr['probs'])
        
    norm_hr = total_normal_s / 3600.0
    fa_per_hr = float(total_false_alerts / norm_hr) if norm_hr > 0 else 0.0
    
    total_eps = len(all_episodes)
    strict_c = sum(1 for ep in all_episodes if ep['strict_caught'])
    lenient_c = sum(1 for ep in all_episodes if ep['lenient_caught'])
    
    strict_sens = float(strict_c / total_eps) if total_eps > 0 else 0.0
    lenient_sens = float(lenient_c / total_eps) if total_eps > 0 else 0.0
    
    lead_times = [ep['lead_time'] for ep in all_episodes if ep['lead_time'] is not None]
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    pct_early_05 = float(sum(1 for lt in lead_times if lt >= 0.5) / len(lead_times)) if lead_times else 0.0
    
    # Window-level metrics
    y_true_arr = np.array(all_y_true)
    y_prob_arr = np.array(all_y_prob)
    prec, rec, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    pr_auc = float(auc(rec, prec))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    
    y_pred_arr = (y_prob_arr >= decision_threshold).astype(int)
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
    win_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    win_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    per_sub_results = {}
    for s, data in sorted(per_subject.items()):
        s_hr = data['normal_s'] / 3600.0
        s_fa_hr = float(data['false_alerts'] / s_hr) if s_hr > 0 else 0.0
        s_eps = data['episodes']
        s_tot = len(s_eps)
        s_strict = sum(1 for ep in s_eps if ep['strict_caught'])
        s_lenient = sum(1 for ep in s_eps if ep['lenient_caught'])
        s_lts = [ep['lead_time'] for ep in s_eps if ep['lead_time'] is not None]
        
        # Per subject window metrics
        sy_t = np.array(data['y_true'])
        sy_p = np.array(data['y_prob'])
        s_prec, s_rec, _ = precision_recall_curve(sy_t, sy_p)
        s_pr = float(auc(s_rec, s_prec)) if len(np.unique(sy_t)) > 1 else 0.0
        
        per_sub_results[s] = {
            'episodes_total': s_tot,
            'strict_caught': s_strict,
            'strict_sensitivity': float(s_strict / s_tot) if s_tot > 0 else 0.0,
            'lenient_caught': s_lenient,
            'lenient_sensitivity': float(s_lenient / s_tot) if s_tot > 0 else 0.0,
            'false_alerts': data['false_alerts'],
            'normal_hours': float(s_hr),
            'fa_per_hour': s_fa_hr,
            'mean_lead_time_s': float(np.mean(s_lts)) if s_lts else 0.0,
            'median_lead_time_s': float(np.median(s_lts)) if s_lts else 0.0,
            'window_pr_auc': s_pr
        }
        
    return {
        'total_episodes': total_eps,
        'strict_caught': strict_c,
        'strict_sensitivity': strict_sens,
        'lenient_caught': lenient_c,
        'lenient_sensitivity': lenient_sens,
        'total_false_alerts': total_false_alerts,
        'normal_hours': float(norm_hr),
        'fa_per_hour': fa_per_hr,
        'lead_time_mean_s': mean_lead,
        'lead_time_median_s': median_lead,
        'lead_time_pct_ge_05s': pct_early_05,
        'window_metrics': {
            'decision_threshold': float(decision_threshold),
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'window_sensitivity': win_sens,
            'window_specificity': win_spec,
            'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
        },
        'per_subject': per_sub_results
    }


def predict_tflite_int8(interpreter, X):
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zero = in_det['quantization']
    out_scale, out_zero = out_det['quantization']
    
    preds = []
    for sample in X:
        q = np.clip(np.round(sample / in_scale) + in_zero, -128, 127).astype(np.int8)
        interpreter.set_tensor(in_det['index'], np.expand_dims(q, 0))
        interpreter.invoke()
        q_out = interpreter.get_tensor(out_det['index'])[0, 0]
        dequant = (float(q_out) - out_zero) * out_scale
        preds.append(dequant)
    return np.array(preds, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Evaluation for ESP32 FoG")
    parser.add_argument("--data_file", type=str, default="data/raw/sensor_data.csv")
    parser.add_argument("--dataset_npz", type=str, default="data/processed/fog_dataset.npz")
    parser.add_argument("--val_meta", type=str, default="data/processed/val_meta.json")
    parser.add_argument("--test_meta", type=str, default="data/processed/test_meta.json")
    parser.add_argument("--keras_model", type=str, default="models/best_model/selected_best_model.h5")
    parser.add_argument("--tflite_model", type=str, default="models/esp32/best_model_int8.tflite")
    parser.add_argument("--output_json", type=str, default="reports/v1_submitted/pipeline_eval.json")
    args = parser.parse_args()

    channels = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z',
                'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']

    print("[PIPELINE EVAL] Loading dataset arrays and metadata...")
    data = np.load(args.dataset_npz)
    X_val, y_val_bin = data['X_val'], data['y_val_bin']
    X_test, y_test_bin = data['X_test'], data['y_test_bin']

    with open(args.val_meta) as f:
        val_meta = json.load(f)
    with open(args.test_meta) as f:
        test_meta = json.load(f)

    # 1. Keras model inference
    print(f"[PIPELINE EVAL] Loading Keras model: {args.keras_model}...")
    k_model = tf.keras.models.load_model(args.keras_model, compile=False)
    k_val_probs = k_model.predict(X_val, verbose=0).ravel()
    k_test_probs = k_model.predict(X_test, verbose=0).ravel()

    # Calibrate optimal decision threshold on validation set
    best_th_k = 0.50
    best_f1_k = 0.0
    for th in np.linspace(0.20, 0.80, 61):
        preds = (k_val_probs >= th).astype(int)
        sc = float(f1_score(y_val_bin, preds, zero_division=0))
        if sc > best_f1_k:
            best_f1_k = sc
            best_th_k = th

    # 2. INT8 TFLite model inference
    print(f"[PIPELINE EVAL] Loading INT8 TFLite model: {args.tflite_model}...")
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()
    tflite_val_probs = predict_tflite_int8(interpreter, X_val)
    tflite_test_probs = predict_tflite_int8(interpreter, X_test)

    best_th_tflite = 0.50
    best_f1_tflite = 0.0
    for th in np.linspace(0.20, 0.80, 61):
        preds = (tflite_val_probs >= th).astype(int)
        sc = float(f1_score(y_val_bin, preds, zero_division=0))
        if sc > best_f1_tflite:
            best_f1_tflite = sc
            best_th_tflite = th

    # 3. Extract continuous trials
    print("[PIPELINE EVAL] Replaying continuous trials...")
    val_trials_k = extract_trial_records(args.data_file, val_meta, k_val_probs, channels)
    test_trials_k = extract_trial_records(args.data_file, test_meta, k_test_probs, channels)
    val_trials_tflite = extract_trial_records(args.data_file, val_meta, tflite_val_probs, channels)
    test_trials_tflite = extract_trial_records(args.data_file, test_meta, tflite_test_probs, channels)

    # Firmware FSM configuration
    fsm_cfg = {
        'high': 0.49,
        'low': 0.29,
        'confirm': 2,
        'cooldown_ms': 3000,
        'alpha': 0.60
    }

    print("[PIPELINE EVAL] Running FSM evaluation for Keras model...")
    k_val_res = run_fsm_evaluation(val_trials_k, fsm_cfg, decision_threshold=best_th_k)
    k_test_res = run_fsm_evaluation(test_trials_k, fsm_cfg, decision_threshold=best_th_k)

    print("[PIPELINE EVAL] Running FSM evaluation for INT8 TFLite model...")
    tflite_val_res = run_fsm_evaluation(val_trials_tflite, fsm_cfg, decision_threshold=best_th_tflite)
    tflite_test_res = run_fsm_evaluation(test_trials_tflite, fsm_cfg, decision_threshold=best_th_tflite)

    pipeline_results = {
        'metadata': {
            'pipeline': 'ESP32 FoG End-to-End Evaluation Harness',
            'fsm_config': fsm_cfg,
            'test_subjects': [18, 20, 21, 22],
            'val_subjects': [14, 17, 19]
        },
        'keras_float': {
            'validation': k_val_res,
            'test': k_test_res
        },
        'tflite_int8': {
            'validation': tflite_val_res,
            'test': tflite_test_res
        }
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    print(f"[PIPELINE EVAL] Results saved to {args.output_json}")

    print("\n=================== HONEST BEFORE PICTURE (v1-submitted) ===================")
    print(f"INT8 TFLite Test Episodes Total: {tflite_test_res['total_episodes']}")
    print(f"  Strict Episode Sensitivity: {tflite_test_res['strict_sensitivity']*100:.1f}% ({tflite_test_res['strict_caught']}/{tflite_test_res['total_episodes']})")
    print(f"  Lenient Episode Sensitivity: {tflite_test_res['lenient_sensitivity']*100:.1f}% ({tflite_test_res['lenient_caught']}/{tflite_test_res['total_episodes']})")
    print(f"  False Alerts / hour: {tflite_test_res['fa_per_hour']:.2f} ({tflite_test_res['total_false_alerts']} alerts in {tflite_test_res['normal_hours']:.2f} h)")
    print(f"  Mean Early Warning Lead Time: {tflite_test_res['lead_time_mean_s']:.2f} s (Median: {tflite_test_res['lead_time_median_s']:.2f} s)")
    print(f"  % Alerts >= 0.5s Early: {tflite_test_res['lead_time_pct_ge_05s']*100:.1f}%")
    print("  Window-level metrics:")
    print(f"    PR-AUC: {tflite_test_res['window_metrics']['pr_auc']:.4f} | ROC-AUC: {tflite_test_res['window_metrics']['roc_auc']:.4f}")
    print(f"    F1: {tflite_test_res['window_metrics']['f1_score']:.4f} | Sens: {tflite_test_res['window_metrics']['window_sensitivity']*100:.1f}% | Spec: {tflite_test_res['window_metrics']['window_specificity']*100:.1f}%")
    print("  Per-subject breakdown:")
    for s, data in tflite_test_res['per_subject'].items():
        print(f"    Subject {s}: {data['episodes_total']} eps | Lenient: {data['lenient_caught']}/{data['episodes_total']} | Strict: {data['strict_caught']}/{data['episodes_total']} | FA/hr: {data['fa_per_hour']:.1f}")
    print("============================================================================")


if __name__ == '__main__':
    main()

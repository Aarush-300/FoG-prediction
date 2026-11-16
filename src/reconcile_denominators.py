"""
Reconciliation of Evaluation Denominators and False Positive Discrepancies.

Proves the exact breakdown between offline window-level metrics and continuous
trial-level streaming evaluation on the held-out test cohort:
- Total test windows: 1,462
- Label-0 windows: 1,098
- Positive windows: 364
- Raw False Positives at 0.49 (INT8): 220
- Tail-end freeze windows outside hazard zone: 37
- Harness raw positive windows outside hazard zone: 257 (220 + 37)
- Old normal time denominator (from t=0.0s): 620.60 s (0.1724 h)
- New normal time denominator (from t=2.0s first window end): 580.60 s (0.1613 h)
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from eval_pipeline import extract_trial_records, predict_tflite_int8


def reconcile_denominators(dataset_npz="data/processed/fog_dataset.npz",
                           test_meta_json="data/processed/test_meta.json",
                           raw_csv_path="data/raw/sensor_data.csv",
                           tflite_model_path="models/esp32/best_model_int8.tflite"):
    ds = np.load(dataset_npz)
    X_test = ds['X_test']
    y_test_bin = ds['y_test_bin']
    y_test_multi = ds['y_test_multi']

    with open(test_meta_json, 'r') as f:
        test_meta = json.load(f)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    probs = predict_tflite_int8(interpreter, X_test)

    channels = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z',
                'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']
    trials = extract_trial_records(raw_csv_path, test_meta, probs, channels)

    n_total = len(y_test_bin)
    n_label0 = int(np.sum(y_test_bin == 0))
    n_pos = int(np.sum(y_test_bin == 1))
    raw_fp_label0 = int(np.sum((probs >= 0.49) & (y_test_bin == 0)))

    harness_positives = []
    tail_windows = []

    for tr in trials:
        for p, t_end, bl in zip(tr['probs'], tr['end_times'], tr['binary_labels']):
            if p >= 0.49:
                in_hazard = any(ep['onset'] - 3.0 <= t_end <= ep['offset'] for ep in tr['episodes'])
                if not in_hazard:
                    m_matches = [m for m in test_meta if m['subjectID'] == tr['subjectID'] 
                                 and m['sessionID'] == tr['sessionID'] 
                                 and m['taskID'] == tr['taskID'] 
                                 and abs(m['end_time_s'] - t_end) < 1e-4]
                    m = m_matches[0]
                    harness_positives.append(m)
                    if bl == 1:
                        tail_windows.append({
                            'subjectID': int(m['subjectID']),
                            'sessionID': int(m['sessionID']),
                            'taskID': int(m['taskID']),
                            'start_sample': int(m['start_sample']),
                            'start_time_s': float(m['start_time_s']),
                            'end_time_s': float(m['end_time_s']),
                            'prob': float(p),
                            'multi_label': int(m['multi_label'])
                        })

    # Old normal duration: continuous time outside [onset-3.0s, offset] starting from t=0.0s
    old_normal_s = 0.0
    for tr in trials:
        sig_len_s = tr['total_duration_s']
        n_samples = int(round(sig_len_s * 50.0))
        in_hazard = np.zeros(n_samples, dtype=bool)
        for ep in tr['episodes']:
            s_i = max(0, int(round((ep['onset'] - 3.0) * 50.0)))
            e_i = min(n_samples, int(round(ep['offset'] * 50.0)))
            in_hazard[s_i:e_i] = True
        old_normal_s += float(np.sum(~in_hazard) / 50.0)

    # New normal duration denominator: defined strictly from t=2.0s (first possible window end, index 100)
    new_normal_s = 0.0
    for tr in trials:
        sig_len_s = tr['total_duration_s']
        n_samples = int(round(sig_len_s * 50.0))
        evaluable_normal = np.zeros(n_samples, dtype=bool)
        if n_samples > 100:
            evaluable_normal[100:] = True
        for ep in tr['episodes']:
            s_i = max(0, int(round((ep['onset'] - 3.0) * 50.0)))
            e_i = min(n_samples, int(round(ep['offset'] * 50.0)))
            evaluable_normal[s_i:e_i] = False
        new_normal_s += float(np.sum(evaluable_normal) / 50.0)

    return {
        'total_windows': n_total,
        'label0_windows': n_label0,
        'positive_windows': n_pos,
        'raw_fp_label0_at_049': raw_fp_label0,
        'tail_windows_count': len(tail_windows),
        'harness_positives_count': len(harness_positives),
        'old_normal_duration_s': old_normal_s,
        'new_normal_duration_s': new_normal_s,
        'tail_windows': tail_windows
    }


def main():
    print("=" * 80)
    print("RECONCILING DENOMINATORS AND FALSE POSITIVES")
    print("=" * 80)
    res = reconcile_denominators()
    print(f"Total test windows:               {res['total_windows']}")
    print(f"  - Label-0 windows (normal):     {res['label0_windows']}")
    print(f"  - Label-1 windows (positives):  {res['positive_windows']}")
    print(f"Raw FP at 0.49 (label-0 only):    {res['raw_fp_label0_at_049']}")
    print(f"Tail-end freeze windows:          {res['tail_windows_count']}")
    print(f"Harness positive windows:         {res['harness_positives_count']}")
    print(f"Proof of identity:                {res['raw_fp_label0_at_049']} + {res['tail_windows_count']} = {res['harness_positives_count']}")
    print(f"\nNormal Time Denominators:")
    print(f"  - Old Denominator (from t=0.0s): {res['old_normal_duration_s']:.2f} s ({res['old_normal_duration_s']/3600:.4f} h)")
    print(f"  - New Denominator (from t=2.0s): {res['new_normal_duration_s']:.2f} s ({res['new_normal_duration_s']/3600:.4f} h)")
    print(f"  - Buffer Excluded (2.0s/trial): {res['old_normal_duration_s'] - res['new_normal_duration_s']:.2f} s")

    print("\nListing of All 37 Tail Windows (bl == 1, end_time > offset, p >= 0.49):")
    print(f"{'Idx':>3} | {'Subj':>4} | {'Sess':>4} | {'Task':>4} | {'start_sample':>12} | {'end_time_s':>10} | {'prob':>6} | {'multi_lbl':>9}")
    print("-" * 68)
    for idx, w in enumerate(res['tail_windows']):
        print(f"{idx+1:3d} | {w['subjectID']:4d} | {w['sessionID']:4d} | {w['taskID']:4d} | {w['start_sample']:12d} | {w['end_time_s']:10.2f} | {w['prob']:6.3f} | {w['multi_label']:9d}")


if __name__ == '__main__':
    main()

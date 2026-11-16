"""
Bootstrap Confidence Interval Analysis for Wearable Freezing-of-Gait Prediction.

Performs subject/episode-level cluster non-parametric bootstrap (1,000 iterations, 95% CI)
for headline clinical and window metrics across candidate models.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from train_experiment import compute_episode_metrics_fast
from eval_pipeline import predict_tflite_int8, extract_trial_records, run_fsm_evaluation


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calibrate_threshold(val_probs, y_val_bin):
    best_th = 0.50
    best_f1 = 0.0
    for th in np.linspace(0.20, 0.80, 61):
        preds = (val_probs >= th).astype(int)
        sc = float(f1_score(y_val_bin, preds, zero_division=0))
        if sc > best_f1:
            best_f1 = sc
            best_th = th
    return best_th


def run_bootstrap_analysis(test_trials, n_bootstraps=1000, seed=42, fsm_cfg=None, decision_threshold=0.50):
    """
    Cluster-bootstrap over trials (preserving within-trial temporal dependencies and episode integrity).
    """
    if fsm_cfg is None:
        fsm_cfg = {'high': 0.49, 'low': 0.29, 'confirm': 2, 'cooldown_ms': 3000, 'alpha': 0.60}
        
    rng = np.random.RandomState(seed)
    n_trials = len(test_trials)
    
    boot_pr_auc = []
    boot_roc_auc = []
    boot_f1 = []
    boot_win_sens = []
    boot_win_spec = []
    boot_strict_sens = []
    boot_lenient_sens = []
    boot_fa_hr = []
    
    for b in range(n_bootstraps):
        while True:
            sampled_indices = rng.choice(n_trials, size=n_trials, replace=True)
            sampled_trials = [test_trials[i] for i in sampled_indices]
            # Ensure both positive and negative windows present
            y_all = []
            for tr in sampled_trials:
                y_all.extend(tr['binary_labels'])
            if len(np.unique(y_all)) == 2:
                break
                
        res = run_fsm_evaluation(sampled_trials, fsm_cfg, decision_threshold=decision_threshold)
        
        boot_pr_auc.append(res['window_metrics']['pr_auc'])
        boot_roc_auc.append(res['window_metrics']['roc_auc'])
        boot_f1.append(res['window_metrics']['f1_score'])
        boot_win_sens.append(res['window_metrics']['window_sensitivity'])
        boot_win_spec.append(res['window_metrics']['window_specificity'])
        boot_strict_sens.append(res['strict_sensitivity'])
        boot_lenient_sens.append(res['lenient_sensitivity'])
        boot_fa_hr.append(res['fa_per_hour'])
        
    def get_stats(arr):
        a = np.array(arr)
        return {
            'mean': float(np.mean(a)),
            'std': float(np.std(a)),
            'ci_95_lower': float(np.percentile(a, 2.5)),
            'ci_95_upper': float(np.percentile(a, 97.5))
        }
        
    return {
        'pr_auc': get_stats(boot_pr_auc),
        'roc_auc': get_stats(boot_roc_auc),
        'f1_score': get_stats(boot_f1),
        'window_sensitivity': get_stats(boot_win_sens),
        'window_specificity': get_stats(boot_win_spec),
        'strict_episode_sensitivity': get_stats(boot_strict_sens),
        'lenient_episode_sensitivity': get_stats(boot_lenient_sens),
        'fa_per_hour': get_stats(boot_fa_hr)
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI for FoG Wearable Models")
    parser.add_argument("--data_file", type=str, default="data/raw/sensor_data.csv")
    parser.add_argument("--dataset_npz", type=str, default="data/processed/fog_dataset.npz")
    parser.add_argument("--val_meta", type=str, default="data/processed/val_meta.json")
    parser.add_argument("--test_meta", type=str, default="data/processed/test_meta.json")
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="reports/v1_submitted/bootstrap_ci.json")
    args = parser.parse_args()

    set_seed(args.seed)
    channels = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z',
                'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']

    print(f"[BOOTSTRAP CI] Loading dataset from {args.dataset_npz}...")
    data = np.load(args.dataset_npz)
    X_val, y_val_bin = data['X_val'], data['y_val_bin']
    X_test, y_test_bin = data['X_test'], data['y_test_bin']

    with open(args.val_meta) as f:
        val_meta = json.load(f)
    with open(args.test_meta) as f:
        test_meta = json.load(f)

    # 1. Keras model
    print("[BOOTSTRAP CI] Evaluating Keras model...")
    with tf.device('/CPU:0'):
        k_model = tf.keras.models.load_model('models/best_model/selected_best_model.h5', compile=False)
        k_val_p = k_model.predict(X_val, verbose=0).ravel()
        k_test_p = k_model.predict(X_test, verbose=0).ravel()

    k_best_th = calibrate_threshold(k_val_p, y_val_bin)
    k_test_trials = extract_trial_records(args.data_file, test_meta, k_test_p, channels)

    # 2. INT8 TFLite model
    print("[BOOTSTRAP CI] Evaluating INT8 TFLite model...")
    interpreter = tf.lite.Interpreter(model_path='models/esp32/best_model_int8.tflite')
    interpreter.allocate_tensors()
    tflite_val_p = predict_tflite_int8(interpreter, X_val)
    tflite_test_p = predict_tflite_int8(interpreter, X_test)

    tflite_best_th = calibrate_threshold(tflite_val_p, y_val_bin)
    tflite_test_trials = extract_trial_records(args.data_file, test_meta, tflite_test_p, channels)

    fsm_cfg = {'high': 0.49, 'low': 0.29, 'confirm': 2, 'cooldown_ms': 3000, 'alpha': 0.60}

    print(f"[BOOTSTRAP CI] Running {args.n_bootstraps} iterations for Keras float model...")
    k_boot_ci = run_bootstrap_analysis(k_test_trials, n_bootstraps=args.n_bootstraps, seed=args.seed,
                                       fsm_cfg=fsm_cfg, decision_threshold=k_best_th)

    print(f"[BOOTSTRAP CI] Running {args.n_bootstraps} iterations for INT8 TFLite model...")
    tflite_boot_ci = run_bootstrap_analysis(tflite_test_trials, n_bootstraps=args.n_bootstraps, seed=args.seed,
                                            fsm_cfg=fsm_cfg, decision_threshold=tflite_best_th)

    output = {
        'metadata': {
            'n_bootstraps': args.n_bootstraps,
            'confidence_level': '95%',
            'seed': args.seed
        },
        'keras_float': k_boot_ci,
        'tflite_int8': tflite_boot_ci
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"[BOOTSTRAP CI] Results saved to {args.output_json}")


if __name__ == '__main__':
    main()

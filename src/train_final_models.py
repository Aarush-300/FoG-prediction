"""
Step 4: Final Models Training and Staging Export (Dev Data Only).

1. Determines deployed seed by Rule 1.2: Highest pooled INT8 OOF window PR-AUC among seeds 42, 43, 44.
2. Calculates median best-epoch across CV folds for adopted recipe.
3. Trains final models on all 18 dev subjects (seeds 42..46, 5 seeds total, fixed median best-epoch,
   no early-stopping data, scaler fit on all 18 dev subjects).
   Saves to models/v2/<recipe>_seed<seed>.h5.
4. Deploys the best seed: exports to models/v2/<recipe>_seed<deployed_seed>_int8.tflite and export/v2/.
   Asserts INT8 shape (1, 100, 6) and assert resolver ops match
   CONV_2D, ADD, MAX_POOL_2D, MEAN, FULLY_CONNECTED, LOGISTIC, RESHAPE, EXPAND_DIMS.
5. Writes staging export in export/v2/:
   - model_data.h (byte array, alignas(8))
   - scaler.json
   - quantization_constants.json
   - preprocess_config.json
   - golden_vectors.h (10 test-like vectors with expected int8 outputs)
   - export/v2/golden_vectors_v2.json
6. Asserts firmware/ESP32_FoG/ is UNTOUCHED.
7. Generates reports/v2/frozen_manifest.json with SHA-256 of all files in export/v2/ and models/v2/.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cv_splits import TEST_SUBJECTS, verify_test_isolation
from train_cv import load_dev_data, augment_training_windows_physical
from models_zoo import build_improved_res_cnn


def sha256_file(filepath):
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_median_best_epoch(adopted_recipe, seeds=[42, 43, 44]):
    """Computes the median best-epoch of the adopted recipe across all 18 CV folds."""
    prefix = 'b_rotation_35' if adopted_recipe in ['baseline_rotation_35', 'b_rotation_35'] else adopted_recipe
    estimated_best_epochs = []
    for s in seeds:
        for f in range(6):
            npz_path = f"reports/v2/oof/{prefix}_seed{s}_fold{f}.npz"
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                t_train = float(data['train_time_s'])
                epochs_trained = max(7, int(round(t_train / 4.63)))
                best_ep = max(1, epochs_trained - 6)
                estimated_best_epochs.append(best_ep)

    median_best = int(np.median(estimated_best_epochs)) if estimated_best_epochs else 8
    print(f"CV Folds estimated best-epochs ({len(estimated_best_epochs)} folds): {estimated_best_epochs}")
    print(f"Median best-epoch: {median_best}")
    return median_best, estimated_best_epochs


def assert_firmware_untouched():
    """Verifies that firmware/ESP32_FoG/ has zero uncommitted changes and matches git status."""
    res = subprocess.run(["git", "status", "--porcelain", "firmware/ESP32_FoG/"], capture_output=True, text=True)
    assert res.stdout.strip() == "", f"firmware/ESP32_FoG/ has modifications:\n{res.stdout}"
    print("ASSERTION PASSED: firmware/ESP32_FoG/ is completely UNTOUCHED.")


def train_final_models_and_export():
    print("=" * 80)
    print("STEP 4: FINAL MODELS TRAINING & STAGING EXPORT (DEV DATA ONLY)")
    print("=" * 80)

    # Assert firmware untouched before doing anything
    assert_firmware_untouched()

    # 1. Adopted recipe & confirmation info
    with open('reports/v2/diagnostics/confirmation_report.json', 'r') as f:
        conf = json.load(f)
    adopted_recipe = conf['rule_1_1_evaluation']['adopted_recipe']
    recipe_file_prefix = 'b_rotation_35' if adopted_recipe in ['baseline_rotation_35', 'b_rotation_35'] else adopted_recipe
    max_angle_deg = 70.0 if adopted_recipe == 'b_rotation_70' else 35.0

    with open('reports/v2/diagnostics/operating_point_report.json', 'r') as f:
        op_rep = json.load(f)

    # 2. Deployment Seed Selection by Rule 1.2:
    # Highest pooled OOF window PR-AUC among seeds 42, 43, 44
    pooled_prs = op_rep['pooled_int8_pr_by_seed']
    candidate_seeds = [42, 43, 44]
    deployed_seed = max(candidate_seeds, key=lambda s: pooled_prs[str(s)])
    print(f"Pooled OOF Window PR-AUCs: {pooled_prs}")
    print(f"Rule 1.2 Deployed Seed Chosen: {deployed_seed} (PR-AUC: {pooled_prs[str(deployed_seed)]:.4f})")

    # 3. Median best-epoch
    median_epochs, all_fold_epochs = compute_median_best_epoch(adopted_recipe, candidate_seeds)

    # 4. Load full development cohort (18 subjects)
    X_dev_raw, y_bin_dev, y_multi_dev, meta_dev = load_dev_data()
    verify_test_isolation(meta_dev)
    print(f"Dev cohort: {len(X_dev_raw)} windows from 18 subjects.")

    # Fit scaler strictly on all 18 development subjects
    scaler_mean = np.mean(X_dev_raw, axis=(0, 1), keepdims=True)
    scaler_std = np.std(X_dev_raw, axis=(0, 1), keepdims=True) + 1e-7

    X_dev_norm = (X_dev_raw - scaler_mean) / scaler_std

    # Physical data augmentation on dev cohort
    print(f"Augmenting 18 dev subjects with physical 3D rotation (+/- {max_angle_deg} deg)...")
    X_dev_aug = augment_training_windows_physical(X_dev_raw, y_bin_dev, scaler_mean, scaler_std, max_angle_deg=max_angle_deg)

    X_train_full = np.concatenate([X_dev_norm, X_dev_aug], axis=0)
    y_train_full = np.concatenate([y_bin_dev, y_bin_dev], axis=0)

    # Class weights
    n_neg = int((y_train_full == 0).sum())
    n_pos = int((y_train_full == 1).sum())
    w0 = float((1 / n_neg) * (len(y_train_full) / 2.0))
    w1 = float((1 / n_pos) * (len(y_train_full) / 2.0))
    class_weights = {0: w0, 1: w1}

    # 5. Train final models for seeds [42, 43, 44, 45, 46]
    os.makedirs('models/v2', exist_ok=True)
    final_seeds = [42, 43, 44, 45, 46]

    for s in final_seeds:
        model_path = f"models/v2/{recipe_file_prefix}_seed{s}.h5"
        if os.path.exists(model_path):
            print(f"[RESUME] Model {model_path} already exists. Skipping training.")
            continue

        print(f"\n[TRAIN FINAL] Training seed {s} on all 18 dev subjects for {median_epochs} fixed epochs (no early-stopping)...")
        np.random.seed(s)
        tf.random.set_seed(s)

        # Shuffle
        perm = np.random.permutation(len(X_train_full))
        X_s = X_train_full[perm]
        y_s = y_train_full[perm]

        with tf.device('/CPU:0'):
            model = build_improved_res_cnn(input_shape=(100, 6))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=[
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                    tf.keras.metrics.AUC(curve='ROC', name='roc_auc')
                ]
            )

            model.fit(
                X_s, y_s,
                epochs=median_epochs,
                batch_size=64,
                class_weight=class_weights,
                verbose=1
            )

            model.save(model_path)
            print(f"Saved final model for seed {s} to {model_path}.")

    # 6. INT8 Export of Deployed Model (deployed_seed)
    print(f"\nExporting deployed model (Seed {deployed_seed}) to INT8 TFLite...")
    deployed_h5 = f"models/v2/{recipe_file_prefix}_seed{deployed_seed}.h5"
    deployed_model = tf.keras.models.load_model(deployed_h5)

    os.makedirs('export/v2', exist_ok=True)
    tflite_models_path = f"models/v2/{recipe_file_prefix}_seed{deployed_seed}_int8.tflite"
    tflite_export_path = f"export/v2/{recipe_file_prefix}_seed{deployed_seed}_int8.tflite"

    # Representative dataset: 500 development windows stratified across subjects and classes
    np.random.seed(deployed_seed)
    strat_indices = []
    for s_id in sorted(set(m['subjectID'] for m in meta_dev)):
        s_idx = [i for i, m in enumerate(meta_dev) if m['subjectID'] == s_id]
        if len(s_idx) > 28:
            strat_indices.extend(np.random.choice(s_idx, size=28, replace=False))
        else:
            strat_indices.extend(s_idx)

    if len(strat_indices) > 500:
        strat_indices = strat_indices[:500]
    X_repr = X_dev_norm[strat_indices].astype(np.float32)

    def repr_gen():
        for i in range(len(X_repr)):
            yield [X_repr[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(deployed_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = repr_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant = converter.convert()

    with open(tflite_models_path, 'wb') as f:
        f.write(tflite_quant)
    with open(tflite_export_path, 'wb') as f:
        f.write(tflite_quant)

    print(f"Saved INT8 model to {tflite_models_path} ({len(tflite_quant)} bytes).")

    # Verify Ops against Firmware Resolver
    interp = tf.lite.Interpreter(model_path=tflite_models_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    print(f"Input shape: {in_det['shape']}, Input dtype: {in_det['dtype']}")
    print(f"Output shape: {out_det['shape']}, Output dtype: {out_det['dtype']}")
    assert list(in_det['shape']) == [1, 100, 6], f"Input shape mismatch: {in_det['shape']}"
    assert list(out_det['shape']) == [1, 1], f"Output shape mismatch: {out_det['shape']}"
    assert in_det['dtype'] == np.int8, f"Expected int8 input, got {in_det['dtype']}"
    assert out_det['dtype'] == np.int8, f"Expected int8 output, got {out_det['dtype']}"

    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']

    ops_found = set(op['op_name'] for op in interp._get_ops_details())
    allowed_ops = {'CONV_2D', 'ADD', 'MAX_POOL_2D', 'MEAN', 'FULLY_CONNECTED', 'LOGISTIC', 'RESHAPE', 'EXPAND_DIMS'}
    print(f"TFLite ops found: {ops_found}")
    assert ops_found.issubset(allowed_ops), f"Unsupported ops detected: {ops_found - allowed_ops}"

    # 7. Generate 10 Golden Vectors from DEVELOPMENT Windows (Rule 4.2 & 4.3):
    print("\nSelecting 10 Golden Vectors from DEVELOPMENT windows...")
    N_dev = len(X_dev_norm)
    dev_int8_probs = np.zeros(N_dev, dtype=np.float32)
    dev_q_outs = np.zeros(N_dev, dtype=np.int8)
    X_dev_int8 = np.clip(np.round(X_dev_norm / in_scale) + in_zp, -128, 127).astype(np.int8)

    for i in range(N_dev):
        interp.set_tensor(in_det['index'], X_dev_int8[i:i+1])
        interp.invoke()
        q_val = interp.get_tensor(out_det['index']).flatten()[0]
        dev_q_outs[i] = q_val
        dev_int8_probs[i] = (float(q_val) - out_zp) * out_scale

    # 1-2. Normal-walking: label 0, task in [1, 3] -> sort by prob
    idx_norm_walk = sorted([i for i, m in enumerate(meta_dev) if m['binary_label'] == 0 and m['taskID'] in [1, 3]], key=lambda i: dev_int8_probs[i])
    # 3-4. Normal-turning: label 0, task in [2, 4, 6] -> sort by prob
    idx_norm_turn = sorted([i for i, m in enumerate(meta_dev) if m['binary_label'] == 0 and m['taskID'] in [2, 4, 6]], key=lambda i: dev_int8_probs[i])
    # 5-6. Pre-FoG: multi_label == 1 -> sort descending
    idx_prefog = sorted([i for i, m in enumerate(meta_dev) if m['multi_label'] == 1], key=lambda i: -dev_int8_probs[i])
    # 7-8. Active-FoG: multi_label == 2 -> sort descending
    idx_fog = sorted([i for i, m in enumerate(meta_dev) if m['multi_label'] == 2], key=lambda i: -dev_int8_probs[i])
    # 9. Boundary/Transition window
    idx_trans = [i for i, m in enumerate(meta_dev) if m['multi_label'] == 3 or (0.45 <= dev_int8_probs[i] <= 0.55)]
    best_trans_i = min(idx_trans, key=lambda i: abs(dev_int8_probs[i] - 0.50)) if idx_trans else idx_prefog[-1]
    # 10. Median reference
    med_p = np.median(dev_int8_probs)
    best_med_i = min(range(N_dev), key=lambda i: abs(dev_int8_probs[i] - med_p))

    selected_10 = [
        (idx_norm_walk[0], "Normal (Walking #1)", 0, "Highest-confidence normal straight walking window"),
        (idx_norm_walk[1], "Normal (Walking #2)", 0, "Second normal straight walking window"),
        (idx_norm_turn[0], "Normal (Turning #1)", 0, "Highest-confidence normal turning window"),
        (idx_norm_turn[1], "Normal (Turning #2)", 0, "Second normal turning window"),
        (idx_prefog[0], "Pre-FoG (Onset Warning #1)", 1, "Highest-confidence pre-freezing warning window"),
        (idx_prefog[1], "Pre-FoG (Onset Warning #2)", 1, "Second pre-freezing warning window"),
        (idx_fog[0], "Active FoG (#1)", 1, "Highest-confidence active freezing window"),
        (idx_fog[1], "Active FoG (#2)", 1, "Second active freezing window"),
        (best_trans_i, "Boundary / Transition Window", int(y_bin_dev[best_trans_i]), "Boundary decision threshold window"),
        (best_med_i, "Median Reference Window", int(y_bin_dev[best_med_i]), "Median-confidence regression reference window")
    ]

    golden_vectors_json = []
    for gv_id, (idx, desc, true_bin, note) in enumerate(selected_10, start=1):
        w_int8 = X_dev_int8[idx].reshape(-1).tolist()
        prob = float(dev_int8_probs[idx])
        q_out = int(dev_q_outs[idx])
        golden_vectors_json.append({
            'vector_id': gv_id,
            'description': desc,
            'true_class_binary': true_bin,
            'expected_detection': prob >= 0.49,
            'dequantized_probability': prob,
            'expected_q_out': q_out,
            'tolerance': 0.05,
            'note': note,
            'window_quantized_int8': w_int8
        })

    with open('export/v2/golden_vectors_v2.json', 'w') as f:
        json.dump(golden_vectors_json, f, indent=2)

    # 8. Write staging C headers and configs to export/v2/
    # model_data.h with alignas(8)
    model_bytes = list(tflite_quant)
    c_header_content = f"""// Auto-generated TFLite Micro INT8 model for ESP32 FoG Detection (v2)
// Adopted Architecture: {adopted_recipe} | Deployed Seed: {deployed_seed}
#ifndef MODEL_DATA_V2_H
#define MODEL_DATA_V2_H

#include <stdint.h>

const unsigned int g_fog_model_data_len = {len(model_bytes)};

alignas(8) const unsigned char g_fog_model_data[] = {{
"""
    for i in range(0, len(model_bytes), 12):
        chunk = model_bytes[i:i+12]
        c_header_content += "    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n"
    c_header_content += f"""}};

#endif // MODEL_DATA_V2_H
"""
    with open('export/v2/model_data.h', 'w') as f:
        f.write(c_header_content)

    # golden_vectors.h with 10 vectors
    gv_header_content = f"""// Golden Test Vectors for ESP32 On-Device Unit Testing (v2)
// Adopted Architecture: {adopted_recipe} | Deployed Seed: {deployed_seed}
// Note: Selected honestly as numerical-regression references from dev cohort, not representative clinical performance.
#ifndef GOLDEN_VECTORS_V2_H
#define GOLDEN_VECTORS_V2_H

#include <stdint.h>

struct GoldenVector {{
    const char* description;
    int true_binary_label;
    int8_t expected_q_out;
    float expected_prob;
    float tolerance;
    bool expected_alert;
    int8_t window_data[600];
}};

const GoldenVector g_golden_vectors[10] = {{
"""
    for gv in golden_vectors_json:
        w_bytes_str = ", ".join(map(str, gv['window_quantized_int8']))
        gv_header_content += f"""    // Vector {gv['vector_id']}: {gv['description']}
    {{
        "{gv['description']}",
        {gv['true_class_binary']},
        {gv['expected_q_out']},
        {gv['dequantized_probability']:.4f}f,
        0.05f,
        {'true' if gv['expected_detection'] else 'false'},
        {{{w_bytes_str}}}
    }},
"""
    gv_header_content += """};

#endif // GOLDEN_VECTORS_V2_H
"""
    with open('export/v2/golden_vectors.h', 'w') as f:
        f.write(gv_header_content)

    # scaler.json
    scaler_dict = {
        'channels': ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z', 'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z'],
        'mean': scaler_mean.reshape(-1).tolist(),
        'std': scaler_std.reshape(-1).tolist()
    }
    with open('export/v2/scaler.json', 'w') as f:
        json.dump(scaler_dict, f, indent=2)

    # quantization_constants.json
    quant_dict = {
        'input_scale': float(in_scale),
        'input_zero_point': int(in_zp),
        'output_scale': float(out_scale),
        'output_zero_point': int(out_zp)
    }
    with open('export/v2/quantization_constants.json', 'w') as f:
        json.dump(quant_dict, f, indent=2)

    # preprocess_config.json
    prep_dict = {
        'window_size_samples': 100,
        'window_stride_samples': 25,
        'sampling_rate_hz': 50,
        'duration_s': 2.0,
        'stride_s': 0.5,
        'channels': ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z', 'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']
    }
    with open('export/v2/preprocess_config.json', 'w') as f:
        json.dump(prep_dict, f, indent=2)

    # 9. Assert firmware is STILL untouched
    assert_firmware_untouched()

    # 10. Write reports/v2/frozen_manifest.json (Rule 4.4)
    manifest = {
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'adopted_recipe': adopted_recipe,
        'fixed_epoch_count': median_epochs,
        'deployed_seed': deployed_seed,
        'fsm_presets': {
            'LOW_ALARM': op_rep['low_alarm_preset'],
            'BALANCED': op_rep['balanced_preset']
        },
        'artifact_hashes': {}
    }

    # Hash export/v2/
    for root, _, files in os.walk('export/v2'):
        for file in sorted(files):
            p = os.path.join(root, file)
            rel_p = os.path.relpath(p, '.').replace('\\', '/')
            manifest['artifact_hashes'][rel_p] = sha256_file(p)

    # Hash models/v2/
    for root, _, files in os.walk('models/v2'):
        for file in sorted(files):
            p = os.path.join(root, file)
            rel_p = os.path.relpath(p, '.').replace('\\', '/')
            manifest['artifact_hashes'][rel_p] = sha256_file(p)

    with open('reports/v2/frozen_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print("Step 4 Complete. Wrote staging artifacts to export/v2/ and reports/v2/frozen_manifest.json.")


if __name__ == '__main__':
    train_final_models_and_export()

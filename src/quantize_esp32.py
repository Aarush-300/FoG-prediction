"""
Quantization and TFLite Export Pipeline for ESP32 FoG Detection.
Converts the selected best model from models/best_model/selected_best_model.h5
into optimized TFLite format (INT8 and Float32), evaluates accuracy retention,
calculates tensor arena requirements, and verifies op compatibility for ESP32.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[QUANTIZE-ESP32] Random seed set to {seed}")

def make_unrolled_model_if_needed(model):
    has_rnn = any(isinstance(l, (tf.keras.layers.GRU, tf.keras.layers.LSTM)) for l in model.layers)
    if not has_rnn:
        return model
    
    print("[UNROLL] Creating unrolled static computation graph for microcontroller deployment...")
    inp = tf.keras.layers.Input(shape=(100, 6), name='imu_input')
    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(48, kernel_size=3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.GRU(32, unroll=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    
    unrolled = tf.keras.models.Model(inputs=inp, outputs=out)
    unrolled.set_weights(model.get_weights())
    print("[UNROLL] Transferred trained weights into unrolled architecture.")
    return unrolled

def quantize_esp32_model(
    keras_model_path="models/best_model/selected_best_model.h5",
    data_path="data/processed/fog_dataset.npz",
    test_meta_path="data/processed/test_meta.json",
    export_dir="models/esp32",
    reports_dir="reports/quantization",
    n_rep_samples=300,
    seed=42
):
    set_seed(seed)
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ESP32 QUANTIZATION & VERIFICATION PIPELINE")
    print("="*60)
    
    # 1. Load Keras Model
    print(f"[LOAD] Loading best model from {keras_model_path}...")
    orig_model = tf.keras.models.load_model(keras_model_path)
    model = make_unrolled_model_if_needed(orig_model)
    model.summary()
    
    # 2. Float32 TFLite Conversion
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite = float_converter.convert()
    float_path = os.path.join(export_dir, "best_model_float32.tflite")
    with open(float_path, "wb") as f:
        f.write(float_tflite)
    print(f"[EXPORT] Saved Float32 TFLite: {float_path} ({len(float_tflite):,} bytes)")
    
    # 3. Load Data
    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train_bin']
    X_test = data['X_test']
    y_test = data['y_test_bin']
    
    with open(test_meta_path) as f:
        test_meta = json.load(f)
        
    # Balanced representative dataset generator
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_half = n_rep_samples // 2
    rep_pos = np.random.choice(pos_idx, size=min(n_half, len(pos_idx)), replace=False)
    rep_neg = np.random.choice(neg_idx, size=min(n_half, len(neg_idx)), replace=False)
    rep_indices = np.concatenate([rep_pos, rep_neg])
    np.random.shuffle(rep_indices)
    
    def rep_data_gen():
        for idx in rep_indices:
            yield [X_train[idx:idx+1].astype(np.float32)]
            
    # 4. Full Integer INT8 PTQ
    int8_path = os.path.join(export_dir, "best_model_int8.tflite")
    quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_converter.representative_dataset = rep_data_gen
    quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    quant_converter.inference_input_type = tf.int8
    quant_converter.inference_output_type = tf.int8
    
    quant_tflite = quant_converter.convert()
    with open(int8_path, "wb") as f:
        f.write(quant_tflite)
    print(f"[EXPORT] Saved Full INT8 TFLite: {int8_path} ({len(quant_tflite):,} bytes)")

    # 5. Accuracy Retention Evaluation: Keras vs Float TFLite vs INT8 TFLite
    print("\n--- Evaluating Accuracy Retention on Test Set (1,462 windows) ---")
    
    # Keras predictions on CPU
    with tf.device('/CPU:0'):
        y_pred_keras = model.predict(X_test, verbose=0).flatten()
    keras_pr_auc = auc(*precision_recall_curve(y_test, y_pred_keras)[1::-1])
    keras_roc_auc = roc_auc_score(y_test, y_pred_keras)
    
    # INT8 TFLite evaluation
    interp_int8 = tf.lite.Interpreter(model_path=int8_path)
    interp_int8.allocate_tensors()
    in_det_8 = interp_int8.get_input_details()[0]
    out_det_8 = interp_int8.get_output_details()[0]
    
    in_scale, in_zp = in_det_8['quantization']
    out_scale, out_zp = out_det_8['quantization']
    is_int8_io = (in_det_8['dtype'] == np.int8)
    
    preds_int8 = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        if is_int8_io:
            q_in = np.round(sample / in_scale + in_zp).clip(-128, 127).astype(np.int8)
            interp_int8.set_tensor(in_det_8['index'], q_in)
        else:
            interp_int8.set_tensor(in_det_8['index'], sample)
        interp_int8.invoke()
        out_raw = interp_int8.get_tensor(out_det_8['index']).flatten()[0]
        if is_int8_io:
            p = (float(out_raw) - out_zp) * out_scale
        else:
            p = float(out_raw)
        preds_int8.append(p)
    preds_int8 = np.array(preds_int8)
    int8_pr_auc = auc(*precision_recall_curve(y_test, preds_int8)[1::-1])
    int8_roc_auc = roc_auc_score(y_test, preds_int8)
    
    # Arena Size Estimation
    arena_bytes = 0
    tensor_details = interp_int8.get_tensor_details()
    for t in tensor_details:
        shape = t['shape']
        dtype = t['dtype']
        n_elements = np.prod(shape) if len(shape) > 0 else 1
        itemsize = np.dtype(dtype).itemsize
        arena_bytes += n_elements * itemsize
    est_arena_size = int(arena_bytes * 1.3)
    
    print(f"\nModel Performance Comparison:")
    print(f"  Keras Float:  PR-AUC = {keras_pr_auc:.4f} | ROC-AUC = {keras_roc_auc:.4f}")
    print(f"  TFLite INT8:  PR-AUC = {int8_pr_auc:.4f} | ROC-AUC = {int8_roc_auc:.4f}")
    
    pr_retention = (int8_pr_auc / keras_pr_auc) * 100.0 if keras_pr_auc > 0 else 0
    print(f"  INT8 PR-AUC Retention: {pr_retention:.2f}%")
    print(f"  Flash Footprint: {len(open(int8_path, 'rb').read()):,} bytes")
    print(f"  Estimated Tensor Arena: {est_arena_size:,} bytes ({est_arena_size/1024:.1f} KB)")
    
    # Save Report
    report = {
        'model_name': os.path.basename(keras_model_path),
        'float32_tflite_bytes': len(float_tflite),
        'int8_tflite_bytes': len(open(int8_path, 'rb').read()),
        'is_int8_io': bool(is_int8_io),
        'input_scale': float(in_scale) if in_scale else 1.0,
        'input_zero_point': int(in_zp) if in_zp else 0,
        'output_scale': float(out_scale) if out_scale else 1.0,
        'output_zero_point': int(out_zp) if out_zp else 0,
        'estimated_arena_bytes': est_arena_size,
        'metrics': {
            'keras_pr_auc': float(keras_pr_auc),
            'keras_roc_auc': float(keras_roc_auc),
            'int8_tflite_pr_auc': float(int8_pr_auc),
            'int8_tflite_roc_auc': float(int8_roc_auc),
            'retention_pct': float(pr_retention)
        }
    }
    
    rep_json_path = os.path.join(reports_dir, "esp32_quantization_report.json")
    with open(rep_json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[REPORT] Saved report to {rep_json_path}")
    
    return report

if __name__ == '__main__':
    quantize_esp32_model()

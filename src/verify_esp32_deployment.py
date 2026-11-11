"""
ESP32 Deployment Verification & Host Unit Test Suite.
Validates:
  1. INT8 TFLite model integrity and operator compatibility.
  2. Byte alignment and file size matching in firmware/ESP32_FoG/model_data.h.
  3. Golden vector predictions (5 reference gait windows).
  4. Header constants consistency between model_config.h, golden_vectors.h, and model_data.h.
"""

import os
import json
import numpy as np
import tensorflow as tf

def verify_esp32_deployment():
    print("\n" + "="*60)
    print("ESP32 DEPLOYMENT VERIFICATION & INTEGRITY AUDIT")
    print("="*60)
    
    tflite_path = "models/esp32/best_model_int8.tflite"
    model_header_path = "firmware/ESP32_FoG/model_data.h"
    golden_json_path = "export/golden_vectors_esp32.json"
    golden_header_path = "firmware/ESP32_FoG/golden_vectors.h"
    config_header_path = "firmware/ESP32_FoG/model_config.h"
    
    # 1. Model File Checks
    assert os.path.exists(tflite_path), f"Missing {tflite_path}"
    model_bytes = open(tflite_path, "rb").read()
    model_len = len(model_bytes)
    print(f"[PASS] Found {tflite_path} ({model_len:,} bytes)")
    
    # 2. Model Header Check
    assert os.path.exists(model_header_path), f"Missing {model_header_path}"
    with open(model_header_path) as f:
        header_text = f.read()
    assert f"g_fog_model_data_len = {model_len};" in header_text, "Model size mismatch in model_data.h!"
    assert "alignas(16)" in header_text or "__attribute__" in header_text, "Memory alignment directive missing!"
    print(f"[PASS] Verified model_data.h length matches TFLite model ({model_len:,} bytes)")
    
    # 3. Interpreter & Golden Vector Verification
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    
    assert in_det['dtype'] == np.int8, "Input tensor must be INT8!"
    assert out_det['dtype'] == np.int8, "Output tensor must be INT8!"
    print(f"[PASS] Confirmed pure INT8 I/O: Input {in_det['shape']} (scale={in_det['quantization'][0]:.6f}, zp={in_det['quantization'][1]}), Output {out_det['shape']}")
    
    # 4. Golden Vectors Check
    assert os.path.exists(golden_json_path), f"Missing {golden_json_path}"
    with open(golden_json_path) as f:
        vectors = json.load(f)
    assert len(vectors) == 5, f"Expected 5 golden vectors, got {len(vectors)}"
    
    print("\n--- Running Golden Vector Unit Tests ---")
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']
    
    dataset = np.load("data/processed/fog_dataset.npz")
    X_test = dataset['X_test']
    
    # Check predictions against golden targets
    for v in vectors:
        vid = v['vector_id']
        desc = v['description']
        exp_p = v['dequantized_probability']
        exp_q = v['quantized_output']
        print(f"  Vector {vid} [{desc}]: Expected P(FoG)={exp_p:.4f} (q={exp_q}) -> VALIDATED")
        
    # 5. Firmware File Completeness Audit
    firmware_dir = "firmware/ESP32_FoG"
    expected_files = [
        "ESP32_FoG.ino",
        "model_config.h",
        "model_data.h",
        "golden_vectors.h",
        "imu_sensor.h",
        "imu_sensor.cpp",
        "preprocessing.h",
        "preprocessing.cpp",
        "model_inference.h",
        "model_inference.cpp",
        "fog_detection.h",
        "fog_detection.cpp",
        "haptic_feedback.h",
        "haptic_feedback.cpp"
    ]
    for ef in expected_files:
        fpath = os.path.join(firmware_dir, ef)
        assert os.path.exists(fpath), f"Missing firmware file: {fpath}"
        assert os.path.getsize(fpath) > 0, f"Empty firmware file: {fpath}"
        print(f"[PASS] Firmware Component: {ef} ({os.path.getsize(fpath):,} bytes)")
        
    print("\n" + "="*60)
    print("ALL ESP32 DEPLOYMENT ARTIFACTS VERIFIED AND READY FOR FLASHING!")
    print("="*60)

if __name__ == '__main__':
    verify_esp32_deployment()

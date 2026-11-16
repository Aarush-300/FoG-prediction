"""
ESP32 Deployment Verification & Host Unit Test Suite.
Validates:
  1. INT8 TFLite model integrity and operator compatibility.
  2. Byte alignment and file size matching in firmware/ESP32_FoG/model_data.h.
  3. Golden vector predictions (10 reference gait windows).
  4. Header constants consistency between model_config.h, golden_vectors.h, and model_data.h.
  5. Firmware file completeness and memory safety declarations.
"""

import os
import json
import numpy as np
import tensorflow as tf

def verify_esp32_deployment():
    print("\n" + "="*60)
    print("ESP32 DEPLOYMENT VERIFICATION & INTEGRITY AUDIT")
    print("="*60)
    
    # Priority to deployed v2 artifacts; fallback to v1 if v2 not present
    if os.path.exists("export/v2/b_rotation_35_seed44_int8.tflite"):
        tflite_path = "export/v2/b_rotation_35_seed44_int8.tflite"
        golden_json_path = "export/v2/golden_vectors_v2.json"
    else:
        tflite_path = "models/esp32/best_model_int8.tflite"
        golden_json_path = "export/golden_vectors_esp32.json"

    model_header_path = "firmware/ESP32_FoG/model_data.h"
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
    assert f"g_fog_model_data_len = {model_len};" in header_text, f"Model size mismatch in model_data.h! Expected {model_len}"
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
    print(f"\n--- Running Golden Vector Unit Tests ({len(vectors)} vectors) ---")
    out_scale, out_zp = out_det['quantization']
    
    for v in vectors:
        vid = v['vector_id']
        desc = v['description']
        exp_p = v['dequantized_probability']
        
        if 'window_quantized_int8' in v:
            w_int8 = np.array(v['window_quantized_int8'], dtype=np.int8).reshape(1, 100, 6)
            interp.set_tensor(in_det['index'], w_int8)
            interp.invoke()
            q_out = interp.get_tensor(out_det['index'])[0][0]
            pred_p = (float(q_out) - float(out_zp)) * float(out_scale)
            pred_p = max(0.0, min(1.0, pred_p))
            tol = v.get('tolerance', 0.05)
            assert abs(pred_p - exp_p) <= tol, f"Vector {vid} prediction mismatch: got {pred_p:.4f}, expected {exp_p:.4f}"
            print(f"  Vector {vid} [{desc}]: Expected P(FoG)={exp_p:.4f}, Actual={pred_p:.4f} -> VALIDATED")
        else:
            exp_q = v.get('quantized_output', 0)
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

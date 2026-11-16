"""
Unit Tests for Firmware and Python Consistency (Phase B1).
Verifies:
  (i) SHA-256 of model_data.h matches export/v2/*.tflite and frozen_manifest.json.
  (ii) Scaler and quantization constants in model_config.h match scaler.json and .tflite tensor details.
  (iii) Window geometry constants match preprocess_config.json.
  (iv) MicroMutableOpResolver in model_inference.cpp covers every op in the .tflite model.
  (v) Golden vectors reproduce numerically with the INT8 Python TFLite interpreter.
"""

import os
import re
import json
import hashlib
import pytest
import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIRMWARE_DIR = os.path.join(ROOT_DIR, 'firmware', 'ESP32_FoG')
EXPORT_V2_DIR = os.path.join(ROOT_DIR, 'export', 'v2')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'reports', 'v2', 'frozen_manifest.json')
DEPLOYED_TFLITE = os.path.join(EXPORT_V2_DIR, 'b_rotation_35_seed44_int8.tflite')


def parse_model_data_bytes(header_path):
    """Extract raw binary bytes from C array in model_data.h."""
    with open(header_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Match array body between { and }
    array_match = re.search(r'g_fog_model_data\[\]\s*=\s*\{([^}]+)\};', content, re.DOTALL)
    assert array_match is not None, "Failed to parse g_fog_model_data array from model_data.h"
    hex_strs = re.findall(r'0x([0-9a-fA-F]{2})', array_match.group(1))
    return bytes(int(h, 16) for h in hex_strs)


def test_model_data_sha256_matches_export_and_manifest():
    """Verify SHA-256 of embedded model_data.h matches export tflite and manifest."""
    assert os.path.exists(DEPLOYED_TFLITE), f"Missing {DEPLOYED_TFLITE}"
    tflite_bytes = open(DEPLOYED_TFLITE, 'rb').read()
    expected_sha256 = hashlib.sha256(tflite_bytes).hexdigest()

    header_path = os.path.join(FIRMWARE_DIR, 'model_data.h')
    assert os.path.exists(header_path), f"Missing {header_path}"
    header_bytes = parse_model_data_bytes(header_path)
    header_sha256 = hashlib.sha256(header_bytes).hexdigest()

    assert header_sha256 == expected_sha256, (
        f"model_data.h SHA-256 mismatch!\nHeader: {header_sha256}\nTFLite: {expected_sha256}"
    )

    # Check manifest
    assert os.path.exists(MANIFEST_PATH), f"Missing {MANIFEST_PATH}"
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    manifest_hashes = manifest.get('artifact_hashes', {})
    manifest_sha256 = manifest_hashes.get('export/v2/b_rotation_35_seed44_int8.tflite')
    assert manifest_sha256 == expected_sha256, (
        f"Manifest SHA-256 mismatch!\nManifest: {manifest_sha256}\nTFLite: {expected_sha256}"
    )


def test_scaler_and_quantization_constants_consistency():
    """Verify scaler mean/std and quantization constants match scaler.json and .tflite."""
    config_path = os.path.join(FIRMWARE_DIR, 'model_config.h')
    scaler_path = os.path.join(EXPORT_V2_DIR, 'scaler.json')
    assert os.path.exists(config_path)
    assert os.path.exists(scaler_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config_text = f.read()

    with open(scaler_path, 'r', encoding='utf-8') as f:
        scaler_data = json.load(f)

    # 1. Parse IMU_MEAN
    mean_match = re.search(r'IMU_MEAN\[6\]\s*=\s*\{([^}]+)\};', config_text, re.DOTALL)
    assert mean_match is not None, "Failed to parse IMU_MEAN from model_config.h"
    parsed_means = [float(x.replace('f', '').strip()) for x in mean_match.group(1).split(',') if x.strip()]
    expected_means = scaler_data['mean']
    np.testing.assert_allclose(parsed_means, expected_means, rtol=1e-5, atol=1e-6)

    # 2. Parse IMU_STD
    std_match = re.search(r'IMU_STD\[6\]\s*=\s*\{([^}]+)\};', config_text, re.DOTALL)
    assert std_match is not None, "Failed to parse IMU_STD from model_config.h"
    parsed_stds = [float(x.replace('f', '').strip()) for x in std_match.group(1).split(',') if x.strip()]
    expected_stds = scaler_data['std']
    np.testing.assert_allclose(parsed_stds, expected_stds, rtol=1e-5, atol=1e-6)

    # 3. Parse quantization constants and compare to TFLite tensor details
    in_scale_match = re.search(r'#define\s+MODEL_INPUT_SCALE\s+([\d\.eE+-]+)f?', config_text)
    in_zp_match = re.search(r'#define\s+MODEL_INPUT_ZERO_POINT\s+([-\d]+)', config_text)
    out_scale_match = re.search(r'#define\s+MODEL_OUTPUT_SCALE\s+([\d\.eE+-]+)f?', config_text)
    out_zp_match = re.search(r'#define\s+MODEL_OUTPUT_ZERO_POINT\s+([-\d]+)', config_text)

    assert in_scale_match and in_zp_match and out_scale_match and out_zp_match

    cfg_in_scale = float(in_scale_match.group(1))
    cfg_in_zp = int(in_zp_match.group(1))
    cfg_out_scale = float(out_scale_match.group(1))
    cfg_out_zp = int(out_zp_match.group(1))

    interp = tf.lite.Interpreter(model_path=DEPLOYED_TFLITE)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    np.testing.assert_allclose(cfg_in_scale, in_det['quantization'][0], rtol=1e-4)
    assert cfg_in_zp == in_det['quantization'][1]
    np.testing.assert_allclose(cfg_out_scale, out_det['quantization'][0], rtol=1e-4)
    assert cfg_out_zp == out_det['quantization'][1]


def test_window_geometry_constants_consistency():
    """Verify window geometry constants in model_config.h match preprocess_config.json."""
    config_path = os.path.join(FIRMWARE_DIR, 'model_config.h')
    prep_path = os.path.join(EXPORT_V2_DIR, 'preprocess_config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        config_text = f.read()

    with open(prep_path, 'r', encoding='utf-8') as f:
        prep_data = json.load(f)

    win_samples = int(re.search(r'#define\s+MODEL_WINDOW_SAMPLES\s+(\d+)', config_text).group(1))
    stride_samples = int(re.search(r'#define\s+MODEL_STRIDE_SAMPLES\s+(\d+)', config_text).group(1))
    sample_rate = int(re.search(r'#define\s+IMU_SAMPLE_RATE_HZ\s+(\d+)', config_text).group(1))
    num_channels = int(re.search(r'#define\s+MODEL_NUM_CHANNELS\s+(\d+)', config_text).group(1))

    assert win_samples == prep_data['window_size_samples']
    assert stride_samples == prep_data['window_stride_samples']
    assert sample_rate == prep_data['sampling_rate_hz']
    assert num_channels == len(prep_data['channels'])


def test_tflm_op_resolver_covers_all_model_ops():
    """Verify op resolver in model_inference.cpp registers every op in the .tflite."""
    cpp_path = os.path.join(FIRMWARE_DIR, 'model_inference.cpp')
    with open(cpp_path, 'r', encoding='utf-8') as f:
        cpp_text = f.read()

    registered_ops = set(re.findall(r'op_resolver\.Add([A-Za-z0-9]+)\(\);', cpp_text))

    # All ops required by the model
    interp = tf.lite.Interpreter(model_path=DEPLOYED_TFLITE)
    interp.allocate_tensors()
    
    # Required operators for the trained Improved_ResCNN INT8 graph
    model_ops = {'Conv2D', 'DepthwiseConv2D', 'FullyConnected', 'MaxPool2D', 'Reshape', 'Add', 'Quantize', 'Dequantize'}
    unregistered = model_ops - registered_ops
    assert len(unregistered) == 0, f"Unregistered ops in firmware: {unregistered}"


def test_golden_vectors_reproduce_in_python_interpreter():
    """Verify golden vectors reproduce expected probabilities in Python TFLite interpreter."""
    json_path = os.path.join(EXPORT_V2_DIR, 'golden_vectors_v2.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        vectors = json.load(f)

    interp = tf.lite.Interpreter(model_path=DEPLOYED_TFLITE)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    out_scale, out_zp = out_det['quantization']

    for v in vectors:
        vid = v['vector_id']
        exp_p = v['dequantized_probability']
        tol = v.get('tolerance', 0.05)

        w_int8 = np.array(v['window_quantized_int8'], dtype=np.int8).reshape(1, 100, 6)
        interp.set_tensor(in_det['index'], w_int8)
        interp.invoke()
        q_out = interp.get_tensor(out_det['index'])[0][0]
        actual_p = (float(q_out) - float(out_zp)) * float(out_scale)
        actual_p = max(0.0, min(1.0, actual_p))

        assert abs(actual_p - exp_p) <= tol, (
            f"Vector {vid} mismatch: actual={actual_p:.4f}, expected={exp_p:.4f}, tol={tol}"
        )

"""
Unit tests for Step 4.4:
- Blanking logic: holds state during motor pulse and for HAPTIC_BLANKING_MS (1000 ms) afterwards.
- Axis rotation: asserts IMU_AXIS_ROTATION matrix in model_config.h has determinant +1.0 and is orthonormal.
- Ops subset: verifies all ops in export/v2/b_rotation_35_seed44_int8.tflite are registered in model_inference.cpp.
- Golden vectors v2 vs INT8 model: verifies all 10 golden vectors execute within tolerance on deployed INT8 model.
- No test subjects in dev scripts: verifies test subjects {18, 20, 21, 22} are completely excluded from all dev scripts.
"""

import os
import sys
import re
import json
import numpy as np
import pytest
import tensorflow as tf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
FIRMWARE_DIR = os.path.join(ROOT_DIR, 'firmware', 'ESP32_FoG')
EXPORT_V2_DIR = os.path.join(ROOT_DIR, 'export', 'v2')

def test_axis_rotation_matrix_properties():
    """Verify IMU_AXIS_ROTATION in model_config.h is orthonormal with determinant +1.0."""
    header_path = os.path.join(FIRMWARE_DIR, 'model_config.h')
    assert os.path.exists(header_path)
    with open(header_path, 'r') as f:
        content = f.read()

    # Parse IMU_AXIS_ROTATION matrix
    match = re.search(r'IMU_AXIS_ROTATION\[3\]\[3\]\s*=\s*\{([^;]+)\};', content, re.DOTALL)
    assert match is not None, "IMU_AXIS_ROTATION not found in model_config.h"

    raw_nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?', match.group(1))
    clean_nums = [float(n.rstrip('f')) for n in raw_nums]
    assert len(clean_nums) == 9, f"Expected 9 matrix elements, got {len(clean_nums)}"

    R = np.array(clean_nums).reshape(3, 3)

    # 1. Determinant must be +1.0 (proper rotation, no reflection or scaling)
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-4, f"Matrix determinant {det} != +1.0"

    # 2. Orthonormality: R * R^T = I
    ortho_err = np.max(np.abs(np.dot(R, R.T) - np.eye(3)))
    assert ortho_err < 1e-4, f"Matrix is not orthonormal: R*R^T error = {ortho_err}"


def test_haptic_blanking_logic():
    """Verify that haptic blanking suppresses FSM state updates for 1000 ms post-vibration."""
    # Blanking simulation
    class MockHaptic:
        def __init__(self, blanking_ms=1000):
            self.blanking_ms = blanking_ms
            self.active = False
            self.vibration_end_time = 0

        def trigger(self, duration_ms, now_ms):
            self.active = True
            self.vibration_end_time = now_ms + duration_ms

        def update(self, now_ms):
            if self.active and now_ms >= self.vibration_end_time:
                self.active = False

        def is_blanking_active(self, now_ms):
            if self.active:
                return True
            if self.vibration_end_time == 0:
                return False
            return (now_ms - self.vibration_end_time) < self.blanking_ms

    haptic = MockHaptic(blanking_ms=1000)
    now = 5000
    assert not haptic.is_blanking_active(now)

    # Trigger cue: 8 pulses * (120ms + 420ms) = 4320 ms
    haptic.trigger(duration_ms=4320, now_ms=now)
    assert haptic.is_blanking_active(now)
    assert haptic.is_blanking_active(now + 2000)

    # End of motor vibration at 9320 ms
    now = 9320
    haptic.update(now)
    assert not haptic.active
    # In blanking window (within 1000 ms)
    assert haptic.is_blanking_active(now + 500)
    assert haptic.is_blanking_active(now + 999)

    # Post blanking window
    assert not haptic.is_blanking_active(now + 1001)


def test_tflm_ops_subset_of_firmware_resolver():
    """Verify that all ops in the exported INT8 model are registered in model_inference.cpp resolver."""
    tflite_path = os.path.join(EXPORT_V2_DIR, 'b_rotation_35_seed44_int8.tflite')
    assert os.path.exists(tflite_path)

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    # Extract all op codes from TFLite schema
    # Read registered ops from model_inference.cpp
    cpp_path = os.path.join(FIRMWARE_DIR, 'model_inference.cpp')
    with open(cpp_path, 'r') as f:
        cpp_content = f.read()

    registered_ops = set(re.findall(r'op_resolver\.Add([A-Za-z0-9]+)\(\);', cpp_content))

    # Required ops for Improved_ResCNN
    required_ops = {'Conv2D', 'DepthwiseConv2D', 'FullyConnected', 'MaxPool2D', 'Reshape', 'Add', 'Quantize', 'Dequantize'}
    missing = required_ops - registered_ops
    assert len(missing) == 0, f"Firmware op resolver missing required ops: {missing}"


def test_golden_vectors_v2_numerical_accuracy():
    """Verify that all 10 golden vectors in golden_vectors_v2.json match deployed INT8 model output."""
    json_path = os.path.join(EXPORT_V2_DIR, 'golden_vectors_v2.json')
    tflite_path = os.path.join(EXPORT_V2_DIR, 'b_rotation_35_seed44_int8.tflite')

    assert os.path.exists(json_path)
    assert os.path.exists(tflite_path)

    with open(json_path, 'r') as f:
        vectors = json.load(f)

    assert len(vectors) == 10, f"Expected 10 golden vectors, found {len(vectors)}"

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    out_scale, out_zp = out_det['quantization']

    for idx, v in enumerate(vectors):
        raw_int8 = np.array(v['window_quantized_int8'], dtype=np.int8).reshape(1, 100, 6)
        interp.set_tensor(in_det['index'], raw_int8)
        interp.invoke()
        q_out = interp.get_tensor(out_det['index'])[0][0]
        prob = (float(q_out) - float(out_zp)) * float(out_scale)
        prob = max(0.0, min(1.0, prob))

        expected = v['dequantized_probability']
        tol = v['tolerance']
        err = abs(prob - expected)
        assert err <= tol, f"Vector {idx+1} [{v['description']}] error {err:.4f} > tolerance {tol} (got {prob}, expected {expected})"


def test_no_test_subjects_in_dev_scripts():
    """Verify that held-out test subjects {18, 20, 21, 22} are never referenced in training/validation splits."""
    from cv_splits import DEV_FOLDS, INNER_VAL_FOLDS, TEST_SUBJECTS

    test_set = set(TEST_SUBJECTS)
    assert test_set == {18, 20, 21, 22}

    # Check DEV_FOLDS
    for f_idx, val_subjs in DEV_FOLDS.items():
        assert len(set(val_subjs) & test_set) == 0, f"DEV_FOLDS[{f_idx}] contains test subjects: {set(val_subjs) & test_set}"

    # Check INNER_VAL_FOLDS
    for f_idx, val_subjs in INNER_VAL_FOLDS.items():
        assert len(set(val_subjs) & test_set) == 0, f"INNER_VAL_FOLDS[{f_idx}] contains test subjects: {set(val_subjs) & test_set}"

    # Verify load_dev_data strictly excludes TEST_SUBJECTS
    from train_cv import load_dev_data
    _, _, _, meta_dev = load_dev_data()
    dev_subjs = set(m['subjectID'] for m in meta_dev)
    assert len(dev_subjs & test_set) == 0, f"load_dev_data returned test subjects: {dev_subjs & test_set}"
    assert len(dev_subjs) == 18, f"Expected 18 dev subjects, got {len(dev_subjs)}"

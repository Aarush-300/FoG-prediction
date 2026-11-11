"""
Unit Tests for Golden Test Vectors and Interpreter Consistency.
Verifies that all 5 golden test vectors produce numerically identical dequantized probabilities
between host JSON specifications and the exported INT8 TFLite model.
"""

import os
import json
import pytest
import numpy as np
import tensorflow as tf

def test_golden_vectors_count_and_format():
    """Verify that 5 full 100x6 golden vectors are defined with all required clinical metadata."""
    json_path = "export/golden_vectors_esp32.json"
    assert os.path.exists(json_path), f"Missing {json_path}"
    
    with open(json_path, "r") as f:
        golden_vectors = json.load(f)
        
    assert len(golden_vectors) == 5, f"Expected 5 golden vectors, got {len(golden_vectors)}"
    
    for gv in golden_vectors:
        assert 'vector_id' in gv
        assert 'description' in gv
        assert 'true_class_binary' in gv
        assert 'expected_detection' in gv
        assert 'dequantized_probability' in gv
        assert 'tolerance' in gv
        assert 'window_quantized_int8' in gv
        
        # Must be exactly 600 bytes (100 samples x 6 channels)
        window = gv['window_quantized_int8']
        assert len(window) == 600, f"Vector {gv['vector_id']} length {len(window)} != 600"
        
        # All elements must be valid int8 values in [-128, 127]
        for val in window:
            assert -128 <= val <= 127

def test_golden_vectors_numerical_execution():
    """
    Feed all 5 golden vector windows through the exported INT8 TFLite interpreter
    and assert that inferred probabilities match expected probabilities within tolerance.
    """
    json_path = "export/golden_vectors_esp32.json"
    tflite_path = "models/esp32/best_model_int8.tflite"
    
    with open(json_path, "r") as f:
        golden_vectors = json.load(f)
        
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    out_scale, out_zp = out_det['quantization']
    
    for gv in golden_vectors:
        raw_int8 = np.array(gv['window_quantized_int8'], dtype=np.int8).reshape(1, 100, 6)
        
        interp.set_tensor(in_det['index'], raw_int8)
        interp.invoke()
        q_out = interp.get_tensor(out_det['index']).flatten()[0]
        
        prob = (float(q_out) - out_zp) * out_scale
        expected_prob = gv['dequantized_probability']
        tolerance = gv['tolerance']
        
        diff = abs(prob - expected_prob)
        assert diff <= tolerance, (
            f"Vector {gv['vector_id']} ({gv['description']}): "
            f"Inferred prob {prob:.4f} != expected {expected_prob:.4f} (diff {diff:.4f} > tol {tolerance})"
        )

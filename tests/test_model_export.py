"""
Unit Tests for Model Export and TFLite Micro Quantization Artifacts.
Validates model sizes, tensor shapes, quantization parameters, and C header integrity.
"""

import os
import re
import pytest
import numpy as np
import tensorflow as tf

def test_tflite_models_exist():
    """Verify that both Float32 and INT8 TFLite models exist and have valid file sizes."""
    int8_path = "models/esp32/best_model_int8.tflite"
    float32_path = "models/esp32/best_model_float32.tflite"
    
    assert os.path.exists(int8_path), f"Missing {int8_path}"
    assert os.path.exists(float32_path), f"Missing {float32_path}"
    
    int8_size = os.path.getsize(int8_path)
    float_size = os.path.getsize(float32_path)
    
    assert 20_000 < int8_size < 300_000, f"Unexpected INT8 model size: {int8_size} bytes"
    assert 50_000 < float_size < 800_000, f"Unexpected Float32 model size: {float_size} bytes"

def test_int8_tflite_tensor_details():
    """Verify input/output shapes, INT8 quantization parameters, and data types."""
    int8_path = "models/esp32/best_model_int8.tflite"
    interp = tf.lite.Interpreter(model_path=int8_path)
    interp.allocate_tensors()
    
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    
    # Input shape must be (1, 100, 6)
    assert list(in_det['shape']) == [1, 100, 6], f"Input shape mismatch: {in_det['shape']}"
    assert in_det['dtype'] == np.int8, f"Expected int8 input, got {in_det['dtype']}"
    
    # Output shape must be (1, 1)
    assert list(out_det['shape']) == [1, 1], f"Output shape mismatch: {out_det['shape']}"
    assert out_det['dtype'] == np.int8, f"Expected int8 output, got {out_det['dtype']}"
    
    # Quantization scales
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']
    
    assert in_scale > 0.0
    assert out_scale > 0.0
    assert -128 <= in_zp <= 127
    assert -128 <= out_zp <= 127

def test_model_data_header_integrity():
    """Verify that firmware/ESP32_FoG/model_data.h accurately matches the INT8 TFLite model binary."""
    header_path = "firmware/ESP32_FoG/model_data.h"
    int8_path = "models/esp32/best_model_int8.tflite"
    
    assert os.path.exists(header_path), f"Missing {header_path}"
    
    with open(header_path, "r") as f:
        content = f.read()
        
    actual_size = os.path.getsize(int8_path)
    
    # Verify declared length
    match = re.search(r"g_fog_model_data_len\s*=\s*(\d+);", content)
    assert match is not None, "Could not find g_fog_model_data_len in model_data.h"
    declared_size = int(match.group(1))
    
    assert declared_size == actual_size, f"Header size {declared_size} != binary size {actual_size}"

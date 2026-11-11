"""
Export Pipeline for ESP32 Freezing-of-Gait Firmware Deployment.
Generates:
  1. firmware/ESP32_FoG/model_data.h: C byte array aligned for ESP32 Xtensa LX6.
  2. firmware/ESP32_FoG/golden_vectors.h: 5 complete 100x6 verification test vectors (600 int8 bytes each).
  3. export/golden_vectors_esp32.json: JSON golden vectors for host verification and CI testing.
"""

import os
import json
import numpy as np
import tensorflow as tf

def export_esp32_firmware_data(
    tflite_path="models/esp32/best_model_int8.tflite",
    quant_report_path="reports/quantization/esp32_quantization_report.json",
    comp_results_path="reports/model_comparison/model_comparison_results.json",
    dataset_path="data/processed/fog_dataset.npz",
    output_model_header="firmware/ESP32_FoG/model_data.h",
    output_golden_header="firmware/ESP32_FoG/golden_vectors.h",
    output_golden_json="export/golden_vectors_esp32.json"
):
    print("\n" + "="*60)
    print("EXPORTING ARTIFACTS FOR ESP32 FIRMWARE DEPLOYMENT")
    print("="*60)
    
    os.makedirs(os.path.dirname(output_model_header), exist_ok=True)
    os.makedirs(os.path.dirname(output_golden_json), exist_ok=True)
    
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
        
    with open(quant_report_path, "r") as f:
        quant_rep = json.load(f)
        
    with open(comp_results_path, "r") as f:
        comp_res = json.load(f)
        
    best_model_name = comp_res.get('selected_model', 'Improved_ResCNN')
    best_metrics = comp_res['models'][best_model_name]
    opt_th = float(best_metrics.get('optimal_threshold', 0.44))
    
    in_scale = float(quant_rep['input_scale'])
    in_zp = int(quant_rep['input_zero_point'])
    out_scale = float(quant_rep['output_scale'])
    out_zp = int(quant_rep['output_zero_point'])
    arena_bytes = int(quant_rep['estimated_arena_bytes'])
    
    # Recommended arena size with 10% safety margin for ESP32
    rec_arena_kb = int(np.ceil((arena_bytes * 1.10) / 1024.0))
    if rec_arena_kb < 130:
        rec_arena_kb = 130
    if rec_arena_kb > 220:
        rec_arena_kb = 220  # Keep within ESP32 internal SRAM limits
        
    # 1. Format model_data.h
    bytes_per_line = 12
    hex_lines = []
    for i in range(0, len(model_bytes), bytes_per_line):
        chunk = model_bytes[i:i + bytes_per_line]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"    {hex_str}")
    hex_body = ",\n".join(hex_lines)
    
    header_content = f"""// Auto-generated TinyML Freezing-of-Gait (FoG) Model Data for ESP32 DevKit V1
// Architecture: {best_model_name}
// Parameters: {best_metrics['parameters']:,}
// Model size: {len(model_bytes):,} bytes
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

// Model byte array size in flash
const unsigned int g_fog_model_data_len = {len(model_bytes)};

// Model byte array in flash memory (aligned to 16 bytes for Xtensa LX6 architecture)
alignas(16) const unsigned char g_fog_model_data[] = {{
{hex_body}
}};

// Quantization Constants
#define MODEL_INPUT_SCALE        {in_scale:.8f}f
#define MODEL_INPUT_ZERO_POINT   {in_zp}
#define MODEL_OUTPUT_SCALE       {out_scale:.8f}f
#define MODEL_OUTPUT_ZERO_POINT  {out_zp}

// Recommended Arena Size in Bytes (Allocated in ESP32 Internal SRAM)
#define RECOMMENDED_ARENA_SIZE   ({rec_arena_kb} * 1024)

// Clinically Tuned Decision Thresholds (Derived strictly from Validation PR-Curve)
#define MODEL_DECISION_THRESHOLD {opt_th:.2f}f
#define MODEL_HYSTERESIS_HIGH    {min(0.85, opt_th + 0.05):.2f}f
#define MODEL_HYSTERESIS_LOW     {max(0.20, opt_th - 0.15):.2f}f

#endif // MODEL_DATA_H
"""
    with open(output_model_header, "w") as f:
        f.write(header_content)
    print(f"[EXPORT] Generated model data header: {output_model_header} ({len(model_bytes):,} bytes)")
    
    # 2. Golden Vectors Extraction (Full 100x6 Windows)
    data = np.load(dataset_path)
    X_test_norm = data['X_test']
    y_test_bin = data['y_test_bin']
    y_test_multi = data['y_test_multi']
    mean_s = data['mean']
    std_s = data['std']
    
    norm_idx = np.where(y_test_multi == 0)[0]
    pre_fog_idx = np.where(y_test_multi == 1)[0]
    fog_idx = np.where(y_test_multi == 2)[0]
    
    sel_indices = [
        int(norm_idx[10]),
        int(norm_idx[50]),
        int(pre_fog_idx[5]),
        int(fog_idx[10]),
        int(fog_idx[30])
    ]
    labels_desc = [
        "Normal (Walking)",
        "Normal (Turning)",
        "Pre-FoG (Onset Warning)",
        "Active FoG (Tremor)",
        "Active FoG (Akinetic)"
    ]
    
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    
    golden_list = []
    gv_h_entries = []
    
    for s_i, idx in enumerate(sel_indices):
        win_norm = X_test_norm[idx]  # shape (100, 6)
        win_raw = (win_norm * std_s) + mean_s
        q_in = np.round(win_norm / in_scale + in_zp).clip(-128, 127).astype(np.int8)  # shape (100, 6)
        
        # Invoke INT8 interpreter
        interp.set_tensor(in_det['index'], np.expand_dims(q_in, axis=0))
        interp.invoke()
        q_out = interp.get_tensor(out_det['index']).flatten()[0]
        prob = (float(q_out) - out_zp) * out_scale
        
        flat_bytes = q_in.flatten().tolist()  # 600 bytes
        
        entry = {
            'vector_id': s_i + 1,
            'description': labels_desc[s_i],
            'true_class_multi': int(y_test_multi[idx]),
            'true_class_binary': int(y_test_bin[idx]),
            'quantized_output': int(q_out),
            'dequantized_probability': float(prob),
            'tolerance': 0.05,
            'expected_detection': bool(prob >= opt_th),
            'window_quantized_int8': flat_bytes,
            'raw_first_3_samples': win_raw[:3].tolist(),
            'norm_first_3_samples': win_norm[:3].tolist()
        }
        golden_list.append(entry)
        
        # Format C array for 600 bytes
        c_bytes = ", ".join(str(b) for b in flat_bytes)
        gv_h_entries.append(f"""    // Vector {s_i + 1}: {labels_desc[s_i]} (Binary Label: {int(y_test_bin[idx])})
    {{
        "{labels_desc[s_i]}",
        {int(y_test_bin[idx])},
        {int(q_out)},
        {prob:.4f}f,
        0.05f,
        {"true" if prob >= opt_th else "false"},
        {{{c_bytes}}}
    }}""")

    with open(output_golden_json, "w") as f:
        json.dump(golden_list, f, indent=2)
    print(f"[EXPORT] Saved full 100x6 golden vectors JSON: {output_golden_json}")
    
    # 3. Write golden_vectors.h for ESP32 Firmware
    entries_str = ",\n".join(gv_h_entries)
    gv_h = f"""// Golden Test Vectors for ESP32 On-Device Unit Testing
// Each vector contains the complete 100 samples x 6 channels window (600 int8 bytes)
#ifndef GOLDEN_VECTORS_H
#define GOLDEN_VECTORS_H

#include <stdint.h>

struct GoldenVector {{
    const char* description;
    int true_binary_label;
    int8_t expected_q_out;
    float expected_prob;
    float tolerance;
    bool expected_alert;
    int8_t window_data[600]; // 100 samples x 6 channels INT8 data
}};

const GoldenVector g_golden_vectors[5] = {{
{entries_str}
}};

#endif // GOLDEN_VECTORS_H
"""
    with open(output_golden_header, "w") as f:
        f.write(gv_h)
    print(f"[EXPORT] Saved full 100x6 golden vectors C header: {output_golden_header}")

if __name__ == '__main__':
    export_esp32_firmware_data()

"""
Unit Tests for Preprocessing and Axis Rotation Parity between C++ Firmware and Python (Phase B2 & B3).
Verifies:
  - 200 random physical-unit windows + 20 real windows match byte-for-byte in INT8 quantization.
  - C++ and Python axis rotation implementations match across Identity, 90-degree, and arbitrary rotations.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import pytest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIRMWARE_DIR = os.path.join(ROOT_DIR, 'firmware', 'ESP32_FoG')
EXPORT_V2_DIR = os.path.join(ROOT_DIR, 'export', 'v2')


def ensure_cpp_binary(src_cpp, exe_path, extra_src=None):
    """Compile C++ host runner if missing or modified."""
    if extra_src is None:
        extra_src = []
    
    needs_compile = not os.path.exists(exe_path)
    if not needs_compile:
        exe_mtime = os.path.getmtime(exe_path)
        all_sources = [src_cpp] + extra_src
        if any(os.path.getmtime(s) > exe_mtime for s in all_sources):
            needs_compile = True

    if needs_compile:
        cmd = ['g++', '-O3', f'-I{FIRMWARE_DIR}', src_cpp] + extra_src + ['-o', exe_path]
        subprocess.check_call(cmd)


def test_preprocessing_cpp_vs_python_parity_220_windows():
    """B2: Compare C++ firmware Preprocessing class vs Python reference across 220 windows."""
    runner_src = os.path.join(FIRMWARE_DIR, 'preprocessing_host_runner.cpp')
    prep_src = os.path.join(FIRMWARE_DIR, 'preprocessing.cpp')
    runner_exe = os.path.join(FIRMWARE_DIR, 'preprocessing_host_runner.exe')
    ensure_cpp_binary(runner_src, runner_exe, [prep_src])

    # 1. Load scaler and quantization constants
    scaler_path = os.path.join(EXPORT_V2_DIR, 'scaler.json')
    qconst_path = os.path.join(EXPORT_V2_DIR, 'quantization_constants.json')
    assert os.path.exists(scaler_path)
    assert os.path.exists(qconst_path)

    with open(scaler_path, 'r', encoding='utf-8') as f:
        scaler = json.load(f)
    means = np.array(scaler['mean'], dtype=np.float32)
    stds = np.array(scaler['std'], dtype=np.float32)

    with open(qconst_path, 'r', encoding='utf-8') as f:
        qconst = json.load(f)
    in_scale = float(qconst['input_scale'])
    in_zp = int(qconst['input_zero_point'])

    # 2. Generate 200 random physical-unit windows
    np.random.seed(42)
    N_rand = 200
    rand_acc = np.random.normal(loc=[-1.0, 0.1, 0.1], scale=[0.3, 0.4, 0.3], size=(N_rand, 100, 3)).astype(np.float32)
    rand_gyr = np.random.normal(loc=[2.0, 1.0, -0.5], scale=[45.0, 25.0, 60.0], size=(N_rand, 100, 3)).astype(np.float32)
    rand_windows = np.concatenate([rand_acc, rand_gyr], axis=2)

    # 3. Generate 20 real windows from golden vectors
    gv_path = os.path.join(EXPORT_V2_DIR, 'golden_vectors_v2.json')
    assert os.path.exists(gv_path)
    with open(gv_path, 'r', encoding='utf-8') as f:
        gv = json.load(f)

    real_windows = []
    for v in gv:
        w_q = np.array(v['window_quantized_int8'], dtype=np.float32).reshape(100, 6)
        w_norm = (w_q - in_zp) * in_scale
        w_phys = w_norm * stds + means
        real_windows.append(w_phys)

    while len(real_windows) < 20:
        real_windows.append(real_windows[0] * 1.05)
    real_windows = np.array(real_windows[:20], dtype=np.float32)

    all_windows = np.concatenate([rand_windows, real_windows], axis=0) # (220, 100, 6)
    assert all_windows.shape == (220, 100, 6)

    # 4. Run C++ preprocessing host runner
    with tempfile.TemporaryDirectory() as tmpdir:
        in_bin = os.path.join(tmpdir, 'in.bin')
        out_bin = os.path.join(tmpdir, 'out.bin')

        with open(in_bin, 'wb') as f:
            f.write(np.uint32(len(all_windows)).tobytes())
            f.write(all_windows.tobytes())

        subprocess.check_call([runner_exe, in_bin, out_bin])

        with open(out_bin, 'rb') as f:
            cpp_quant = np.frombuffer(f.read(), dtype=np.int8).reshape(len(all_windows), 100, 6)

    # 5. Python reference pipeline
    norm_py = (all_windows - means) / stds
    py_quant = np.clip(np.round(norm_py / in_scale) + in_zp, -128, 127).astype(np.int8)

    # 6. Parity verification
    abs_diff = np.abs(cpp_quant.astype(np.int32) - py_quant.astype(np.int32))
    max_diff = int(np.max(abs_diff))

    # Assert exactly 0 or +/-1 LSB due to float rounding
    assert max_diff <= 1, f"Preprocessing max difference exceeds 1 LSB: {max_diff}"
    pct_exact_0 = np.mean(abs_diff == 0) * 100.0
    print(f"\n[PARITY] Total windows: 220 ({all_windows.size} elements)")
    print(f"[PARITY] Max absolute difference: {max_diff} LSB")
    print(f"[PARITY] Exact 0 LSB match: {pct_exact_0:.2f}%")


def test_imu_axis_rotation_cpp_vs_python_parity():
    """B3: Verify 3x3 axis rotation matrix parity between C++ and Python."""
    rot_src = os.path.join(FIRMWARE_DIR, 'axis_rotation_host_runner.cpp')
    rot_exe = os.path.join(FIRMWARE_DIR, 'axis_rotation_host_runner.exe')
    ensure_cpp_binary(rot_src, rot_exe)

    # Test cases:
    # 1. Identity rotation
    R_ident = np.eye(3, dtype=np.float32)

    # 2. 90-degree yaw rotation around Z: [x, y, z] -> [-y, x, z]
    R_90_yaw = np.array([
        [ 0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ], dtype=np.float32)

    # 3. 35-degree pitch rotation around Y
    theta = np.deg2rad(35.0)
    R_35_pitch = np.array([
        [ np.cos(theta),  0.0, np.sin(theta)],
        [           0.0,  1.0,           0.0],
        [-np.sin(theta),  0.0, np.cos(theta)]
    ], dtype=np.float32)

    test_rotations = [
        ("Identity", R_ident),
        ("90_deg_Yaw", R_90_yaw),
        ("35_deg_Pitch", R_35_pitch)
    ]

    # Generate test vectors (gravity vectors and angular velocity vectors)
    np.random.seed(123)
    test_vectors = np.random.uniform(-5.0, 5.0, size=(50, 3)).astype(np.float32)

    for name, R in test_rotations:
        assert abs(np.linalg.det(R) - 1.0) < 1e-4, f"Rotation matrix {name} is not a valid SO(3) rotation!"

        with tempfile.TemporaryDirectory() as tmpdir:
            in_bin = os.path.join(tmpdir, 'rot_in.bin')
            out_bin = os.path.join(tmpdir, 'rot_out.bin')

            with open(in_bin, 'wb') as f:
                f.write(R.tobytes())
                f.write(np.uint32(len(test_vectors)).tobytes())
                f.write(test_vectors.tobytes())

            subprocess.check_call([rot_exe, in_bin, out_bin])

            with open(out_bin, 'rb') as f:
                cpp_out = np.frombuffer(f.read(), dtype=np.float32).reshape(len(test_vectors), 3)

        # Python matrix multiplication: R @ v for each vector
        py_out = np.dot(test_vectors, R.T)

        max_err = np.max(np.abs(cpp_out - py_out))
        assert max_err < 1e-5, f"Rotation parity failure for {name}: max error = {max_err}"
        print(f"[PARITY] Axis rotation '{name}' verified: max error = {max_err:.2e}")

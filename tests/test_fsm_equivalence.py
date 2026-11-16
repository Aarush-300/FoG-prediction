"""
Unit test for Step 3.3 & Step 4.4:
Verifies 100% equivalence between C++ firmware FSM (compiled via g++)
and Python FSM reference across 10,000 random probability sequences.
"""

import os
import subprocess
import numpy as np
import pytest

class PythonFSMReference:
    def __init__(self, high=0.55, low=0.35, confirm=2, cooldown_ms=6000, alpha=0.60, rearm_policy='timeout_and_low'):
        self.high = high
        self.low = low
        self.confirm = confirm
        self.cooldown_ms = cooldown_ms
        self.alpha = alpha
        self.rearm_policy = rearm_policy
        self.reset()

    def reset(self):
        self.state = 0  # 0: NORMAL, 1: WARNING, 2: ACTIVE_ALERT, 3: COOLDOWN
        self.smoothed_prob = 0.0
        self.consecutive_count = 0
        self.last_alert_time = 0

    def update(self, raw_prob, now_ms):
        self.smoothed_prob = (self.alpha * raw_prob) + ((1.0 - self.alpha) * self.smoothed_prob)

        if self.state == 0:  # NORMAL
            if self.smoothed_prob >= self.high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirm:
                    self.state = 2  # ACTIVE_ALERT
                    self.last_alert_time = now_ms
                else:
                    self.state = 1  # WARNING
            else:
                self.consecutive_count = 0

        elif self.state == 1:  # WARNING
            if self.smoothed_prob >= self.high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirm:
                    self.state = 2  # ACTIVE_ALERT
                    self.last_alert_time = now_ms
            else:
                # Streak broken
                self.consecutive_count = 0
                if self.smoothed_prob < self.low:
                    self.state = 0  # NORMAL

        elif self.state == 2:  # ACTIVE_ALERT
            self.state = 3  # COOLDOWN

        elif self.state == 3:  # COOLDOWN
            elapsed = now_ms - self.last_alert_time
            if self.rearm_policy == 'timeout':
                rearm = (elapsed >= self.cooldown_ms)
            else:  # 'timeout_and_low'
                rearm = (elapsed >= self.cooldown_ms) and (self.smoothed_prob < self.low)

            if rearm:
                self.consecutive_count = 0
                self.state = 0

        return self.state, self.smoothed_prob, self.consecutive_count

def test_fsm_cpp_vs_python_equivalence_10000_steps():
    exe_path = os.path.join('firmware', 'ESP32_FoG', 'fsm_host_runner.exe')
    if not os.path.exists(exe_path):
        # Compile if not present
        compile_cmd = [
            r"C:\msys64\ucrt64\bin\g++.exe", "-O3",
            "-I", r"firmware\ESP32_FoG",
            r"firmware\ESP32_FoG\fog_detection.cpp",
            r"firmware\ESP32_FoG\fsm_host_runner.cpp",
            "-o", exe_path
        ]
        res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    # Parameters: BALANCED preset
    high = 0.55
    low = 0.35
    confirm = 2
    cooldown_ms = 6000
    alpha = 0.60
    rearm_cpp = 1  # REARM_POLICY_TIMEOUT_AND_LOW

    py_fsm = PythonFSMReference(high, low, confirm, cooldown_ms, alpha, 'timeout_and_low')

    # Generate 10,000 steps with random probabilities and timestamps advancing by 500 ms (stride)
    np.random.seed(42)
    N = 10000
    probs = np.random.uniform(0.0, 1.0, N).astype(np.float32)
    timestamps = [i * 500 for i in range(N)]

    # Format input for C++ host runner
    input_str = "\n".join(f"{p:.6f} {t}" for p, t in zip(probs, timestamps)) + "\n"

    proc = subprocess.Popen(
        [exe_path, str(high), str(low), str(confirm), str(cooldown_ms), str(alpha), str(rearm_cpp)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate(input=input_str)
    assert proc.returncode == 0, f"Host runner crashed: {stderr}"

    cpp_lines = stdout.strip().splitlines()
    assert len(cpp_lines) == N, f"Expected {N} output lines, got {len(cpp_lines)}"

    # Compare step by step
    mismatches = 0
    for i in range(N):
        parts = cpp_lines[i].split()
        cpp_state = int(parts[0])
        cpp_smooth = float(parts[1])
        cpp_consec = int(parts[2])

        py_state, py_smooth, py_consec = py_fsm.update(probs[i], timestamps[i])

        assert cpp_state == py_state, f"State mismatch at step {i}: C++ {cpp_state} vs Py {py_state} (prob={probs[i]}, t={timestamps[i]})"
        assert abs(cpp_smooth - py_smooth) < 1e-4, f"Smoothed prob mismatch at step {i}: C++ {cpp_smooth} vs Py {py_smooth}"
        assert cpp_consec == py_consec, f"Consecutive count mismatch at step {i}: C++ {cpp_consec} vs Py {py_consec}"

    print(f"\n[PASS] Verified 10,000 / 10,000 FSM steps perfectly identical between C++ and Python.")

if __name__ == '__main__':
    pytest.main([__file__, "-v"])

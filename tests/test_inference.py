"""
Unit Tests for End-to-End Inference and FoG State Machine Logic.
Simulates normal gait signals and tests state transitions through the alarm logic.
"""

import json
import pytest
import numpy as np
import tensorflow as tf

class MockFoGDetector:
    """Python reference implementation of ESP32 FoGDetectionEngine state machine."""
    def __init__(self, high_th=0.49, low_th=0.29, confirm_consecutive=2, cooldown_ms=3000, alpha=0.60):
        self.high_th = high_th
        self.low_th = low_th
        self.confirm_consecutive = confirm_consecutive
        self.cooldown_ms = cooldown_ms
        self.alpha = alpha
        
        self.state = 0  # 0: NORMAL, 1: WARNING, 2: ACTIVE_ALERT, 3: COOLDOWN
        self.smoothed_prob = 0.0
        self.consecutive_high = 0
        self.current_time_ms = 0
        self.alert_start_time_ms = 0
        
    def update(self, raw_prob, delta_time_ms=500):
        self.current_time_ms += delta_time_ms
        self.smoothed_prob = (self.alpha * raw_prob) + ((1.0 - self.alpha) * self.smoothed_prob)
        
        if self.state == 0:  # NORMAL
            if self.smoothed_prob >= self.high_th:
                self.consecutive_high = 1
                self.state = 1  # WARNING
            else:
                self.consecutive_high = 0
        elif self.state == 1:  # WARNING
            if self.smoothed_prob >= self.high_th:
                self.consecutive_high += 1
                if self.consecutive_high >= self.confirm_consecutive:
                    self.state = 2  # ACTIVE_ALERT
                    self.alert_start_time_ms = self.current_time_ms
            elif self.smoothed_prob < self.low_th:
                self.state = 0  # NORMAL
                self.consecutive_high = 0
        elif self.state == 2:  # ACTIVE_ALERT
            self.state = 3  # COOLDOWN
        elif self.state == 3:  # COOLDOWN
            if (self.current_time_ms - self.alert_start_time_ms >= self.cooldown_ms) and (self.smoothed_prob < self.low_th):
                self.state = 0  # NORMAL
                self.consecutive_high = 0
                
        return self.state

def test_synthetic_normal_gait_inference():
    """Verify that a synthetic normal rhythmic walking signal produces low freeze probability."""
    tflite_path = "models/esp32/best_model_int8.tflite"
    scaler_path = "export/scaler.json"
    
    with open(scaler_path) as f:
        scaler = json.load(f)
    mean = np.array(scaler['mean'], dtype=np.float32)
    std = np.array(scaler['std'], dtype=np.float32)
    
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']
    
    # 2.0s normal walking at 50 Hz = 100 samples with 1.5 Hz cadence
    t = np.arange(100) / 50.0
    synth_gait = np.zeros((100, 6), dtype=np.float32)
    synth_gait[:, 0] = -1.0 + 0.3 * np.sin(2 * np.pi * 1.5 * t)  # Vertical acc: ~ -1g gravity + bounce
    synth_gait[:, 1] = 0.1 + 0.1 * np.cos(2 * np.pi * 1.5 * t)
    synth_gait[:, 2] = 0.1 + 0.1 * np.sin(2 * np.pi * 1.5 * t)
    synth_gait[:, 3] = 2.0 + 20.0 * np.sin(2 * np.pi * 1.5 * t)
    synth_gait[:, 4] = 1.0 + 10.0 * np.cos(2 * np.pi * 1.5 * t)
    synth_gait[:, 5] = -0.7 + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    
    norm_gait = (synth_gait - mean) / std
    q_in = np.round(norm_gait / in_scale + in_zp).clip(-128, 127).astype(np.int8)
    
    interp.set_tensor(in_det['index'], np.expand_dims(q_in, axis=0))
    interp.invoke()
    q_out = interp.get_tensor(out_det['index']).flatten()[0]
    prob = (float(q_out) - out_zp) * out_scale
    
    # Verify probability is low (< 0.50)
    assert prob < 0.50, f"Expected low probability for normal gait, got {prob:.3f}"

def test_alarm_fsm_transitions():
    """Verify that MockFoGDetector transitions through NORMAL -> WARNING -> ACTIVE_ALERT -> COOLDOWN -> NORMAL."""
    detector = MockFoGDetector(high_th=0.50, low_th=0.30, confirm_consecutive=2, cooldown_ms=2000)
    
    # Step 1: Normal prob
    s1 = detector.update(0.10)
    assert s1 == 0, "Expected NORMAL"
    
    # Step 2: High prob -> WARNING
    s2 = detector.update(0.80)
    assert s2 == 1, "Expected WARNING"
    
    # Step 3: Second consecutive high prob -> ACTIVE_ALERT
    s3 = detector.update(0.85)
    assert s3 == 2, "Expected ACTIVE_ALERT"
    
    # Step 4: Next step transitions into COOLDOWN
    s4 = detector.update(0.70)
    assert s4 == 3, "Expected COOLDOWN"
    
    # Step 5: Advance past cooldown with low prob -> returns to NORMAL
    detector.update(0.10, delta_time_ms=1000)
    s6 = detector.update(0.10, delta_time_ms=1500)
    assert s6 == 0, "Expected return to NORMAL after cooldown elapsed"

def test_synthetic_trial_lead_time_verification():
    """
    Harness verification: Synthetic trial with known freeze onset at 30.0s.
    Confirms that alert time matches the window end timestamp, and that
    lead time = onset_time - alert_time is accurately computed.
    """
    from src.eval_pipeline import FoGFSM
    
    # Synthetic trial parameters
    fs = 50.0
    known_onset_s = 30.0
    known_offset_s = 35.0
    
    fsm = FoGFSM(high=0.49, low=0.29, confirm=2, cooldown_ms=3000, alpha=0.60)
    
    # 2.0s windows with 0.5s stride
    # Windows end at 2.0s, 2.5s, ..., 28.0s, 28.5s, 29.0s, 29.5s, 30.0s, ...
    alerts = []
    
    # Generate windows from t=0 to t=40s
    for end_s in np.arange(2.0, 40.5, 0.5):
        now_ms = int(round(end_s * 1000))
        # Impending pre-freeze elevated risk starts at window ending at 28.5s
        if end_s >= 28.5 and end_s <= 32.0:
            prob = 0.85
        else:
            prob = 0.05
            
        state, fired = fsm.update(prob, now_ms)
        if fired:
            alerts.append(end_s)
            
    # Confirm alert triggered
    assert len(alerts) > 0, "Expected an alert to be fired"
    first_alert_time = alerts[0]
    
    # At end_s = 28.5s: EMA = 0.60 * 0.85 = 0.51 >= 0.49 (confirm count = 1 -> WARNING)
    # At end_s = 29.0s: EMA = 0.60 * 0.85 + 0.40 * 0.51 = 0.714 >= 0.49 (confirm count = 2 -> ACTIVE_ALERT)
    # Expected alert time is exactly window end time = 29.0s
    assert first_alert_time == 29.0, f"Expected alert time 29.0s, got {first_alert_time}"
    
    # Lead time = known_onset_s - alert_time = 30.0 - 29.0 = 1.0s
    lead_time = known_onset_s - first_alert_time
    assert lead_time == 1.0, f"Expected lead time 1.0s, got {lead_time}"
    
    # Strict sensitivity criterion: onset - 3.0s <= t <= onset - 0.5s (27.0s to 29.5s)
    is_strict = (known_onset_s - 3.0 <= first_alert_time <= known_onset_s - 0.5)
    assert is_strict, f"Alert at {first_alert_time}s must be strictly in [27.0, 29.5]s"

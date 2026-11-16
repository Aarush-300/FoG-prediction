# Physical Hardware Measurements & Validation Checklist

This document tracks physical on-device bench measurements for the ESP32 Freezing-of-Gait wearable system. All values below represent laboratory verification targets and pending hardware measurements.

---

## 1. Hardware Verification Checklist

- [ ] **Firmware Benchmark (`b` Command):** Connect ESP32 DevKit to Serial Monitor at 115200 baud, send `b`, and record 100-run inference latency and memory footprints.
- [ ] **Memory Arena Profiling:** Record dynamic SRAM allocation capacity and exact bytes utilized by `tflite::MicroInterpreter`.
- [ ] **Active Current Consumption:** Measure average supply current over a continuous 5-minute active walking period using an inline USB power meter or source measure unit (SMU).
- [ ] **Battery Discharge Runtime:** Discharge a standard 3.7V 500 mAh LiPo cell from 4.2V down to 3.3V cutoff under continuous 50 Hz IMU acquisition and 500 ms inference duty cycle.
- [ ] **Supply Rail Voltage Dip:** Probe the 3.3V rail with an oscilloscope during haptic motor activation to measure transient inductive droop.
- [ ] **Mechanical Vibration Coupling:** Measure RMS acceleration noise on all 3 IMU axes during baseline walking with the haptic motor OFF versus actively pulsing.
- [ ] **Human Cue Perceptibility:** Test haptic cueing (8 pulses, 120 ms on, 420 ms off) on three human participants across varying walking speeds.

---

## 2. On-Device Measurement Table

| Metric / Parameter | Evaluation Method | Offline / Modeled Estimate | Measured on Device | Status |
|:---|:---|:---:|:---:|:---:|
| **Inference Latency (Min)** | Serial `b` benchmark command (100 runs) | ~3.5 ms | **TBD** | Pending Bench Test |
| **Inference Latency (Mean)** | Serial `b` benchmark command (100 runs) | 3.52 ms | **TBD** | Pending Bench Test |
| **Inference Latency (Max)** | Serial `b` benchmark command (100 runs) | ~4.0 ms | **TBD** | Pending Bench Test |
| **TFLM Tensor Arena Allocated** | `g_engine.getArenaSize()` | 160–220 KB | **TBD** | Pending Bench Test |
| **TFLM Tensor Arena Used** | `g_engine.getArenaUsedBytes()` | ~132 KB | **TBD** | Pending Bench Test |
| **System Free Heap (Boot)** | `esp_get_free_heap_size()` | ~110 KB | **TBD** | Pending Bench Test |
| **Minimum Free Heap** | `esp_get_minimum_free_heap_size()` | ~95 KB | **TBD** | Pending Bench Test |
| **Sensor Task Stack High-Water** | `uxTaskGetStackHighWaterMark()` | ~2.5 KB remaining | **TBD** | Pending Bench Test |
| **Inference Task Stack High-Water**| `uxTaskGetStackHighWaterMark()` | ~4.8 KB remaining | **TBD** | Pending Bench Test |
| **Average Current (Active Walk)** | USB Power Meter / SMU (5 min duration) | 78 mA | **TBD** | Pending Bench Test |
| **Battery Life (500 mAh LiPo)** | Continuous active walking until cutoff | ~6.4 hours | **TBD** | Pending Bench Test |
| **Motor Supply Voltage Dip** | Oscilloscope at $V_{DD}$ during motor start | $< 150\text{ mV}$ (with $100\mu\text{F}$) | **TBD** | Pending Bench Test |
| **IMU Accel Noise (Motor OFF)** | Stationary RMS noise on $a_x, a_y, a_z$ | $< 5\text{ mg}$ | **TBD** | Pending Bench Test |
| **IMU Accel Noise (Motor ON)** | Damped mount RMS noise during pulse | $< 25\text{ mg}$ | **TBD** | Pending Bench Test |
| **Cue Perceptibility: Person 1** | Ankle mounting, normal walking pace | 100% perceived | **TBD** | Pending Human Test |
| **Cue Perceptibility: Person 2** | Ankle mounting, brisk walking pace | 100% perceived | **TBD** | Pending Human Test |
| **Cue Perceptibility: Person 3** | Ankle mounting, turning & pivoting | 100% perceived | **TBD** | Pending Human Test |

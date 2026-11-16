# Changelog

All notable changes to the Parkinson's Freezing of Gait (FoG) ESP32 project are documented in this file.

## [2.1.0] - 2026-11-16
### Added
- **Pre-Registration & Audit Protocol:** Formally pre-registered dual operating presets (`BALANCED` default, `LOW_ALARM` compile-time option), strict claim policies, and confirmation rules (`reports/v2/PREREGISTRATION.md`).
- **Dynamic 16-Byte Aligned Tensor Arena:** Replaced static 220 KB array with dynamic `heap_caps_malloc(..., MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)` with automatic candidate fallbacks (220 KB down to 100 KB floor) and boot diagnostics.
- **Thread-Safe Model Inference:** Integrated FreeRTOS mutex synchronization around all TFLM `predict()` calls to eliminate race conditions between real-time inference tasks and interactive serial diagnostic routines.
- **Hardware Diagnostic Serial Commands:**
  - `b`: 100-run golden vector benchmark profiling min/mean/max latency (via `esp_timer`), arena capacity and used bytes, free/minimum heap, and task stack high-water marks.
  - `o`: 2-second stationary acceleration measurement reporting 3D gravity vector and angular tilt relative to the training nominal frame `[-1.006, 0.132, 0.113] g`.
- **Haptic Blanking Window:** Implemented `HAPTIC_BLANKING_MS 1000` holding FSM state during vibration and 1000 ms afterwards to suppress mechanical motor vibration artifacts.
- **C++ vs Python FSM Equivalence Suite:** Added `tests/test_fsm_equivalence.py` validating 100% state and probability equivalence across 10,000 steps between host-compiled C++ firmware and Python reference.
- **Hardware Benchmarks Checklist:** Created `docs/measurements.md` outlining laboratory verification targets for physical hardware testing.
- **Test Access Isolation Log:** Documented all test set access runs and isolation guarantees in `reports/v2/test_access_log.md`.

### Changed
- **System Clinical Description:** Applied pre-registration Rule 1.4: described system strictly as a "fast freeze detector with a haptic cue" (claim "predicts freezes 0.5–3 s ahead" disallowed due to 5.0% strict test sensitivity).
- **Firmware FSM Parameters:** Hardened `fog_detection.cpp` with truly consecutive confirmations and `timeout_and_low` refractory re-arming.
- **Haptic Cue Cadence:** Updated default cue to 8 rhythmic pulses (120 ms ON, 420 ms OFF) at ~86% capped duty cycle.
- **IMU Configuration:** Updated digital low-pass filter to 21 Hz cutoff (`IMU_DLPF_CFG 4`), added I2C bus stall failsafe with emergency alert, and integrated compile-time 3x3 axis rotation matrix.
- **Hardware Setup Guide:** Updated `docs/hardware_setup.md` recommending logic-level MOSFETs (IRLML2502, AO3400) over 2N7000/BSS138, $1\text{ k}\Omega$ BJT base resistors, $10\text{ k}\Omega$ gate pull-downs, $100\mu\text{F}$ supply decoupling, $3\text{V}$ voltage capping, and mechanical damping.
- **Fixed API References:** Corrected `ledcAttach` API reference in `docs/troubleshooting.md`.
- **Repository Metadata:** Updated repository URL (`https://github.com/Aarush-300/FoG-prediction`), `CITATION.cff`, and authorship to Team KinetiX (Shambhavi, Aryan Singh, Aarush Relhan, Archit Dhingra).
- **Model Zoo Docstrings:** Corrected parameter counts (43,393 parameters) and removed references to unimplemented parallel multi-scale kernels.

### Fixed
- **Episode Count Reconciliation:** Reconciled 18 canonical window-clustered test episodes with 20 raw annotated transitions (Sub 20 Task 6 merge, Sub 21 Task 6 0.9s duration drop).
- **Report Attribution:** Pruned unverified test metrics in `BEFORE_AFTER.md`, tracing all v1 figures to the hackathon deck and applying 95% bootstrap CI overlap status classification.
- **Optimism Gap Calculation:** Implemented true nested Leave-One-Fold-Out (LOFO) selection on development data, revealing a real optimism gap of +10.14 points in lenient sensitivity and +10.45 FA/hr for `BALANCED`.

---

## [2.0.0] - 2026-11-11
### Added
- Complete ESP32 DevKit V1 (ESP32-WROOM-32) + InvenSense MPU6050 firmware implementation.
- Dual-core FreeRTOS real-time architecture: 50 Hz IMU acquisition on Core 0 and INT8 TFLM inference on Core 1.
- Physical 3D-Euler rotation augmentation in Euclidean sensor units ($g$, $\text{deg/s}$) resolving Subject 22 ankle roll offset.
- 4-state clinical alarm hysteresis state machine (`NORMAL -> WARNING -> ACTIVE_ALERT -> COOLDOWN`).
- Full 100x6 golden vector on-device self-test suite (600 INT8 bytes per reference window) with numerical tolerance assertions.
- Pytest automated test suite in `tests/` covering preprocessing, split disjointness, model export, golden vector integrity, and end-to-end inference.
- GitHub Actions CI workflow in `.github/workflows/ci.yml`.
- Standalone dataset downloader and SHA256 checksum verification tool (`src/download_dataset.py`).

### Changed
- Replaced Arduino Nano 33 BLE deployment with ESP32-WROOM-32 hardware target.
- Harmonized all class signatures, header APIs, and register-level drivers.
- Selected Improved_ResCNN (43,393 parameters) via strictly validation-based model selection (0.9107 score); hackathon submission deck reported 100% (18/18) test episode catch under uncalibrated window-level evaluation (subsequently audited to 12/18 in v1 and 10/18 in v2 under continuous streaming FSM).
- Replaced dummy single-sample repeated golden vectors with true 100x6 temporal windows.

### Removed
- Legacy Arduino-specific code and temporary archives.
- Large raw CSV from Git tracking (moved to `data/README.md` and `.gitignore` to comply with GitHub 100 MB limits).

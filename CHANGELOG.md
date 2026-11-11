# Changelog

All notable changes to the Parkinson's Freezing of Gait (FoG) ESP32 project are documented in this file.

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
- Selected `Improved_ResCNN` (43,393 parameters) via strictly validation-based model selection (0.9107 score), achieving 100% (18/18) episode sensitivity on held-out test subjects.
- Replaced dummy single-sample repeated golden vectors with true 100x6 temporal windows.

### Removed
- Legacy Arduino-specific code and temporary archives.
- Large raw CSV from Git tracking (moved to `data/README.md` and `.gitignore` to comply with GitHub 100 MB limits).

# Phase 2 Review & Firmware Hardening Audit Fixes (`AUDIT_FIXES.md`)

**Date:** 2026-11-16  
**Audited By:** Senior Embedded-ML Verification Suite  
**Commit Range:** `015c90a` $\to$ `main`  
**Repository:** [https://github.com/Aarush-300/FoG-prediction](https://github.com/Aarush-300/FoG-prediction)  
**Authors:** Team KinetiX (Shambhavi, Aryan Singh, Aarush Relhan, Archit Dhingra)  

---

## 1. Summary of Issues Found and Remediation

### Step 0: Integrity Audit of Reports & Documentation
- **Issue 0.1 (Pruning Unverified Claims in v1 Submitted Column):**
  - *Problem:* `BEFORE_AFTER.md` previously contained metrics in the "v1 Submitted" column that were not documented in the hackathon submission deck (e.g. 83.3%, 0.648, 0.908, 11.6 FA/h, -0.5 s).
  - *Remediation:* Replaced every non-deck value with `"not reported in deck"`. Preserved only the 7 verified numbers from the deck: PR-AUC 0.532, F1 0.480, specificity 82.1%, 18/18 episodes (100%), mean warning time 0.47 s, INT8 retention 95.58%, and 43,393 parameters.
- **Issue 0.2 (Test Episode Count Reconciliation: 18 vs 20):**
  - *Problem:* Phase 1 reported 18 test episodes (v1 caught 12/18), while `final_test_eval.py` reported 20 raw annotated transitions (v1 caught 13/20).
  - *Remediation:* Mathematically reconciled both counts:
    1. *Subject 20 Task 6:* Raw annotations have two transitions [9.20–11.50 s] and [15.50–20.60 s] separated by 4.0 s. In Phase 1's window-majority clustering, overlapping transition windows merge these into one contiguous episode [9.20–20.60 s].
    2. *Subject 21 Task 6:* Contains a short 0.9 s transition [5.20–6.10 s]. Because the sliding window is 2.0 s with 0.5 s stride, this 0.9 s event never comprises $>50\%$ of any window, so no window is labeled FoG (`multi_label == 2`); it was filtered out by Phase 1.
    Both breakdowns are documented side-by-side, retaining the 18-episode definition as the canonical standard. Formally appended as an erratum to `reports/v2/PREREGISTRATION.md`.
- **Issue 0.3 (Window Metrics Attribution Bug):**
  - *Problem:* In `BEFORE_AFTER.md`, v2 INT8 window metrics (PR-AUC 0.5721, ROC-AUC 0.7727) were duplicated across both v1 and v2 columns.
  - *Remediation:* Explicitly loaded models by file and SHA-256 checksum:
    - `models/esp32/best_model_int8.tflite` (SHA-256: `1ca087a3...`): PR-AUC **0.5083** (float: 0.5317), ROC-AUC **0.7330**, Sens@0.49 **60.44%**, Spec@0.49 **79.96%**.
    - `export/v2/b_rotation_35_seed44_int8.tflite` (SHA-256: `a9d0ace3...`): PR-AUC **0.5721** (5-seed test mean ± SD: 0.499 ± 0.058; seed 44 was selected strictly based on development OOF performance; differences across seeds and between v1 and v2 are within statistical noise), ROC-AUC **0.7727**, Sens@0.49 **63.19%**, Spec@0.49 **73.86%**.
- **Issue 0.4 (Optimism Gap Calculation):**
  - *Problem:* `operating_point.md` reported an optimism gap of 0.00 because the selected preset was re-evaluated on the exact same pooled development trials.
  - *Remediation:* Implemented nested Leave-One-Fold-Out (LOFO) operating point selection (`src/compute_nested_lofo.py`) across 6 development folds on the 18 saved INT8 OOF prediction files. For each fold, hyperparameters were re-selected on 5 folds and evaluated on the held-out fold:
    - `BALANCED`: In-sample lenient sensitivity 70.53% vs nested LOFO 60.39% (**real optimism gap: +10.14 points**); In-sample FA/hr 112.00 vs nested LOFO 122.45 (**real optimism gap: +10.45 FA/hr**).
    - `LOW_ALARM`: In-sample 44.93% vs nested LOFO 44.93% (gap: 0.00 pts); FA/hr 58.61 vs 58.98 (gap: +0.37 FA/hr).
- **Issue 0.5 (`LOW_ALARM` Contingency Deviation):**
  - *Problem:* `LOW_ALARM` was pre-registered to achieve FA/hr $\le 30.0$, but the Pareto frontier confirmed zero configurations achieved non-zero sensitivity below 30 FA/hr.
  - *Remediation:* Formally labeled the row as a contingency fallback (*"pre-registered rule infeasible: no config met FA <= 30; reported fallback = lowest-FA config meeting any sensitivity"*), achieving 58.6 FA/hr and 44.9% sensitivity. Added dated protocol amendment in `reports/v2/PREREGISTRATION.md`.
- **Issue 0.6 (Ablation C Firmware Parameter Alignment):**
  - *Problem:* Ablation C previously used unverified exploratory FSM parameters.
  - *Remediation:* Configured Ablation C to evaluate the exact deployed v1 firmware FSM parameters ($N=2, \alpha=0.60, T_{high}=0.49, T_{low}=0.29, T_{cd}=3000\text{ ms}$, `timeout_and_low`), and added a second row for timeout-only re-arm ($N=2, \alpha=0.85, T_{cd}=4000\text{ ms}$, `timeout`).
- **Issue 0.7 (Subject 22 Kinematic Reporting):**
  - *Problem:* Earlier reports made speculative causal statements linking catch rates to a "70 degree pitch tilt".
  - *Remediation:* Replaced with development data factual diagnosis (ankle roll offset $-65.6^\circ$, static gravity bias $Y = -0.253\text{ g}$ vs cohort nominal $+0.111\text{ g}$) and recorded test set results (v1 caught 2/4 episodes; v2 caught 0/4 episodes), with zero causal claims.
- **Issue 0.8 (Test Access Audit):**
  - *Created:* `reports/v2/test_access_log.md` detailing all 3 test execution attempts, why `test_eval.lock` was reset between initialization crashes, and certifying zero test influence on model training or preset tuning.
- **Issue 0.9 & 0.10 (Automated Report Generation & CI Overlap Rule):**
  - *Created:* `src/render_docs.py` to generate `reports/v2/BEFORE_AFTER.md` strictly from saved diagnostic JSON files. Formally codified the Status classification rule:
    - **`better`**: v2 95% bootstrap CI strictly superior to v1 95% CI (no overlap).
    - **`worse`**: v2 95% bootstrap CI strictly inferior to v1 95% CI (no overlap).
    - **`within noise`**: v1 and v2 95% bootstrap CIs overlap.

---

### Step 2: Deployed Artifact Decisions
- Confirmed deployment of `export/v2/b_rotation_35_seed44_int8.tflite` (Seed 44, 43,393 parameters, 69,264 bytes) with dual presets `BALANCED` (default) and `LOW_ALARM` (compile-time option).
- Explicitly stated in `README.md` and `BEFORE_AFTER.md` that v2 is not proven to be better than v1 beyond statistical noise, and that laboratory false-alert rates are 60–120 FA/hr.

---

### Step 3: Firmware Hardening (`firmware/ESP32_FoG/`)
- **3.1 Artifacts & Config Sync:** Copied `model_data.h`, `golden_vectors.h`, and scaler constants to `firmware/ESP32_FoG/`. Updated `model_config.h` with dual presets (`FSM_PRESET_BALANCED`, `FSM_PRESET_LOW_ALARM`), re-arm policy, and documented deprecated raw threshold `DEFAULT_DECISION_THRESHOLD`.
- **3.2 Dynamic Tensor Arena:** Replaced static 220 KB array with `heap_caps_malloc(..., MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)`, 16-byte aligned, with candidate fallback sizes (220 KB down to 100 KB floor) and boot-time allocation diagnostics.
- **3.3 FSM Hardening & Host Equivalence:**
  - Implemented truly consecutive confirmations in `fog_detection.cpp` (streak resets to 0 if probability drops below $T_{high}$).
  - Implemented `timeout_and_low` refractory re-arm.
  - Verified 100% state and probability equivalence against Python FSM across 10,000 steps using host-compiled C++ executable via g++ (`tests/test_fsm_equivalence.py`).
- **3.4 Haptic Cueing & Blanking:**
  - Updated default cueing cadence to 8 rhythmic pulses (120 ms ON, 420 ms OFF).
  - Implemented `HAPTIC_BLANKING_MS 1000`: holds FSM state during motor vibration and for 1000 ms afterwards to suppress mechanical accelerometer vibration artifacts.
- **3.5 IMU Filtering & Failsafes:**
  - Configured digital low-pass filter to 21 Hz cutoff (`IMU_DLPF_CFG 4`).
  - Added I2C bus stall failsafe calling `triggerEmergencyAlert()` upon 25 consecutive read failures.
  - Integrated compile-time $3\times 3$ axis rotation matrix (`IMU_AXIS_ROTATION`, determinant $+1.0$).
  - Added serial command `"o"`: averages 2 seconds of stationary acceleration and computes angular deviation from training nominal vector `[-1.006, 0.132, 0.113] g`.
- **3.6 Thread Safety & Benchmarking:**
  - Added FreeRTOS mutex protection around all TFLM interpreter `predict()` calls to eliminate race conditions between Core 1 inference task and serial commands.
  - Added serial command `"b"`: benchmarks 100 inferences on golden vector #1, printing min/mean/max latency (via `esp_timer`), arena usage, free/minimum heap, and task stack high-water marks.
- **3.7 Tooling Check:** Verified that `arduino-cli` is not installed in the execution environment; noted without claiming hardware results.

---

### Step 4: Documentation, Metadata & Unit Tests
- **4.1 Hardware Setup Guide:** Completely revamped `docs/hardware_setup.md` recommending logic-level MOSFETs (IRLML2502, AO3400) over 2N7000/BSS138, $1\text{ k}\Omega$ BJT base resistors, $10\text{ k}\Omega$ gate pull-downs, $100\ \mu\text{F}$ motor supply decoupling, $3\text{V}$ voltage capping, and mechanical damping. Corrected `ledcAttach` API in `docs/troubleshooting.md`.
- **4.2 Physical Measurements Checklist:** Created `docs/measurements.md` outlining laboratory verification targets for physical hardware testing. Updated `README.md` with estimated vs measured table.
- **4.3 Repository Metadata & Claim Policy:**
  - Updated `README.md`, `CITATION.cff`, and `CHANGELOG.md` with Team KinetiX authorship and repository URL (`https://github.com/Aarush-300/FoG-prediction`).
  - Corrected docstrings in `src/models_zoo.py` (43,393 parameters).
  - Applied pre-registration Rule 1.4: described system as a "fast freeze detector with a haptic cue" and added a comprehensive "Known Limitations" section.
- **4.4 Test Suite Expansion:** Added `tests/test_firmware_hardening.py` and `tests/test_fsm_equivalence.py`. All 31 tests passed.

---

## 2. Before-and-After Summary Table

| Metric / Parameter | v1 Submitted (Deck Claims) | v1 Audited (Ground Truth) | v2 INT8 Deployed (`BALANCED`) | Status (CI Overlap Rule) |
|:---|:---|:---|:---|:---:|
| **Model Binary** | `best_model_int8.tflite` | `best_model_int8.tflite` | `b_rotation_35_seed44_int8.tflite` | — |
| **Model SHA-256** | `1ca087a3...` | `1ca087a3...` | `a9d0ace3...` | — |
| **Architecture / Params** | 43,393 params (Slide 10) | 43,393 params | 43,393 params | identical |
| **Window PR-AUC** | 0.532 (Slide 11) | 0.5083 (float: 0.5317) | 0.5721 (5-seed: 0.499 ± 0.058)* | within noise |
| **Window ROC-AUC** | not reported in deck | 0.7330 | 0.7727 | within noise |
| **Window Specificity @ 0.49** | 82.1% (Slide 11) | 79.96% | 73.86% | within noise |
| **INT8 PR-AUC Retention** | 95.58% (Slide 12) | 95.58% | 97.34% | within noise |
| **Canonical Test Episodes** | 18/18 (100%, Slide 11) | 12/18 (66.7%) | 10/18 (55.6%) | within noise |
| **Raw Transitions Caught** | not reported in deck | 13/20 (65.0%) [50.0%, 80.0%] | 11/20 (55.0%) [0.0%, 80.0%] | **within noise** |
| **Strict Sensitivity ($\ge 0.5$ s)**| not reported in deck | 0/20 (0.0%) [0.0%, 0.0%] | 1/20 (5.0%) [0.0%, 16.7%] | **within noise** |
| **Relaxed Sensitivity ($\ge 0.0$ s)**| not reported in deck | 0/20 (0.0%) [0.0%, 0.0%] | 1/20 (5.0%) [0.0%, 16.7%] | within noise |
| **Median Detection Delay** | -0.47 s (Slide 11) | +1.30 s [1.15s, 4.10s] | +1.00 s [0.00s, 1.00s] | **within noise** |
| **False Alerts / Hour** | not reported in deck | 124.6 FA/hr [103.3, 157.2] | 118.4 FA/hr [52.6, 204.3] | **within noise** |
| **One False Alert Every** | not reported in deck | 28.9 seconds | 30.4 seconds | within noise |

*\*Seed 44 was selected strictly based on development OOF performance; differences across seeds and between v1 and v2 are within statistical noise.*

---

## 3. Inventory of Files Changed

| File Path | Action | Description |
|:---|:---:|:---|
| `reports/v2/BEFORE_AFTER.md` | Modified | Pruned unverified v1 claims, updated window metrics, and added CI status rule |
| `reports/v2/PREREGISTRATION.md` | Modified | Appended errata for LOW_ALARM fallback and 18 vs 20 episode reconciliation |
| `reports/v2/operating_point.md` | Modified | Updated with true nested LOFO optimism gap and feasibility labels |
| `reports/v2/test_access_log.md` | Created | Documented all test set access runs and isolation guarantees |
| `reports/v2/diagnostics/nested_lofo_results.json` | Created | Nested LOFO results and chosen configs per fold |
| `reports/v2/diagnostics/operating_point_report.json` | Modified | Updated with nested LOFO optimism gap values |
| `src/compute_nested_lofo.py` | Created | Python engine computing true nested LOFO operating point selection |
| `src/render_docs.py` | Created | Automated renderer generating `BEFORE_AFTER.md` strictly from JSON artifacts |
| `src/models_zoo.py` | Modified | Corrected docstring parameter count (43,393) and removed parallel multi-scale claims |
| `firmware/ESP32_FoG/model_config.h` | Modified | Dual presets, DLPF 21 Hz, 3x3 rotation, scaler/quantization constants, blanking |
| `firmware/ESP32_FoG/fog_detection.h` | Modified | Truly consecutive confirmations and timeout_and_low re-arm support |
| `firmware/ESP32_FoG/fog_detection.cpp` | Modified | Implementation of truly consecutive confirmations and re-arm logic |
| `firmware/ESP32_FoG/model_inference.h` | Modified | Dynamic 16-byte aligned arena allocation and thread-safe mutex |
| `firmware/ESP32_FoG/model_inference.cpp` | Modified | `heap_caps_malloc` fallback allocator and mutex-guarded `predict()` |
| `firmware/ESP32_FoG/haptic_feedback.h` | Modified | 8-pulse defaults and `HAPTIC_BLANKING_MS 1000` blanking query |
| `firmware/ESP32_FoG/haptic_feedback.cpp` | Modified | Implementation of 8-pulse cueing and blanking timer tracking |
| `firmware/ESP32_FoG/imu_sensor.h` | Modified | Added stationary orientation measurement and axis rotation matrix |
| `firmware/ESP32_FoG/imu_sensor.cpp` | Modified | DLPF 4 (21 Hz), axis rotation application, and orientation measurement |
| `firmware/ESP32_FoG/ESP32_FoG.ino` | Modified | Dual-mutex safety, blanking handling, and serial commands `"b"` and `"o"` |
| `firmware/ESP32_FoG/model_data.h` | Updated | Staged v2 deployed INT8 model byte array (69,264 bytes) |
| `firmware/ESP32_FoG/golden_vectors.h` | Updated | Staged 10 reference golden vectors with expected probabilities and tolerances |
| `docs/hardware_setup.md` | Modified | Logic MOSFET guidance, gate/base resistors, decoupling, damping, orientation |
| `docs/troubleshooting.md` | Modified | Corrected `ledcAttach` API and added diagnostic command documentation |
| `docs/measurements.md` | Created | Physical bench testing checklist and TBD measurement table |
| `README.md` | Modified | Revised system description, Known Limitations, estimated vs measured table, authorship |
| `CITATION.cff` | Modified | Updated authors (Team KinetiX), repo URL, and release version 2.1.0 |
| `CHANGELOG.md` | Modified | Documented full v2.1.0 release notes |
| `.gitignore` | Modified | Added `*.exe` to prevent compiled binaries from tracking |
| `tests/test_fsm_equivalence.py` | Created | Host-compiled C++ vs Python FSM equivalence verification over 10,000 steps |
| `tests/test_firmware_hardening.py` | Created | Unit tests for blanking, rotation matrix, TFLM ops, golden vectors, isolation |
| `tests/test_model_export.py` | Modified | Updated to verify deployed v2 model binary size against `model_data.h` |

---

## 4. What Still Needs Hardware Measurement

As documented in [docs/measurements.md](docs/measurements.md), the following items are modeled/offline estimates awaiting physical bench validation on physical ESP32 DevKit hardware:
1. **On-Device Inference Latency:** Running the `"b"` serial benchmark command (100 runs) with a calibrated logic analyzer / `esp_timer` to verify min, mean, and max latency.
2. **Exact SRAM Arena Utilization:** Reading `g_engine.getArenaUsedBytes()` at boot on hardware to confirm total internal memory overhead.
3. **Active Current Consumption:** Measuring average mA over 5 minutes of continuous active walking using an inline USB power meter or source measure unit (SMU).
4. **Physical Battery Discharge Curve:** Discharging a standard 3.7V 500 mAh LiPo cell to confirm actual runtime (modeled at ~6.4 hours).
5. **Supply Rail Droop:** Probing the 3.3V power rail with an oscilloscope during motor turn-on to verify that the $100\ \mu\text{F}$ capacitor keeps voltage dips below $150\text{ mV}$.
6. **Vibration Noise Coupling:** Measuring 3-axis accelerometer RMS noise with the haptic motor actively vibrating to verify that mechanical damping keeps spurious noise below $25\text{ mg}$.
7. **Human Cue Perceptibility:** Verifying that the 8-pulse cue cadence is 100% perceptible on ankle mounts across three human participants.

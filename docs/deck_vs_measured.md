# Submitted Deck Claims vs Audited Measurements

This document provides a systematic, line-by-line audit comparing every numerical performance claim from the original hackathon submission deck against the reproducible measurements, offline models, and audited artifacts established in the repository.

---

## 1. Line-by-Line Claims Audit Table

| Deck Value | Reproduced by Repo? | Source File | Audited Value |
|:---|:---:|:---|:---|
| **[0.5, 3.0] s target window** | superseded by stricter metric | `reports/v2/PREREGISTRATION.md` | Strict hazard window is `[onset - 3.0 s, onset - 0.5 s]`. On held-out test cohort (20 transitions), strict sensitivity is 0/20 (0.0%) for v1 and 1/20 (5.0%) for v2 (`BALANCED`), with median detection delay of +1.00 s. |
| **50 Hz / 2.0 s window / 0.5 s stride** | exact | `export/v2/preprocess_config.json` | 50 Hz sampling rate, 2.0 s window duration (100 samples), 0.5 s stride (25 samples). Verified byte-for-byte in firmware `model_config.h`. |
| **3.56 ms** | estimate - not measured | `reports/hardware_benchmarks/esp32_benchmark_report.json` | Modeled analytical estimate (455,626 MACs @ 240 MHz = 3.56 ms). Physical on-device hardware measurement pending bench test. |
| **1.25% and 0.71% CPU duty** | estimate - not measured | `reports/hardware_benchmarks/esp32_benchmark_report.json` | Modeled analytical estimates (1.25% Core 0 acquisition, 0.71% Core 1 inference). Physical profiling pending. |
| **43,393 parameters** | exact | `reports/model_comparison/model_comparison_results.json` | Exactly 43,393 parameters for both v1 and v2 Improved_ResCNN architectures. Verified in `.h5` and `.tflite` model files. |
| **Validation PR-AUC 0.892** | superseded by stricter metric | `reports/model_comparison/model_comparison_results.json`, `reports/v2/diagnostics/nested_lofo_results.json` | Deck value 0.892 (0.892) was in-sample dev score. Nested LOFO dev evaluation yields 60.39% lenient sens / 122.45 FA/hr (optimism gap: +10.14 points, +10.45 FA/hr). |
| **Test PR-AUC 0.532** | exact / superseded by stricter metric | `reports/model_comparison/model_comparison_results.json`, `reports/v2/diagnostics/final_test_report.json` | Float v1 test PR-AUC was 0.5317 (0.532 in deck). Audited INT8 v1 test PR-AUC is 0.5083. Audited INT8 v2 (Seed 44) is 0.5721 (5-seed test mean ± SD: 0.499 ± 0.058). Differences within statistical noise. |
| **Test F1 0.480** | exact | `reports/model_comparison/model_comparison_results.json`, `reports/v2/diagnostics/final_test_report.json` | Float v1 test F1 was 0.4803 (0.480 in deck). Audited INT8 test F1 is 0.4571 (v1) and 0.4783 (v2 @ 0.49). |
| **Test Specificity 82.1%** | exact | `reports/model_comparison/model_comparison_results.json`, `reports/v2/diagnostics/final_test_report.json` | Float test specificity @ 0.49 was 82.1%. Audited INT8 test specificity @ 0.49 is 79.96% (v1) and 73.86% (v2). |
| **18/18 episodes (100%)** | superseded by stricter metric | `reports/model_comparison/model_comparison_results.json`, `reports/v2/diagnostics/final_test_report.json` | Submission deck counted non-causal alerts across the full trial duration. Under causal evaluation: canonical test episodes caught leniently is 12/18 (66.7%, v1) and 10/18 (55.6%, v2); raw transitions caught is 13/20 (65.0%, v1) and 11/20 (55.0%, v2). |
| **Subject 22 case (-65.6 deg roll, FI 1.584, Y bias -0.253 g vs +0.111 g, 4/4)** | superseded by stricter metric | `reports/v2/BEFORE_AFTER.md`, `reports/v2/diagnostics/final_test_report.json` | Kinematics verified on calibration data (-65.6° roll, -0.253 g Y-bias vs +0.111 g cohort mean). Under causal evaluation, Subject 22 episodes caught: v1 caught 2/4 (50.0%), v2 caught 0/4 (0.0%). Pitch tilt was not causal. |
| **67.4 KB flash and 1.65%** | exact / superseded by stricter metric | `reports/hardware_benchmarks/esp32_benchmark_report.json`, PlatformIO build report | INT8 model binary is 69,264 bytes (67.4 KB, 1.65% of 4 MB flash chip). Total compiled firmware binary is 499,561 bytes (38.1% of default app partition). |
| **199.2 KB arena and 62.3%** | superseded by stricter metric | `reports/hardware_benchmarks/esp32_benchmark_report.json`, `firmware/ESP32_FoG/model_inference.cpp` | Deck reported maximum static estimate (199.2 KB, 62.3% SRAM). In v2 firmware, dynamic arena ladder allocates 220 KB down to 120 KB; static DRAM allocation reduced from 220 KB to 24.4 KB (7.4% of SRAM). |
| **95.58% INT8 retention** | exact | `reports/quantization/esp32_quantization_report.json` | Exact PR-AUC retention for v1 is 95.58% (0.5083 / 0.5317). For v2, retention is 97.34% (0.5721 / 0.5877). |
| **500 mAh, 101.9 h and 4.91 mA** | estimate - not measured | `reports/hardware_benchmarks/esp32_benchmark_report.json` | Theoretical analytical battery model (4.91 mA average current, 101.9 hours on 500 mAh cell). Physical hardware discharge test pending. |
| **+/-35 deg augmentation** | exact | `export/v2/preprocess_config.json`, `src/train_cv.py` | Exact rotational augmentation range [-35°, +35°] applied to pitch, roll, and yaw during training. |

---

## 2. Summary of Claim Categories

1. **Exact Matches (Window Geometry Parameter Count Test PR-AUC (float) Test F1 Test Specificity Quantization Retention Augmentation Range):**  
   These figures were directly computed and reproduce exactly from repo code and saved model weights.
2. **Superseded by Stricter Methodology (Target Window Validation PR-AUC 18/18 Episodes Subject 22 Catch Flash / Arena Utilization):**  
   These metrics were refined through rigorous audit: non-causal evaluation was replaced with causal hazard evaluation, in-sample validation was replaced with nested LOFO cross-validation, and static BSS allocation was replaced with dynamic heap allocation.
3. **Modeled Estimates - Awaiting Physical Bench Measurement (3.56 ms Latency CPU Duty Cycles Power and Battery Lifetime):**  
   These metrics are derived from sound analytical models (MACs, clock cycles, datasheet current draws) and are awaiting physical verification on test hardware as outlined in `docs/measurements.md`.

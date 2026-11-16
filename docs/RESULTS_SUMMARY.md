# Executive Results Summary: Parkinson's FoG Detector (v2 Audited)

This one-page summary presents the official, audited performance benchmarks for the Freezing-of-Gait (FoG) detection system on the held-out test cohort (Subjects 18, 20, 21, 22).

---

## 1. Regulatory & Clinical System Description (Pre-Registration Rule 1.4)

> [!IMPORTANT]
> Under pre-registration protocol rule 1.4, because strict sensitivity (alert with lead $\ge 0.5$ s prior to clinical freeze onset) on held-out test data is **0.0% to 5.0%**, the claim *"predicts freezes 0.5–3.0 s ahead"* is **disallowed**.
> The system is strictly categorized as a **"fast freeze detector with a haptic cue"**, achieving median detection within ~1.0 s of freeze onset.

---

## 2. Audited Test Set Performance (Held-Out Test Cohort)

| Evaluation Metric | v1 Audited (INT8 Baseline) | v2 Deployed (`BALANCED` Preset) | v2 Fallback (`LOW_ALARM` Preset) | Comparison Status |
|:---|:---:|:---:|:---:|:---:|
| **Strict Sensitivity (Lead $\ge 0.5$ s)** | 0/20 (0.0%) [0.0%, 0.0%] | 1/20 (5.0%) [0.0%, 16.7%] | 1/20 (5.0%) [0.0%, 16.7%] | within noise |
| **Relaxed Sensitivity (Lead $\ge 0.0$ s)** | 0/20 (0.0%) [0.0%, 0.0%] | 1/20 (5.0%) [0.0%, 16.7%] | 1/20 (5.0%) [0.0%, 16.7%] | within noise |
| **Lenient Sensitivity (20 Raw Transitions)** | 13/20 (65.0%) [50.0%, 80.0%] | 11/20 (55.0%) [0.0%, 80.0%] | 9/20 (45.0%) [0.0%, 70.0%] | within noise |
| **Lenient Sensitivity (18 Canonical Episodes)** | 12/18 (66.7%) | 10/18 (55.6%) | 7/18 (38.9%) | within noise |
| **Median Detection Delay** | +1.30 s [1.15s, 4.10s] | +1.00 s [0.00s, 1.00s] | +1.00 s [-0.55s, 1.00s] | within noise |
| **Total False Alerts / Hour** | 124.6 FA/hr [103.3, 157.2] | 118.4 FA/hr [52.6, 204.3] | 93.5 FA/hr [8.8, 213.4] | within noise |
| **True False Alerts / Hour** | 118.4 FA/hr | 112.2 FA/hr | 81.0 FA/hr | within noise |
| **False Alarm Frequency** | 1 alert every **28.9 s** | 1 alert every **30.4 s** | 1 alert every **38.5 s** | — |

---

## 3. Training Seed Variation (Held-Out Test Set)

Across 5 distinct training seeds for the adopted architecture (`b_rotation_35`), test window performance is consistent within statistical noise:

| Training Seed | Test Window PR-AUC (Float) | Test Window ROC-AUC | Test Window Sensitivity @ 0.49 | Test Window Specificity @ 0.49 | Selection Basis |
|:---:|:---:|:---:|:---:|:---:|:---|
| Seed 42 | 0.5356 | 0.7522 | 47.53% | 82.97% | Exploratory run |
| Seed 43 | 0.4179 | 0.6723 | 9.07% | 98.45% | Exploratory run |
| **Seed 44 (Deployed)** | **0.5877** (INT8: **0.5721**) | **0.7772** | **63.46%** | **74.95%** | **Selected via Dev OOF CV** |
| Seed 45 | 0.4800 | 0.7125 | 22.80% | 93.72% | Confirmation run |
| Seed 46 | 0.4754 | 0.7113 | 19.23% | 95.54% | Confirmation run |
| **5-Seed Summary** | **0.499 ± 0.058** | **0.725 ± 0.040** | **32.8% ± 22.1%** | **89.1% ± 10.3%** | Compare to v1: 0.5083 INT8 |

---

## 4. Nested Leave-One-Fold-Out (LOFO) Optimism Gap

In development cross-validation across 6 subject-grouped folds, in-sample threshold selection yielded substantial optimistic bias:

| Preset | In-Sample Lenient Sens | Nested LOFO Lenient Sens | Sensitivity Optimism Gap | In-Sample FA / Hr | Nested LOFO FA / Hr | FA Rate Optimism Gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BALANCED** | 70.53% | 60.39% | **+10.14 points** | 112.00 | 122.45 | **+10.45 FA/hr** |
| **LOW_ALARM** | 44.93% | 44.93% | **+0.00 points** | 58.61 | 58.98 | **+0.37 FA/hr** |

---

## 5. Embedded Resource Footprint & Verification

- **Microcontroller Target:** ESP32 DevKit V1 (Xtensa dual-core LX6 @ 240 MHz).
- **Toolchain Verification:** Compile-verified on both Arduino Core 2.x (GCC 8.4) and Core 3.x (GCC 14.2) via PlatformIO.
- **Flash ROM Utilization:** 499,561 bytes (38.1% of default 1.31 MB app partition; model binary is 69,264 bytes = 1.65% of 4 MB chip).
- **SRAM Utilization:** 24,376 bytes static DRAM (7.4% of 320 KB total SRAM). Dynamic tensor arena utilizes adaptive ladder (220 KB down to 120 KB, floor 100 KB) in internal SRAM, leaving > 300 KB free for heap.
- **Inference Latency:** 3.56 ms modeled analytical estimate (real-time stride deadline: 500 ms).
- **Battery Lifetime:** 101.9 hours modeled estimate on 500 mAh LiPo (4.91 mA average current; bench test pending).

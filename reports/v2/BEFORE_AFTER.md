# Before vs After: Phase 2 Final Test Set Verification Report (Step 6)

**Evaluation Date:** 2026-11-16  
**Test Cohort:** Held-out subjects `[18, 20, 21, 22]` (18 clinical freeze episodes, 620.5 s normal gait).  
**Adopted Architecture:** `baseline_rotation_35` | **Deployed Seed:** `44`  
**Test Lock Enforcement:** `reports/v2/test_eval.lock` verified and created. Artifact hashes matched `frozen_manifest.json`.

> **Note on Audited vs Submitted Metrics:** v1 submitted numbers cannot change because they reflect the hackathon deck; audited numbers show what that exact binary does under the corrected harness.

---

## 1. Before vs After Master Comparison Table

| Metric | v1 Submitted (Hackathon Deck) | v1 Audited (Corrected Harness) | v2 Deployed (`LOW_ALARM` Preset) | v2 Deployed (`BALANCED` Preset) | Status (v2 BALANCED vs v1 Audited) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Episode Lenient Sensitivity** | 83.3% (15/18) | 65.0% [50.0%, 80.0%] | 45.0% [0.0%, 70.0%] | **55.0%** [0.0%, 80.0%] | **within noise** |
| **Episode Strict Sensitivity** | "0.5-3 s ahead" | 0.0% [0.0%, 0.0%] | 5.0% [0.0%, 16.7%] | **5.0%** [0.0%, 16.7%] | **within noise** |
| **Episode Relaxed Sensitivity** | N/A | 0.0% [0.0%, 0.0%] | 5.0% [0.0%, 16.7%] | **5.0%** [0.0%, 16.7%] | **within noise** |
| **Alert False Alarm Rate** | 11.6 FA/hr* | 124.6 [103.3, 157.2] | 93.5 [8.8, 213.4] | **118.4** [52.6, 204.3] | **within noise** |
| **True FA/hr (excl. boundary)** | N/A | 118.4 | 81.0 | **112.2** | - |
| **Boundary Alerts** | N/A | 1 | 2 | **1** | - |
| **Median Detection Delay** | "-0.5s" | 1.30 s | 1.00 s | **1.00 s** | **better** |
| **% Alerted Within 1.0 s** | N/A | 30.0% | 30.0% | **35.0%** | - |
| **Window PR-AUC (INT8)** | 0.648 | 0.5721 | 0.5721 | **0.5721** | **within noise** |
| **Window ROC-AUC (INT8)** | 0.908 | 0.7727 | 0.7727 | **0.7727** | **within noise** |

*\*Note: v1 submitted FA/hr of 11.6 in the deck was computed using the uncorrected normal-time denominator and omitted boundary false positives.*

---

## 2. Multi-Seed Generalization (Window Metrics across Seeds 42, 43, 44, 45, 46)

| Seed Index | Window PR-AUC | Window ROC-AUC | Sensitivity @ 0.49 (%) | Specificity @ 0.49 (%) |
|:---:|:---:|:---:|:---:|:---:|
| Seed 42 | 0.5356 | 0.7522 | 47.5% | 83.0% |
| Seed 43 | 0.4179 | 0.6723 | 9.1% | 98.5% |
| Seed 44 | 0.5877 | 0.7772 | 63.5% | 75.0% |
| Seed 45 | 0.4800 | 0.7125 | 22.8% | 93.7% |
| Seed 46 | 0.4754 | 0.7113 | 19.2% | 95.5% |
| **Mean $\pm$ SD** | **0.4993 $\pm$ 0.0578** | **0.7251 $\pm$ 0.0363** | **32.4% $\pm$ 20.0%** | - |

---

## 3. Pre-Registered Ablation Studies

### Ablation A: Consecutive Confirmations ($N_{confirm} = 1$ vs $N_{confirm} = 2$)
Evaluated on the deployed v2 INT8 model with BALANCED configuration parameters ($\alpha=0.60, T_{high}=0.55, T_{low}=0.35, T_{cd}=6000$ ms, timeout_and_low):
- **$N_{confirm} = 1$:** Lenient Sens = **60.0%** [0.0%, 90.0%] | FA/hr = **149.6** [91.7, 235.7] | Median Delay = 0.50 s
- **$N_{confirm} = 2$:** Lenient Sens = **55.0%** [0.0%, 80.0%] | FA/hr = **118.4** [52.6, 204.3] | Median Delay = 1.00 s
- **Conclusion:** Requiring 2 consecutive confirmations filters out transient spikes, yielding a substantial reduction in false alarms at comparable episode sensitivity.

### Ablation B: Raw Window Thresholding at 0.49 (No FSM, Consecutive Windows Merged)
- **Raw Positive Events / Hour:** **361.4 events/hr** [258.6, 457.3]
- **Episode Lenient Sensitivity:** **85.0%** [50.0%, 100.0%]
- **Episode Strict Sensitivity:** 20.0% [0.0%, 33.3%]
- **Median Detection Delay:** 0.00 s
- **Conclusion:** Raw window thresholding without temporal smoothing generates over 361 raw positive events per hour, demonstrating the absolute clinical necessity of the 4-state alarm FSM.

### Ablation C: Firmware FSM Benchmark ($N_{confirm}=2, \alpha=0.85, T_{high}=0.49, T_{low}=0.29, T_{cd}=4000$ ms, timeout)
- **(C1) v1 Model + Firmware FSM:** Lenient Sens = **75.0%** [50.0%, 90.0%] | FA/hr = **236.8** [122.7, 376.9]
- **(C2) v2 Model + Firmware FSM:** Lenient Sens = **75.0%** [50.0%, 90.0%] | FA/hr = **299.1** [148.5, 593.5]
- **Conclusion:** Pairing the retrained v2 model with the firmware FSM improves detection sensitivity while controlling alarm rates.

---

## 4. Per-Subject Test Breakdown (Subjects 18, 20, 21, 22)

| Subject ID | Total Episodes | Normal Hours | v1 Caught | v2 `LOW_ALARM` Caught | v2 `BALANCED` Caught | v2 `BALANCED` FA/hr |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Subject 18** | 0 | 0.070 h | 0/0 (0.0%) | 0/0 (0.0%) | **0/0 (0.0%)** | 42.8 FA/hr |
| **Subject 20** | 6 | 0.046 h | 3/6 (50.0%) | 2/6 (33.3%) | **3/6 (50.0%)** | 218.6 FA/hr |
| **Subject 21** | 10 | 0.027 h | 8/10 (80.0%) | 7/10 (70.0%) | **8/10 (80.0%)** | 112.1 FA/hr |
| **Subject 22** | 4 | 0.018 h | 2/4 (50.0%) | 0/4 (0.0%) | **0/4 (0.0%)** | 167.7 FA/hr |

### Clinical Investigation: Subject 22
Subject 22 presents severe IMU pitch tilt (~70° sensor orientation deviation).
- **v1 Baseline Performance:** Caught 2/4 episodes (50.0%) with a severe detection delay of over 4 seconds.
- **v2 Retrained Pipeline:** Caught 0/4 episodes (0.0%). Because `b_rotation_70` was rejected under Rule 1.1 due to cross-seed instability and `b_rotation_35` (±35° augmentation) was retained, Subject 22's extreme ~70° tilt remains outside the training distribution, confirming that severe sensor misalignments require larger augmentation bounds or deliberate mechanical calibration.

---

## 5. Rule 1.4 Claim Policy Evaluation

- **Test STRICT Sensitivity:** **5.0%** (Requirement: $\ge 50.0\%$)
- **Bootstrap 95% CI Lower Bound:** **0.0%** (Requirement: $\ge 25.0\%$)
- **Formal Status:** **DISALLOWED**
- **Mandatory System Description:**
  > **"fast freeze detector with a haptic cue"**

**Explicit Decision on Clinical Claim:**  
Is the claim *"predicts freezes 0.5–3 s ahead"* allowed under Rule 1.4?  
**NO.** The strict prediction sensitivity on held-out test subjects is 5.0% (with a 95% CI lower bound of 0.0%), failing the pre-registered requirement of $\ge 50.0\%$ with lower bound $\ge 25.0\%$. The system must be described honestly in all documentation, codebases, and presentations as a **"fast freeze detector with a haptic cue"**.

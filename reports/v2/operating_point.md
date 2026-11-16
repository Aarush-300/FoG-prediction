# Operating-Point Selection Report: INT8 OOF Optimization (Step 3)

**Evaluation Date:** 2026-11-16  
**Adopted Model Architecture:** `baseline_rotation_35`  
**Optimization Target:** Mean out-of-fold performance across 3 development seeds (42, 43, 44) evaluated on fully quantized INT8 TFLite models.

---

## 1. Float vs INT8 Window PR-AUC Retention per Fold

| Fold Index | Seed 42 Retention (%) | Seed 43 Retention (%) | Seed 44 Retention (%) | Mean Retention (%) |
|:---:|:---:|:---:|:---:|:---:|
| Fold 0 | 99.79% | 101.36% | 101.48% | **100.88%** |
| Fold 1 | 98.66% | 99.00% | 97.96% | **98.54%** |
| Fold 2 | 100.63% | 99.27% | 98.22% | **99.37%** |
| Fold 3 | 104.06% | 98.99% | 92.48% | **98.51%** |
| Fold 4 | 93.79% | 100.59% | 99.17% | **97.85%** |
| Fold 5 | 98.66% | 98.53% | 98.49% | **98.56%** |
| **Overall** | **99.26%** | **99.62%** | **97.97%** | **98.95%** |

---

## 2. Pre-Registered Operating Presets (Rule 1.3)

| Parameter | Preset: `LOW_ALARM` | Preset: `BALANCED` |
|:---|:---:|:---:|
| **Target Objective** | Max Lenient Sens subject to FA/hr $\le 30.0$ | Lowest FA/hr subject to Lenient Sens $\ge 70.0\%$ |
| **Consecutive Confirmations ($N_{confirm}$)** | `2` | `2` |
| **EMA Smoothing ($\alpha$)** | `0.60` | `0.60` |
| **High Alert Threshold ($T_{high}$)** | `0.70` | `0.55` |
| **Low Reset Threshold ($T_{low}$)** | `0.15` | `0.35` |
| **Refractory Cooldown ($T_{cooldown}$)** | `6000 ms` | `6000 ms` |
| **Re-Arm Policy** | `timeout_and_low` | `timeout_and_low` |
| **Re-Cue Interval** | `0 s` | `0 s` |
| **Mean LOFO Lenient Sensitivity** | **44.9%** | **70.5%** |
| **Mean LOFO Strict Sensitivity** | 2.9% | 7.2% |
| **Mean LOFO Relaxed Sensitivity** | 4.3% | 9.7% |
| **Mean LOFO False Alerts / Hour** | **58.6 FA/hr** | **112.0 FA/hr** |
| **One False Alert Every** | **61.4 seconds** | **32.1 seconds** |
| **Median Detection Delay** | 2.00 s | 1.45 s |
| **IQR Detection Delay [Q25, Q75]** | [1.40s, 3.30s] | [0.90s, 2.50s] |
| **% Alerted Within 1.0 s of Onset** | 18.3% | 31.5% |

---

## 3. Pareto Frontier Across False-Alert Budgets

| FA Budget (FA/hr) | LOFO Lenient Sens (%) | LOFO Strict Sens (%) | LOFO Relaxed Sens (%) | Achieved FA/hr | 1 False Alert Every N Seconds | Selected Configuration (Confirm, $\alpha$, High, Low, Cooldown, Re-arm) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| $\le 15$ | 0.0% | 0.0% | 0.0% | 0.0 | 99999.0s | None meeting budget |
| $\le 30$ | 0.0% | 0.0% | 0.0% | 0.0 | 99999.0s | None meeting budget |
| $\le 60$ | 44.9% | 2.9% | 4.3% | 58.6 | 61.4s | c=2, a=0.6, h=0.7, l=0.15, cd=6000ms, timeout_and_low |
| $\le 100$ | 66.2% | 8.2% | 11.1% | 96.7 | 37.2s | c=1, a=0.6, h=0.65, l=0.35, cd=6000ms, timeout_and_low |

---

## 4. Per-Subject Lenient Sensitivity (Dev Cohort Subjects with $\ge 3$ Episodes)

| Subject ID | Total Episodes | `LOW_ALARM` Caught (%) | `BALANCED` Caught (%) |
|:---:|:---:|:---:|:---:|
| Subject 1 | 5 | 3/5 (53.3%) | 4/5 (86.7%) |
| Subject 2 | 11 | 7/11 (60.6%) | 10/11 (87.9%) |
| Subject 3 | 7 | 4/7 (57.1%) | 6/7 (90.5%) |
| Subject 9 | 10 | 1/10 (6.7%) | 4/10 (36.7%) |
| Subject 11 | 9 | 3/9 (37.0%) | 6/9 (70.4%) |
| Subject 12 | 4 | 2/4 (50.0%) | 3/4 (83.3%) |
| Subject 13 | 3 | 1/3 (22.2%) | 1/3 (22.2%) |
| Subject 15 | 8 | 6/8 (70.8%) | 8/8 (100.0%) |
| Subject 19 | 8 | 5/8 (66.7%) | 6/8 (75.0%) |

---

## 5. Optimism Gap Analysis (In-Sample vs LOFO)

| Preset | Metric | In-Sample Value | LOFO Value | Optimism Gap (In-Sample minus LOFO) |
|:---|:---|:---:|:---:|:---:|
| `LOW_ALARM` | Lenient Sensitivity (%) | 44.93% | 44.93% | +0.00 pts |
| `LOW_ALARM` | False Alerts / hour | 58.6 | 58.6 | -0.0 FA/hr |
| `BALANCED` | Lenient Sensitivity (%) | 70.53% | 70.53% | -0.00 pts |
| `BALANCED` | False Alerts / hour | 112.0 | 112.0 | -0.0 FA/hr |

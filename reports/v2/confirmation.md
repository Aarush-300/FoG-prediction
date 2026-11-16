# Confirmation Report: Rotation Recipe Generalization (Step 2)

**Evaluation Date:** 2026-11-16  
**Adopted Recipe Determination:** `baseline_rotation_35`  
**Pre-Registered Rule 1.1 Criteria:**
1. Mean paired LOFO lenient sensitivity difference (b_rotation_70 minus baseline, same seed) $\ge +5.86$ points.
2. Mean paired FA/h change not worse than baseline by more than $+10.0\%$.

---

## 1. Three-Seed Comparison Table

| Seed | Recipe | Window ROC-AUC | Window PR-AUC | LOFO Lenient Sens (%) | LOFO Strict Sens (%) | LOFO Relaxed Sens (%) | Median Delay (s) | LOFO FA/hr |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | Baseline (±35°) | 0.7090 | 0.5578 | 59.4% | 4.3% | 9.4% | 1.89s | 99.7 |
| 42 | b_rotation_70 (±70°) | 0.7152 | 0.5650 | 72.5% | 8.5% | 10.0% | 1.49s | 89.6 |
| 43 | Baseline (±35°) | 0.6912 | 0.5100 | 55.1% | 10.4% | 14.9% | 0.47s | 134.4 |
| 43 | b_rotation_70 (±70°) | 0.6847 | 0.5156 | 55.1% | 9.9% | 11.7% | 0.55s | 129.9 |
| 44 | Baseline (±35°) | 0.7451 | 0.5820 | 66.7% | 9.6% | 16.2% | 1.82s | 94.1 |
| 44 | b_rotation_70 (±70°) | 0.6930 | 0.5294 | 52.2% | 15.7% | 17.3% | 1.24s | 145.6 |

---

## 2. Per-Seed Paired Differences (b_rotation_70 minus Baseline)

| Seed | Paired Lenient Diff (pts) | Paired FA/hr Diff | FA/hr Relative Change (%) |
|:---:|:---:|:---:|:---:|
| 42 | +13.04 | -10.1 | -10.1% |
| 43 | +0.00 | -4.5 | -3.3% |
| 44 | -14.49 | +51.5 | +54.8% |
| **Mean** | **-0.48** | **+12.3** | **+13.8%** |

---

## 3. Per-Fold Paired Differences Across Folds

| Fold Index | Subjects in Fold | Mean Paired Lenient Diff (pts) | SD Across Seeds (pts) |
|:---:|:---|:---:|:---:|
| Fold 0 | Subjs [2, 8, 16] | +11.11 | 3.93 |
| Fold 1 | Subjs [19, 6, 17] | -37.50 | 20.41 |
| Fold 2 | Subjs [1, 7, 12] | -7.41 | 34.35 |
| Fold 3 | Subjs [3, 10, 4] | +25.93 | 20.95 |
| Fold 4 | Subjs [15, 14, 13] | -33.33 | 55.71 |
| Fold 5 | Subjs [11, 9, 5] | +16.67 | 36.82 |
| **All Folds** | **All 18 Dev Subjects** | **-4.09** | **40.94** |

---

## 4. Rule 1.1 Adoption Decision

- **Condition (a) - Sensitivity Gain:** Mean paired lenient sensitivity difference is **-0.48 points** (Required: $\ge +5.86$ points) $\to$ **NOT MET**.
- **Condition (b) - False Alert Rate:** Mean relative FA/hr change is **+13.77%** (Required: $\le +10.0\%$) $\to$ **NOT MET**.
- **Formal Adoption Decision:**
  > **Baseline recipe (±35° rotation augmentation) is RETAINED.**

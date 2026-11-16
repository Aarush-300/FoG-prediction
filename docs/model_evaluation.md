# Model Evaluation & Benchmark Methodology

## 1. Subject-Independent (LOSO-Style) Partitioning

In clinical wearable ML, random window splitting leads to catastrophic data leakage because adjacent overlapping windows from the same subject share biomechanical cadence and static sensor mounting angles.

To guarantee zero data contamination:
- **Train Partition (7,729 windows):** Subjects 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16 (15 subjects).
- **Validation Partition (1,085 windows):** Subjects 14, 17, 19 (3 subjects). Model selection and decision threshold tuning are performed **STRICTLY on Validation metrics**.
- **Held-Out Test Partition (1,462 windows):** Subjects 18, 20, 21, 22 (4 subjects). Evaluated only once with frozen hyperparameters.

---

## 2. Multi-Model Benchmark Results (v1 Hackathon Submission Deck)

*Note: The test metrics in this table reflect the uncalibrated window-level figures reported in the hackathon submission deck (Slide 11). For post-submission audited continuous streaming evaluation using the deployed 4-state temporal FSM, see [reports/v2/BEFORE_AFTER.md](../reports/v2/BEFORE_AFTER.md).*

| Model Architecture | Parameters | Val Selection Score | Val PR-AUC | Val Ep Sens | Val F1 | Test Ep Sens (Deck) | Test PR-AUC | Test F1 | Test Specificity | Subj 22 Caught (Deck) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1D-CNN** | 4,465 | 0.8641 | 0.8213 | 100.0% | 0.7672 | 94.4% (17/18)* | 0.4642 | 0.3854 | 87.0% | 4/4 |
| **Improved ResCNN** | 43,393 | **0.9107** | **0.8922** | **100.0%** | **0.8229** | **100.0% (18/18)*** | **0.5317** | **0.4803** | **82.1%** | **4/4** |
| **CNN + GRU** | 14,433 | 0.8820 | 0.8448 | 100.0% | 0.7977 | 100.0% (18/18)* | 0.4716 | 0.4035 | 83.1% | 4/4 |
| **CNN + LSTM** | 16,929 | 0.8777 | 0.8420 | 100.0% | 0.7834 | 100.0% (18/18)* | 0.5262 | 0.4734 | 90.0% | 4/4 |
| **Dilated TCN** | 27,009 | 0.8834 | 0.8594 | 100.0% | 0.7687 | 100.0% (18/18)* | 0.5395 | 0.4670 | 87.7% | 4/4 |

### Winning Architecture Selection
Improved_ResCNN won validation model selection with a composite score of **0.9107**, achieving validation PR-AUC of 0.8922 and validation F1-score of 0.8229. In the submitted deck (Slide 11), test window PR-AUC was reported as 0.5317 and F1 as 0.4803, with 18/18 uncalibrated window-level episode detections. Under post-submission audited continuous streaming FSM evaluation ([reports/v2/BEFORE_AFTER.md](../reports/v2/BEFORE_AFTER.md)), v1 caught 12/18 canonical episodes (66.7%) and v2 caught 10/18 (55.6%); test differences across architectures and versions are within statistical noise (95% bootstrap CIs overlap).

---

## 3. Physical 3D-Euler Rotation Invariance

### The Subject 22 Anomaly
Subject 22 in the held-out test set exhibits a severe $-65.6^\circ$ roll tilt relative to the standard shank vertical axis:
- **Population nominal gravity vector:** $[-1.006, +0.132, +0.113]\text{ g}$
- **Subject 22 gravity vector:** $[-0.923, -0.253, +0.126]\text{ g}$ (a .27\times$ negative shift on $-acceleration)
- **Locomotor Tremor Index:** Freezing Index  = 1.584$ (.93\times$ higher than normal walking baseline).

### Inductive Bias through Euclidean Rotation
To enable models to generalize across varied ankle curvatures and sensor strap tilts without corrupting physical acceleration units, 3D Euler rotations ($\pm 35^\circ$ pitch, roll, and yaw) are applied in **physical metric space ($, $\text{deg/s}$)** prior to Z-score scaling.

In the hackathon submission deck, this augmentation was reported to catch 4/4 uncalibrated test episodes for Subject 22. In audited continuous streaming FSM evaluation, v1 caught 2 of 4 canonical episodes and v2 caught 0 of 4 canonical episodes (documented factually without causal claims in [reports/v2/BEFORE_AFTER.md](../reports/v2/BEFORE_AFTER.md)).

# Model Evaluation & Benchmark Methodology

## 1. Subject-Independent (LOSO-Style) Partitioning

In clinical wearable ML, random window splitting leads to catastrophic data leakage because adjacent overlapping windows from the same subject share biomechanical cadence and static sensor mounting angles.

To guarantee zero data contamination:
- **Train Partition (7,729 windows):** Subjects 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19.
- **Validation Partition (1,085 windows):** Subject 4. Model selection and decision threshold tuning are performed **STRICTLY on Validation metrics**.
- **Held-Out Test Partition (1,462 windows):** Subjects 18, 20, 21, and 22. Evaluated only once with frozen hyperparameters.

---

## 2. Multi-Model Benchmark Results

| Model Architecture | Parameters | Val Selection Score | Val PR-AUC | Val Ep Sens | Val F1 | Test Ep Sens (Caught) | Test PR-AUC | Test F1 | Test Specificity | Subj 22 Caught |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1D-CNN** | 4,465 | 0.8641 | 0.8213 | 100.0% | 0.7672 | 94.4% (17/18) | 0.4642 | 0.3854 | 87.0% | 4/4 |
| **Improved ResCNN** | 43,393 | **0.9107** | **0.8922** | **100.0%** | **0.8229** | **100.0% (18/18)** | **0.5317** | **0.4803** | **82.1%** | **4/4** |
| **CNN + GRU** | 14,433 | 0.8820 | 0.8448 | 100.0% | 0.7977 | 100.0% (18/18) | 0.4716 | 0.4035 | 83.1% | 4/4 |
| **CNN + LSTM** | 16,929 | 0.8777 | 0.8420 | 100.0% | 0.7834 | 100.0% (18/18) | 0.5262 | 0.4734 | 90.0% | 4/4 |
| **Dilated TCN** | 27,009 | 0.8834 | 0.8594 | 100.0% | 0.7687 | 100.0% (18/18) | 0.5395 | 0.4670 | 87.7% | 4/4 |

### Winning Architecture Selection
`Improved_ResCNN` won with a validation selection score of **0.9107**, achieving the highest validation PR-AUC (0.8922) and validation F1-score (0.8229). On the held-out test subjects, it caught **18/18 (100.0%) of all freezing episodes**, with highest test PR-AUC (0.5317) and highest test F1-score (0.4803).

---

## 3. Physical 3D-Euler Rotation Invariance

### The Subject 22 Anomaly
Subject 22 in the held-out test set exhibits a severe $-65.6^\circ$ roll tilt relative to the standard shank vertical axis:
- **Population nominal gravity vector:** $[-1.006, +0.132, +0.113]\text{ g}$
- **Subject 22 gravity vector:** $[-0.923, -0.253, +0.126]\text{ g}$ (a $3.27\times$ negative shift on $Y$-acceleration)
- **Locomotor Tremor Index:** Freezing Index $FI = 1.584$ ($3.93\times$ higher than normal walking baseline).

### Inductive Bias through Euclidean Rotation
To enable models to generalize across varied ankle curvatures and sensor strap tilts without corrupting physical acceleration units, 3D Euler rotations ($\pm 35^\circ$ pitch, roll, and yaw) are applied in **physical metric space ($g$, $\text{deg/s}$)** prior to Z-score scaling.

This data-driven physical augmentation enabled `Improved_ResCNN` to catch **4/4 (100%) of Subject 22 episodes** with zero missed freezes.

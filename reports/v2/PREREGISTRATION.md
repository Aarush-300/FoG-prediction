# Pre-Registration Plan: Wearable FoG Prediction v2 Improvement & Hardening

**Document Status:** Locked & Pre-Registered Prior to Any Model Training  
**Date:** November 16, 2026 (Amended with Enhanced FSM Grid & Pre-FoG Weighting)  
**Target Repository:** ESP32 Parkinson's Freezing-of-Gait Prediction System  
**Git Tag Reference:** `v1-submitted` (Hackathon baseline snapshot)

---

## 1. Test Set Separation & Test Discipline

1. **Held-Out Test Cohort:**
   - Test subjects remain strictly frozen: **Subjects 18, 20, 21, and 22** ($1,462$ windows, $18$ ground-truth freeze episodes).
   - Matches the hackathon submission to ensure unbiased, head-to-head comparability.
2. **Strict Test Isolation Rule:**
   - **Zero Test Feedback:** Nothing about the test set (features, predictions, losses, or distributions) shall influence any development decision (architecture selection, data augmentations, loss functions, learning rate schedules, operating thresholds, FSM parameters, or random seeds).
   - **Single Evaluation:** The held-out test cohort shall be evaluated **exactly once** at the very conclusion of the pipeline (Phase 3), after all model weights, quantization tables, scaler constants, and FSM parameters are completely frozen.

---

## 2. Development Protocol & Cross-Validation (Phase 2)

1. **Development Cohort Pooling:**
   - All 18 non-test subjects (previously split into 15 train and 3 validation) are pooled into a single development cohort:
     $$\text{Dev Subjects} = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19\}$$
     Total windows: $8,814$.
2. **6-Fold Subject-Grouped Cross-Validation:**
   - 18 subjects split into 6 disjoint folds (3 subjects per fold).
   - Strict subject-grouping: No subject appears in both training and validation folds.
   - Stratified grouping ensures freeze episodes are distributed across folds.
   - Out-of-Fold (OOF) predictions are recorded for all $8,814$ windows across all folds.
3. **Scaler Fitting Discipline:**
   - In cross-validation, the standard Z-score scaler $(\mu, \sigma)$ is fit **strictly on the training fold subjects only**, and applied to the validation fold subjects.
   - The final deployable production scaler is fit across all 18 development subjects pooled together. No test data is ever seen by the scaler.
4. **Computational Budget:**
   - CPU-only training execution.
   - Each training run constrained to under 10 minutes.
   - Fixed random seeds logged in all experiment scripts.

---

## 3. Evaluation Metric Definitions (Phase 1 Harness)

Continuous trials are evaluated by ordering windows chronologically by timestamp and replaying through the exact 4-state alarm FSM (`fog_detection.cpp`, $50\text{ Hz}$, $0.5\text{ s}$ stride):

1. **Ground Truth Episode Boundaries:**
   - Physical onset $T_{\text{onset}}$ and offset $T_{\text{offset}}$ are derived directly from dataset label annotations (`fog` column).
   - **Freeze Hazard Zone:** $[T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}]$.
2. **False Alerts per Hour of Normal Gait:**
   - Normal gait duration is defined as all trial time outside freeze hazard zones.
   - An alert is classified as **False** if it triggers at time $t$ where:
     $$t \notin [T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}] \quad \forall \text{ episodes}$$
   - Metric reported in three groups:
     * **True False Alerts:** Alerts during clean normal locomotion ($> 3.0\text{ s}$ before any freeze onset AND $> 3.0\text{ s}$ after offset).
     * **Boundary Alerts:** Alerts in the immediate pre-freeze buffer ($[T_{\text{onset}} - 0.5\text{ s}, T_{\text{onset}}]$) or post-freeze recovery ($[T_{\text{offset}}, T_{\text{offset}} + 3.0\text{ s}]$).
     * **Total False Alerts:** Sum of True False Alerts + Boundary Alerts.
   - Metric:
     $$\text{FA / hr} = \frac{\text{Total False Alerts}}{\text{Total Normal Gait Hours}}$$
3. **STRICT Episode Sensitivity:**
   - An episode is classified as caught strictly if at least one alert fires in the pre-freeze warning horizon:
     $$T_{\text{onset}} - 3.0\text{ s} \le t_{\text{alert}} \le T_{\text{onset}} - 0.5\text{ s}$$
4. **RELAXED Early Warning Metric (Lead Time $\ge 0.0\text{ s}$):**
   - An episode is classified as caught under relaxed early warning if an alert fires on or before physical freeze onset:
     $$T_{\text{onset}} - 3.0\text{ s} \le t_{\text{alert}} \le T_{\text{onset}} \quad (\text{i.e. } \Delta t_{\text{lead}} \ge 0.0\text{ s})$$
5. **LENIENT Episode Sensitivity:**
   - An episode is classified as caught leniently if at least one alert fires between pre-freeze onset and freeze termination:
     $$T_{\text{onset}} - 3.0\text{ s} \le t_{\text{alert}} \le T_{\text{offset}}$$
6. **Lead-Time Distribution & Decision Latency:**
   - For all caught episodes, early warning lead time is $\Delta t_{\text{lead}} = T_{\text{onset}} - t_{\text{first alert}}$.
   - Metrics reported: Mean lead time, Median lead time, and $\% \text{ alerts} \ge 0.5\text{ s early}$.
   - **Decision Latency:** Total physical elapsed time from window completion to haptic firing:
     $$\text{Latency} = t_{\text{inference}} + (N_{\text{confirm}} - 1) \times 0.5\text{ s}$$
7. **Window-Level Metrics:**
   - PR-AUC, ROC-AUC, $F_1$-score, Window Sensitivity, Window Specificity, and Raw Window False Alerts/hr (without FSM filtering).

---

## 4. Model Architecture & Embedded Op Constraints

1. **Architecture Family Invariant:**
   - Strictly remains within the **Improved_ResCNN** family (`src/models_zoo.py: build_improved_res_cnn`):
     - Conv1D stem (kernel 5)
     - Residual Stage 1 (two 32-filter Conv1D blocks + residual identity skip)
     - Residual Stage 2 (two 64-filter Conv1D blocks + 1x1 Conv1D projection skip)
     - Feature Aggregation Conv1D (64 filters)
     - Global Average Pooling (GAP)
     - Dense hidden layer + Dropout
     - Sigmoid single-unit output
   - Widths may be adjusted by at most $\pm 25\%$.
2. **Firmware TFLM Op Resolver Constraint:**
   - The INT8 quantized `.tflite` model must only contain operators supported by the firmware resolver:
     - `CONV_2D`, `ADD`, `MAX_POOL_2D`, `MEAN` (Global Average Pooling), `FULLY_CONNECTED`, `LOGISTIC` (Sigmoid), `RESHAPE`, `EXPAND_DIMS`, `QUANTIZE`, `DEQUANTIZE`.
   - Input shape: `(1, 100, 6)` INT8. Output shape: `(1, 1)` INT8.

---

## 5. Experimentation Protocol & Exploration Matrix

Evaluated against the baseline v1 recipe using 6-fold OOF CV across 3 random seeds (`[42, 43, 44]`):

| Category | Candidate Variants | Validation Selection Criterion |
|---|---|---|
| **a) Baseline** | v1 training recipe adapted to 6-fold CV | Benchmark reference |
| **b) 3D Euler Rotation Augmentation** | Roll up to $\pm 35^\circ$ vs $\pm 50^\circ$ vs $\pm 70^\circ$ (pitch/yaw $\le 35^\circ$), applied in physical units ($g, ^\circ/\text{s}$) prior to normalization | Highest OOF PR-AUC & resilience to extreme sensor tilt |
| **c) Sensor Mismatch & Noise** | Per-window gain ($\pm 5\%$), accel bias ($\pm 0.03\text{ g}$), gyro bias ($\pm 3^\circ/\text{s}$), Gaussian jitter, and simulated $20\text{--}24\text{ Hz}$ haptic vibration noise | Robustness against MPU6050 calibration drift and motor mechanical coupling |
| **d) Loss & Imbalance** | Standard BCE vs Class Weighted BCE vs Focal Loss ($\gamma \in \{1.0, 2.0\}$) vs Label Smoothing ($0.05$) | Highest OOF PR-AUC with low calibration error |
| **e) Pre-FoG Horizon Weighting** | Pre-FoG window loss weighting at **$1\times$ vs $2\times$ vs $4\times$** relative to normal windows | Prioritize pre-freeze feature learning to maximize lead time |
| **f) Regularization & Schedule** | Dropout ($\{0.2, 0.3, 0.4\}$), $L_2$ weight decay ($\{0, 10^{-4}\}$), Cosine Annealing learning rate schedule | Best OOF generalization without overfitting |

**Significance Rule:** Any candidate variant must beat the baseline recipe by more than one standard error across CV folds to be adopted into the final configuration; otherwise, the simpler baseline variant is retained and the negative finding is documented.

---

## 6. Pre-Registered Operating Point Selection Rule (Phase 2.3)

Using pooled Out-of-Fold (OOF) continuous trial predictions from all 18 development subjects:
1. **FSM Hyperparameter Grid:**
   - Consecutive confirmations: $N_{\text{confirm}} \in \{1, 2, 3\}$ (explicitly testing $N_{\text{confirm}}=1$ for prompt immediate alerting).
   - EMA smoothing: $\alpha \in \{0.60, 0.70, 0.80, 0.90, 1.00\}$ (including $\alpha=1.00$ for zero EMA filter delay).
   - High decision threshold: $T_{\text{high}} \in [0.35, 0.70]$.
   - Low reset threshold: $T_{\text{low}} \in [0.15, 0.35]$ ($T_{\text{low}} < T_{\text{high}}$).
   - Refractory cooldown: $T_{\text{cooldown}} \in [2000, 6000]\text{ ms}$.
2. **Operating-Point Rule:**
   - **Primary Objective:** **Maximize STRICT episode sensitivity** subject to the hard constraint:
     $$\text{False Alerts / hour} \le 30.0 \quad \text{on pooled OOF development trials}$$
   - **Fall-Through Clause:** If strict episode sensitivity remains below $30.0\%$ under this $\le 30\text{ FA/hr}$ constraint, explicitly state so, document the trade-off, and report the full Pareto frontier across (Strict Sensitivity, Relaxed Sensitivity [lead $\ge 0$], Lenient Sensitivity, False Alerts/hr).

---

## 7. Final Training & Random Seed Protocol (Phase 2.4)

1. **Full Development Retraining:**
   - Retrain the winning architecture and augmentation recipe on all 18 development subjects using the optimal epoch count determined by CV.
2. **Multi-Seed Stability Evaluation:**
   - Train across **5 fixed random seeds:** `[42, 43, 44, 45, 46]`.
   - Calculate OOF development performance for all 5 seeds.
3. **Deployment Seed Selection:**
   - The single seed with the **highest OOF-derived score on development data** will be selected as the final deployed model.
   - **Test set performance will NEVER be used to choose the model seed.**
   - All 5 test results will be reported (mean $\pm$ std) in the final report to provide full visibility into seed variance.

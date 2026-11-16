# Pre-Registration Plan: Wearable FoG Prediction v2 Improvement & Hardening

**Document Status:** Locked & Pre-Registered Prior to Any Model Training  
**Date:** November 16, 2026  
**Target Repository:** ESP32 Parkinson's Freezing-of-Gait Prediction System  
**Git Tag Reference:** `v1-submitted` (Hackathon baseline snapshot)

---

## 1. Test Set Separation & Test Discipline

1. **Held-Out Test Cohort:**
   - Test subjects remain strictly frozen: **Subjects 18, 20, 21, and 22** ($1,462$ windows, $18$ ground-truth freeze episodes).
   - This exact cohort matches the hackathon submission to ensure unbiased, head-to-head comparability.
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
   - For each freeze event, physical onset is $T_{\text{onset}}$ and offset is $T_{\text{offset}}$ derived from continuous trial annotations.
   - **Freeze Hazard Zone:** $[T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}]$.
2. **False Alerts per Hour of Normal Gait:**
   - Normal gait duration is defined as all trial time outside freeze hazard zones.
   - An alert is classified as **False** if it triggers at time $t$ where:
     $$t \notin [T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}] \quad \forall \text{ episodes}$$
   - Metric:
     $$\text{FA / hr} = \frac{\text{Total False Alerts}}{\text{Total Normal Gait Hours}}$$
3. **STRICT Episode Sensitivity:**
   - An episode is classified as caught strictly if at least one alert fires in the pre-freeze warning horizon:
     $$T_{\text{onset}} - 3.0\text{ s} \le t_{\text{alert}} \le T_{\text{onset}} - 0.5\text{ s}$$
   - Percentage of total episodes caught within this pre-freeze window.
4. **LENIENT Episode Sensitivity:**
   - An episode is classified as caught leniently if at least one alert fires between pre-freeze onset and freeze termination:
     $$T_{\text{onset}} - 3.0\text{ s} \le t_{\text{alert}} \le T_{\text{offset}}$$
5. **Lead-Time Distribution:**
   - For all caught episodes, early warning lead time is:
     $$\Delta t_{\text{lead}} = T_{\text{onset}} - t_{\text{first alert}}$$
   - Metrics reported: Mean lead time, Median lead time, and $\% \text{ alerts} \ge 0.5\text{ s early}$.
6. **Window-Level Metrics:**
   - Area Under the Precision-Recall Curve (PR-AUC).
   - Area Under the ROC Curve (ROC-AUC).
   - Window Sensitivity (Recall), Specificity, and $F_1$-score.

---

## 4. Model Architecture & Embedded Op Constraints

1. **Architecture Family Invariant:**
   - The architecture must strictly remain within the **Improved_ResCNN** family (`src/models_zoo.py: build_improved_res_cnn`):
     - Conv1D stem (kernel 5)
     - Residual Stage 1 (two 32-filter Conv1D blocks + residual identity skip)
     - Residual Stage 2 (two 64-filter Conv1D blocks + 1x1 Conv1D projection skip)
     - Feature Aggregation Conv1D (64 filters)
     - Global Average Pooling (GAP)
     - Dense hidden layer + Dropout
     - Sigmoid single-unit output
   - No recurrent cells (LSTM/GRU), Dilated TCNs, or Transformers permitted.
   - Widths may be adjusted by at most $\pm 25\%$.
2. **Firmware TFLM Op Resolver Constraint:**
   - The INT8 quantized `.tflite` model must only contain operators already supported by the ESP32 TFLM MicroMutableOpResolver:
     - `CONV_2D`, `ADD`, `MAX_POOL_2D`, `MEAN` (Global Average Pooling), `FULLY_CONNECTED`, `LOGISTIC` (Sigmoid), `RESHAPE`, `EXPAND_DIMS`, `QUANTIZE`, `DEQUANTIZE`.
   - Input shape must remain `(1, 100, 6)` INT8, output shape `(1, 1)` INT8.

---

## 5. Experimentation Protocol & Exploration Matrix

A systematic series of controlled ablation experiments will be evaluated against the baseline v1 recipe using 6-fold OOF CV (evaluated across 3 random seeds: [42, 43, 44]):

| Category | Candidate Variants | Validation Selection Criterion |
|---|---|---|
| **a) Baseline** | v1 training recipe adapted to 6-fold CV | Benchmark reference |
| **b) 3D Euler Rotation Augmentation** | Roll up to $\pm 35^\circ$ vs $\pm 50^\circ$ vs $\pm 70^\circ$ (pitch/yaw $\le 35^\circ$), applied in physical units ($g, ^\circ/\text{s}$) prior to normalization | Highest OOF PR-AUC & resilience to extreme sensor tilt |
| **c) Sensor Mismatch & Noise** | Per-window gain ($\pm 5\%$), accel bias ($\pm 0.03\text{ g}$), gyro bias ($\pm 3^\circ/\text{s}$), Gaussian jitter, and simulated $20\text{--}24\text{ Hz}$ haptic vibration noise | Robustness against MPU6050 calibration drift and motor mechanical coupling |
| **d) Loss & Imbalance** | Standard BCE vs Class Weighted BCE vs Focal Loss ($\gamma \in \{1.0, 2.0\}$) vs Label Smoothing ($0.05$) | Highest OOF PR-AUC with low calibration error |
| **e) Regularization & Schedule** | Dropout ($\{0.2, 0.3, 0.4\}$), $L_2$ weight decay ($\{0, 10^{-4}\}$), Cosine Annealing learning rate schedule | Best OOF generalization without overfitting |

**Significance Rule:** Any candidate variant must beat the baseline recipe by more than one standard error across CV folds to be adopted into the final configuration; otherwise, the simpler baseline variant is retained and the negative finding is documented.

---

## 6. Pre-Registered Operating Point Selection Rule (Phase 2.3)

Using pooled Out-of-Fold (OOF) continuous trial predictions from all 18 development subjects:
1. Grid search FSM hyperparameter space:
   - EMA smoothing $\alpha \in [0.40, 0.80]$
   - High decision threshold $T_{\text{high}} \in [0.40, 0.70]$
   - Low reset threshold $T_{\text{low}} \in [0.15, 0.35]$ ($T_{\text{low}} < T_{\text{high}}$)
   - Consecutive confirmations $N_{\text{confirm}} \in \{1, 2, 3\}$
   - Refractory cooldown $T_{\text{cooldown}} \in [2000, 6000]\text{ ms}$
2. **Selection Objective:**
   - **Primary Constraint:** Achieve **STRICT Episode Sensitivity $\ge 75\%$** on OOF development episodes.
   - **Optimization Target:** Minimize **False Alerts per Hour of Normal Gait**.
   - *Fallback Clause:* If $\ge 75\%$ strict sensitivity cannot be achieved due to the physical confirmation delay, select the elbow/knee of the Pareto frontier (maximizing strict sensitivity while keeping false alerts under an acceptable ceiling) and explicitly document the clinical trade-off.

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

# Pre-Registration Plan: Wearable FoG Prediction & Fast Detection v2

**Document Status:** Locked & Pre-Registered Prior to Any Model Training  
**Date:** November 16, 2026 (Amended Following Phase 0/1 Diagnostics & Denominator Reconciliation)  
**Target Repository:** ESP32 Parkinson's Freezing-of-Gait Detection System  
**Git Tag Reference:** `v1-submitted` (Hackathon baseline snapshot)

> [!NOTE]
> **Protocol Amendment & Diagnostic Disclosure (November 16, 2026):**  
> This amended pre-registration plan supersedes prior drafts following comprehensive Phase 0 and Phase 1 diagnostics:
> 1. **Early Signal Reality:** Empirical diagnostic testing on both validation and held-out test sets revealed that kinematic IMU features carry negligible signal earlier than $0.5\text{ s}$ before freeze onset (pure pre-FoG ROC-AUC was $0.574$, hovering flat at baseline probability $\sim 0.20$).
> 2. **FSM Cooldown Locking:** Detailed timeline analysis revealed that pre-freeze spikes frequently locked the firmware FSM into persistent `COOLDOWN` due to the "needs $p < \text{low}$" re-arm condition, muting alerts throughout long freeze episodes.
> 3. **Denominator Reconciliation:** Normal gait time is now formally defined from $t = 2.0\text{ s}$ (the earliest possible 100-sample window completion) to hazard onset, removing $40.0\text{ s}$ of un-evaluable initial buffer across test trials ($580.60\text{ s} = 0.1613\text{ h}$).
> All model training, hyperparameter tuning, and operating point decisions remain strictly blinded to test set data, executed strictly on the 18 development subjects via 6-fold cross-validation.

---

## 1. Test Set Separation & Test Discipline

1. **Held-Out Test Cohort:**
   - Test subjects remain strictly frozen: **Subjects 18, 20, 21, and 22** ($1,462$ windows, $18$ ground-truth freeze episodes).
   - Matches the hackathon submission to ensure unbiased, head-to-head comparability.
2. **Strict Test Isolation Rule:**
   - **Zero Test Feedback:** Nothing about the test set (features, predictions, losses, or distributions) shall influence any development decision (architecture selection, data augmentations, loss functions, learning rate schedules, operating thresholds, FSM parameters, or random seeds).
   - **Single Evaluation:** The held-out test cohort shall be evaluated **exactly once** at the conclusion of the pipeline (Phase 3), after all model weights, quantization tables, scaler constants, and FSM parameters are completely frozen.

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
   - Constrained to under 10 minutes per training run.
   - Fixed random seeds logged in all experiment scripts.

---

## 3. Evaluation Metric Definitions & Primary Decision Criteria

Continuous trials are evaluated by ordering windows chronologically by timestamp and replaying through the exact 4-state alarm FSM (`fog_detection.cpp`, $50\text{ Hz}$, $0.5\text{ s}$ stride):

1. **Ground Truth Episode Boundaries:**
   - Physical onset $T_{\text{onset}}$ and offset $T_{\text{offset}}$ are derived directly from dataset label annotations (`fog` column).
   - **Freeze Hazard Zone:** $[T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}]$.
2. **Evaluable Normal Gait Time Denominator:**
   - Normal gait is defined strictly from $t \ge 2.0\text{ s}$ (the first possible window end timestamp) to hazard onset ($T_{\text{onset}} - 3.0\text{ s}$), plus post-freeze recovery ($t > T_{\text{offset}}$).
   - An alert is classified as **False** if its timestamp $t_{\text{alert}} \notin [T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}]$.
   - Metric:
     $$\text{FA / hr} = \frac{\text{Total False Alerts}}{\text{Total Evaluable Normal Gait Hours}}$$
3. **Episode Sensitivity Metrics:**
   - **STRICT Episode Sensitivity:** Alert fires in $[T_{\text{onset}} - 3.0\text{ s}, T_{\text{onset}} - 0.5\text{ s}]$.
   - **RELAXED Early Warning Sensitivity:** Alert fires in $[T_{\text{onset}} - 3.0\text{ s}, T_{\text{onset}}]$ ($\Delta t_{\text{lead}} \ge 0.0\text{ s}$).
   - **LENIENT Episode Sensitivity:** Alert fires in $[T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}]$.
4. **Fast-Detection Metrics (Primary if re-scoped, Secondary otherwise):**
   - **Median Detection Delay:** $t_{\text{alert}} - T_{\text{onset}}$ for caught episodes (negative = early warning, positive = reaction latency).
   - **% Alerted within $1.0\text{ s}$ of Onset:** Fraction of caught episodes with alert timestamp $\le T_{\text{onset}} + 1.0\text{ s}$.
   - **Lenient Episode Sensitivity** and **False Alerts / Hour**.
   - Note: STRICT sensitivity is retained and reported in all summaries regardless of scoping.

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
2. **Firmware TFLM Op Resolver Constraint:**
   - Supported ops: `CONV_2D`, `ADD`, `MAX_POOL_2D`, `MEAN` (GAP), `FULLY_CONNECTED`, `LOGISTIC` (Sigmoid), `RESHAPE`, `EXPAND_DIMS`, `QUANTIZE`, `DEQUANTIZE`.
   - Input shape: `(1, 100, 6)` INT8. Output shape: `(1, 1)` INT8.

---

## 5. Experimentation Protocol & Ordered Exploration Matrix

Experiments proceed in strict priority order. Any variant must demonstrate statistically significant improvement over baseline on 6-fold OOF CV (mean gain exceeding standard error across folds) to be retained.

### Priority Execution Order:

1. **Experiment (g) — Early Signal Check (GO/NO-GO Gateway):**
   - Train `Improved_ResCNN` strictly on **Pure Pre-FoG (label 1) vs Normal Gait (label 0)** windows only (dropping transition label -1 and active freeze label 2).
   - Evaluated across 6-fold subject-grouped CV with 3 fixed random seeds (`[42, 43, 44]`).
   - **Pre-Registered Decision Rule:**
     $$\text{If pooled OOF ROC-AUC } < 0.65 \quad \text{OR 95\% bootstrap CI includes } 0.50:$$
     The scientific claim of *"anticipatory early prediction 0.5–3.0s ahead"* is declared **empirically unsupported**, and the primary objective of the system is formally re-scoped to **FAST DETECTION** (rapid reaction upon freeze inception).
2. **Experiment (b) — 3D Euler Rotation Range:**
   - Test roll range up to $\pm 35^\circ$ vs $\pm 50^\circ$ vs $\pm 70^\circ$ (pitch/yaw $\le 35^\circ$) applied in physical units ($g, ^\circ/\text{s}$) prior to normalization. Selection by highest OOF PR-AUC and resilience to extreme sensor mounting tilt.
3. **Experiment (e) — Pre-FoG Loss Weighting:**
   - Pre-FoG window loss weights of $1\times$ vs $2\times$ vs $4\times$ relative to normal gait windows.
4. **Experiment (f) — Label Definition Variants:**
   - Variant 1: v1 default labels (transition windows count as positive).
   - Variant 2: Dropping transition windows (label -1) entirely from training data.
   - Variant 3: Clean positives: positives = pure pre-FoG (label 1) + active FoG (label 2) only.
   - Evaluated using both STRICT and RELAXED metrics.
5. **Contingent Exploration (Only if budget permits):**
   - (c) Sensor mismatch & noise injection (gain $\pm 5\%$, bias $\pm 0.03\text{ g}$, $20\text{--}24\text{ Hz}$ motor vibration).
   - (d) Loss formulations (Focal Loss $\gamma \in \{1, 2\}$, Label Smoothing $0.05$).
   - (h) Regularization (Dropout $\{0.2, 0.3, 0.4\}$, $L_2$ decay $\{0, 10^{-4}\}$).

---

## 6. Pre-Registered Operating Point Selection Rule (Phase 2.3)

Using pooled Out-of-Fold (OOF) continuous trial predictions from all 18 development subjects:

1. **FSM Hyperparameter Grid:**
   - Consecutive confirmations: $N_{\text{confirm}} \in \{1, 2, 3\}$.
   - EMA smoothing: $\alpha \in \{0.60, 0.70, 0.80, 0.90, 1.00\}$.
   - High decision threshold: $T_{\text{high}} \in [0.35, 0.70]$.
   - Low reset threshold: $T_{\text{low}} \in [0.15, 0.35]$ ($T_{\text{low}} < T_{\text{high}}$).
   - Refractory cooldown: $T_{\text{cooldown}} \in [2000, 6000]\text{ ms}$.
   - **Re-Arm Policy:**
     - Traditional policy: Cooldown ends after $T_{\text{cooldown}}$ **AND** $p < T_{\text{low}}$.
     - Timeout policy: Cooldown ends after $T_{\text{cooldown}}$ **regardless of probability** (prevents lockout during sustained freezes).
   - **Re-Cueing Policy:**
     - Optional periodic re-cue every $\{5, 10\}\text{ s}$ if probability remains continuously above $T_{\text{high}}$.
     - Re-cues are evaluated for total cue count per episode, but episode sensitivity and detection latency are measured strictly from the first alert.
2. **Operating-Point Selection Rule:**
   - **If Fast-Detection Re-scoping Fires:**
     - **Primary Objective:** Maximize **LENIENT episode sensitivity** subject to:
       $$\text{False Alerts / hour} \le 30.0 \quad \text{on pooled OOF development trials}$$
     - **Fall-Through Clause:** If $\le 30.0\text{ FA/hr}$ is unattainable while maintaining $\ge 70\%$ lenient sensitivity, report the full Pareto frontier and select the operating point with the lowest false alerts/hour that satisfies $\text{Lenient Sensitivity} \ge 70.0\%$.
   - **If Predictive Claim Holds:**
     - Maximize **STRICT episode sensitivity** subject to $\text{False Alerts / hour} \le 30.0$.

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

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
> 4. **Inner Validation Discipline for Early Stopping (Pre-Training Amendment):** To prevent leakage into out-of-fold (OOF) evaluation, early stopping and model checkpointing during 6-fold CV shall NOT monitor the outer held-out fold. Instead, each fold holds out a deterministic, stratified inner validation set of 2 subjects from the 15 training subjects. The 3 outer held-out subjects are strictly evaluated post-training for pure OOF generation.
> 5. **Experiment (g) Conclusion & Protocol Gateway (Pre-Training Amendment):** Experiment (g) empirical evaluation confirmed a weak early kinematic signal present (pooled ROC-AUC $0.7339$, $1.92\text{--}2.36\times$ PR lift over base rate, but matched normal controls showed overlapping CI difference $[-0.115, +0.479]$). Experiment (g) is officially accepted as "weak early signal present", not "prediction supported". The pre-FoG-only model in (g) is strictly a diagnostic, not a deployable candidate.
> 6. **Deployable Objective & Winner Rule for Experiments (b), (e), (f) (Pre-Training Amendment):**
>    - **Deployable Training Objective:** Experiments (b), (e), (f) use the full deployable binary task: positives = pre-FoG (1) + transition (-1) + active FoG (2) windows (v1 binary label), unless variant (f) explicitly states otherwise.
>    - **Full Continuous Trial & FSM Evaluation:** Evaluated on continuous trial replays using pooled OOF predictions with the FSM grid selected to maximize lenient sensitivity subject to $\text{FA/hr} \le 30.0$. Metrics reported: strict sensitivity, relaxed sensitivity ($\Delta t_{\text{lead}} \ge 0$), lenient sensitivity, median detection delay, % alerted within $1.0\text{ s}$ of onset, and FA/hr.
>    - **Optimism Gap Reporting:** FSM parameters and metrics are reported both (i) in-sample on pooled OOF and (ii) via leave-one-fold-out (LOFO: parameters chosen on 5 folds, evaluated on the 6th held-out fold).
>    - **Winner Rule:** A candidate variant replaces the baseline recipe ONLY IF its fold-level paired improvement in lenient sensitivity at the $\le 30.0\text{ FA/hr}$ budget exceeds the seed-to-seed standard deviation ($\sigma_{\text{seed}}$) measured from 3 seeds of the baseline (`baseline_v1` across seeds 42, 43, 44). Otherwise, the v1 recipe is retained.
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
2. **6-Fold Subject-Grouped Cross-Validation with Inner Validation Discipline:**
   - 18 subjects split into 6 disjoint outer folds (3 subjects per fold):
     - Fold 0: `[2, 8, 16]`
     - Fold 1: `[19, 6, 17]`
     - Fold 2: `[1, 7, 12]`
     - Fold 3: `[3, 10, 4]`
     - Fold 4: `[15, 14, 13]`
     - Fold 5: `[11, 9, 5]`
   - **Inner Validation Early-Stopping Sets:** To eliminate evaluation leakage, early stopping and model checkpointing do not monitor the outer fold. Instead, 2 subjects from the 15 training subjects are deterministically held out for validation during training:
     - Fold 0: `[19, 6]` (inner val) | `[1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17]` (train)
     - Fold 1: `[1, 7]` (inner val) | `[2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16]` (train)
     - Fold 2: `[3, 10]` (inner val) | `[2, 4, 5, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19]` (train)
     - Fold 3: `[15, 14]` (inner val) | `[1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 19]` (train)
     - Fold 4: `[11, 8]` (inner val) | `[1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 16, 17, 19]` (train)
     - Fold 5: `[2, 16]` (inner val) | `[1, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 17, 19]` (train)
   - Strict subject-grouping: The inner validation set and outer held-out fold share zero subjects across all folds.
   - Out-of-Fold (OOF) predictions are generated strictly on the outer held-out fold after training completes.
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
5. **Secondary Evaluation Horizon Analyses (Phase 1 Diagnostics Follow-Up):**
   - *Primary Hazard Window:* Starts at $T_{\text{onset}} - 3.0\text{ s}$ ($[T_{\text{onset}} - 3.0\text{ s}, T_{\text{offset}}]$).
   - *Secondary Hazard Window:* Starts at $T_{\text{onset}} - 5.0\text{ s}$ ($[T_{\text{onset}} - 5.0\text{ s}, T_{\text{offset}}]$).
   - Both false alerts per hour and lenient episode sensitivity will be reported side-by-side using the primary $3.0\text{ s}$ horizon and secondary $5.0\text{ s}$ horizon to evaluate whether earlier alerts (e.g. at $t = 27.5\text{ s}$ in Sub 20 Task 6) reflect extended anticipatory cues rather than false positives.
   - Primary decision rules remain strictly anchored to the primary $3.0\text{ s}$ hazard window.

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

### Priority Execution Order & Winner Rules for Deployable Candidates:

> [!IMPORTANT]
> **Deployable Candidate Training Constraint:** Experiments (b), (e), (f) are trained with the **deployable objective**: positives = pre-FoG (1) + transition (-1) + active FoG (2) windows (v1 binary label), unless variant (f) explicitly states otherwise. Experiment (g)'s pre-FoG-only model was purely a diagnostic, not a deployable candidate.
>
> **Winner Selection Rule:** A candidate variant replaces the baseline recipe ONLY IF its fold-level paired improvement in lenient sensitivity at the $\le 30.0\text{ FA/hr}$ budget exceeds the seed-to-seed standard deviation ($\sigma_{\text{seed}}$) measured across 3 seeds (`[42, 43, 44]`) of the baseline recipe (`b_rotation_35`). Otherwise, the baseline recipe is retained.
>
> **Optimism Gap Discipline:** For every variant, FSM parameters are reported both (i) in-sample on pooled OOF and (ii) by leave-one-fold-out (LOFO: parameters tuned on 5 training folds, evaluated on the held-out 6th fold).

1. **Baseline Recipe Retrain (`b_rotation_35` / Baseline v1):**
   - 3D Euler rotation range $\pm 35^\circ$, 1x pre-FoG loss weight, standard binary labels.
   - Evaluated across 3 seeds (`[42, 43, 44]`) to measure baseline performance and seed-to-seed standard deviation $\sigma_{\text{seed}}$.
2. **Experiment (b) — 3D Euler Rotation Range:**
   - Test roll range up to $\pm 35^\circ$ (baseline) vs $\pm 50^\circ$ (`b_rotation_50`) vs $\pm 70^\circ$ (`b_rotation_70`) applied in physical units ($g, ^\circ/\text{s}$) prior to normalization. Seed 42.
3. **Experiment (e) — Pre-FoG Loss Weighting:**
   - Pre-FoG window loss weights of $1\times$ (baseline) vs $2\times$ (`e_prefog_2x`) vs $4\times$ (`e_prefog_4x`) relative to normal gait windows. Seed 42.
4. **Experiment (f) — Label Definition Variants:**
   - Variant 1: Baseline v1 labels (transition windows count as positive).
   - Variant 2: `f_drop_transition`: Dropping transition windows (label -1) entirely from training data. Seed 42.
   - Variant 3: `f_clean_positives`: Positives = pure pre-FoG (label 1) + active FoG (label 2) only; label -1 dropped from train and eval. Seed 42.
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


---

## 8. Erratum and Step 1 Confirmation & Deployment Specifications (Dated: 2026-11-16)

### 8.0 Erratum on Experiment (g) Decision Label
The decision label `PREDICTION_CLAIM_SUPPORTED` is formally replaced by `WEAK_SIGNAL_PRESENT`.
**Reason:** The matched-normal control difference was +18.8 points with subject-bootstrap 95% CI [-11.5, +47.9], which includes zero, per-subject ROC-AUC exceeded 0.6 in 5 of 9 subjects, and strict sensitivity of the deployable pipeline is 4-19%.

### 8.1 Confirmation Rule for Rotation Recipe
Run `b_rotation_70` with seeds 43 and 44 (seed 42 already exists), giving 3 seeds each for baseline (+/-35 deg) and `b_rotation_70`. Compute, per seed, the paired difference (`b_rotation_70` minus baseline, same seed) in LOFO lenient sensitivity under the same selection procedure, and the same for FA/h.
`b_rotation_70` is adopted if and only if:
1. Mean paired lenient difference >= +5.86 points, AND
2. Mean paired FA/h is not worse than baseline by more than 10%.

Otherwise, the baseline recipe (+/-35 deg) is kept and the report states so. Per-fold paired differences (mean, SD) are also reported to verify that the effect is not carried by one fold alone.

### 8.2 Deployment Seed Rule
Final models are trained for seeds 42-46 on all 18 development subjects with a fixed epoch count equal to the median best-epoch of the adopted recipe's CV folds, with no early-stopping data (scaler fit on all 18 subjects). The deployed seed is the one among 42, 43, 44 with the highest pooled OOF window PR-AUC of the adopted recipe's CV runs. Test results are never used to choose a seed.

### 8.3 FSM Selection Rule
Tune on the INT8 OOF probabilities (quantize every fold model of the adopted recipe; the deployed artifact is INT8) pooled across the 3 seeds: pick the configuration with the best MEAN-across-seeds LOFO metric. Two presets are pre-declared:
1. **LOW_ALARM:** Maximize mean lenient sensitivity subject to mean FA/h <= 30 (report the result even if sensitivity is low).
2. **BALANCED:** Lowest mean FA/h subject to mean lenient sensitivity >= 70% (the existing fall-through rule).

The hyperparameter grid includes the re-arm policy (timeout vs 'timeout AND p < low') and re-cue interval {none, 5 s, 10 s}; the chosen values will be reported explicitly.

### 8.4 Claim Policy for README / Documentation
(Generated by `src/render_docs.py` from reports only):
The sentence "predicts freezes 0.5-3 s ahead" is allowed ONLY if test STRICT episode sensitivity >= 50% AND its bootstrap 95% CI lower bound >= 25%.
Otherwise, documentation must state the measured strict, relaxed, and lenient sensitivity, the median detection delay, and FA/h, and describe the system as a "fast freeze detector with a haptic cue". Test-set results are always reported with CIs and with the v1 numbers beside them, marked better / worse / within noise.

### 8.5 Test Discipline
The test set is evaluated exactly once with `src/final_test_eval.py`, which must:
1. Verify the SHA-256 of every frozen artifact against `reports/v2/frozen_manifest.json`.
2. Refuse to run if `reports/v2/test_eval.lock` exists.
3. Create that lock file when it starts. No `--force` option is supported.

Evaluate both presets (`LOW_ALARM` and `BALANCED`), plus these pre-declared ablations:
- (A) v1 INT8 + v1 firmware FSM (already done)
- (B) v1 INT8 + v1 FSM thresholds with only the timeout re-arm fix
- (C) v2 model + v2 FSM presets.
Nothing is tuned on any test result; if v2 is worse than v1 on any metric, it is reported.

### 8.6 Knowledge Note on Subject 22
The rotation candidates {35, 50, 70} were chosen knowing that Subject 22 (a test subject) had a 65.6 deg tilt; the selection itself uses development subjects only.

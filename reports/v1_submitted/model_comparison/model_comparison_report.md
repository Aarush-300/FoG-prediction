# Subject-Independent Model Selection & Validation Report

**Evaluation Mandate:** Select the highest performing model based **STRICTLY on subject-independent validation data** to prevent test set contamination.  
**Held-Out Test Set:** Subjects 18, 20, 21, and 22 evaluated once using the frozen validation threshold.

---

## 1. Candidate Architecture Comparison Table

| Architecture | Parameters | Val PR-AUC | Val Ep Sens | Val F1 | Validation Score | Test Ep Sens (Caught) | Test PR-AUC | Test F1 | Test Specificity | Subj 22 Caught | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline_1DCNN** | 4,465 | 0.8213 | 100.0% | 0.7672 | **0.8641** | 94.4% (17/18) | 0.4642 | 0.3854 | 87.0% | 4/4 | Candidate |
| **Improved_ResCNN** | 43,393 | 0.8922 | 100.0% | 0.8229 | **0.9107** | 100.0% (18/18) | 0.5317 | 0.4803 | 82.1% | 4/4 | **SELECTED BEST** |
| **CNN_GRU** | 14,433 | 0.8448 | 100.0% | 0.7977 | **0.8820** | 100.0% (18/18) | 0.4716 | 0.4035 | 83.1% | 4/4 | Candidate |
| **CNN_LSTM** | 16,929 | 0.8420 | 100.0% | 0.7834 | **0.8777** | 100.0% (18/18) | 0.5262 | 0.4734 | 90.0% | 4/4 | Candidate |
| **Dilated_TCN** | 27,009 | 0.8594 | 100.0% | 0.7687 | **0.8834** | 100.0% (18/18) | 0.5395 | 0.4670 | 87.7% | 4/4 | Candidate |

---

## 2. Selection Rationale

**Selected Architecture:** `Improved_ResCNN`  
- **Validation Selection Score:** `0.9107`
- **Reason:** Demonstrated the highest generalizable predictive performance on subject-independent validation subjects without any test set feedback.
- **Held-Out Test Verification:** On held-out Subjects 18, 20, 21, and 22, the model caught **18/18 episodes (100.0%)**, with Test PR-AUC of **0.5317** and Test F1 of **0.4803**.
- **Subject 22 Resolution:** Successfully caught **4/4** of recorded episodes despite severe mounting roll tilt.

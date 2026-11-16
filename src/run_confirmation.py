"""
Run confirmation for b_rotation_70 across seeds 43 and 44 (seed 42 already completed).
Computes 3-seed comparison vs baseline (+/-35 deg), paired differences, per-fold paired differences,
and evaluates pre-registered Rule 1.1 to decide whether b_rotation_70 is adopted.
Outputs reports/v2/confirmation.md and reports/v2/diagnostics/confirmation_report.json.
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cv_splits import DEV_FOLDS, INNER_VAL_FOLDS, TEST_SUBJECTS, verify_test_isolation
from train_cv import load_dev_data
from run_phase2_experiments import train_and_get_full_oof, build_dev_trials, eval_in_sample_and_lofo


def run_confirmation():
    print("=" * 80)
    print("STEP 2: CONFIRMATION RUNS FOR B_ROTATION_70 (SEEDS 43 & 44)")
    print("=" * 80)

    # Load baseline runs from phase2_experiments_full_report.json
    with open('reports/v2/diagnostics/phase2_experiments_full_report.json', 'r') as f:
        full_rep = json.load(f)

    baseline_runs = full_rep['baseline_stability']['seed_runs']
    b70_seed42 = full_rep['variants']['b_rotation_70']

    _, _, _, meta_dev = load_dev_data()
    verify_test_isolation(meta_dev)

    b70_seed_runs = {'42': b70_seed42}

    # Run seeds 43 and 44 for b_rotation_70
    for seed in [43, 44]:
        print(f"\n>>> Running b_rotation_70 for SEED {seed}...")
        probs, w_roc, w_pr, t_time = train_and_get_full_oof(
            exp_id='b_rotation_70',
            seed=seed,
            max_angle_deg=70.0,
            prefog_weight=1.0,
            epochs=40,
            batch_size=64
        )

        dev_trials = build_dev_trials(probs, meta_dev)
        eval_res = eval_in_sample_and_lofo(dev_trials, max_fa_hr=30.0)

        b70_seed_runs[str(seed)] = {
            'exp_id': 'b_rotation_70',
            'seed': seed,
            'max_angle_deg': 70.0,
            'window_roc': w_roc,
            'window_pr': w_pr,
            'train_time_s': t_time,
            'eval_res': eval_res,
            'fold_lenient_sens': eval_res['fold_lenient_sens']
        }
        print(f"Seed {seed} finished. Window PR: {w_pr:.4f}, LOFO Lenient: {eval_res['lofo_metrics']['lenient_sensitivity']*100:.1f}%, FA/hr: {eval_res['lofo_metrics']['fa_hr']:.1f}")

    # Build 3-Seed Comparison Table
    # Baseline (+/-35 deg) vs b_rotation_70
    seeds = [42, 43, 44]
    paired_lenient_diffs = []
    paired_fa_diffs = []
    paired_fa_pct_diffs = []

    per_seed_table = []
    for s in seeds:
        s_str = str(s)
        base = baseline_runs[s_str]
        b70 = b70_seed_runs[s_str]

        base_lofo = base['eval_res']['lofo_metrics']
        b70_lofo = b70['eval_res']['lofo_metrics']

        d_lenient = (b70_lofo['lenient_sensitivity'] - base_lofo['lenient_sensitivity']) * 100.0
        d_fa = b70_lofo['fa_hr'] - base_lofo['fa_hr']
        d_fa_pct = (d_fa / base_lofo['fa_hr']) * 100.0

        paired_lenient_diffs.append(d_lenient)
        paired_fa_diffs.append(d_fa)
        paired_fa_pct_diffs.append(d_fa_pct)

        per_seed_table.append({
            'seed': s,
            'base_window_roc': base['window_roc'],
            'base_window_pr': base['window_pr'],
            'base_lenient': base_lofo['lenient_sensitivity'] * 100.0,
            'base_strict': base_lofo['strict_sensitivity'] * 100.0,
            'base_relaxed': base_lofo['relaxed_sensitivity'] * 100.0,
            'base_median_delay': base_lofo['median_delay_s'],
            'base_fa_hr': base_lofo['fa_hr'],
            'b70_window_roc': b70['window_roc'],
            'b70_window_pr': b70['window_pr'],
            'b70_lenient': b70_lofo['lenient_sensitivity'] * 100.0,
            'b70_strict': b70_lofo['strict_sensitivity'] * 100.0,
            'b70_relaxed': b70_lofo['relaxed_sensitivity'] * 100.0,
            'b70_median_delay': b70_lofo['median_delay_s'],
            'b70_fa_hr': b70_lofo['fa_hr'],
            'paired_lenient_diff': d_lenient,
            'paired_fa_diff': d_fa,
            'paired_fa_pct_diff': d_fa_pct
        })

    mean_paired_lenient = float(np.mean(paired_lenient_diffs))
    mean_paired_fa_pct = float(np.mean(paired_fa_pct_diffs))
    mean_paired_fa_diff = float(np.mean(paired_fa_diffs))

    # Per-fold paired differences across all 6 folds (averaged across seeds)
    # Also individual fold paired differences across all 18 fold-seed evaluations
    fold_paired_diffs_all = []
    fold_paired_diffs_by_fold = {f: [] for f in range(6)}

    for s in seeds:
        s_str = str(s)
        base_folds = baseline_runs[s_str]['eval_res']['lofo_fold_metrics']
        b70_folds = b70_seed_runs[s_str]['eval_res']['lofo_fold_metrics']
        for f in range(6):
            base_fold_data = base_folds.get(f) if f in base_folds else base_folds[str(f)]
            b70_fold_data = b70_folds.get(f) if f in b70_folds else b70_folds[str(f)]
            base_f_sens = base_fold_data['lenient_sensitivity'] * 100.0
            b70_f_sens = b70_fold_data['lenient_sensitivity'] * 100.0
            diff = b70_f_sens - base_f_sens
            fold_paired_diffs_all.append(diff)
            fold_paired_diffs_by_fold[f].append(diff)

    per_fold_summary = {}
    for f in range(6):
        per_fold_summary[f] = {
            'mean_diff': float(np.mean(fold_paired_diffs_by_fold[f])),
            'sd_diff': float(np.std(fold_paired_diffs_by_fold[f]))
        }

    overall_fold_mean = float(np.mean(fold_paired_diffs_all))
    overall_fold_sd = float(np.std(fold_paired_diffs_all))

    # Evaluate Rule 1.1:
    # (a) mean paired lenient difference >= +5.86 points
    # (b) mean paired FA/h is not worse than baseline by more than 10% (i.e. mean_paired_fa_pct <= +10.0%)
    cond_a = mean_paired_lenient >= 5.86
    cond_b = mean_paired_fa_pct <= 10.0

    adopted_recipe = 'b_rotation_70' if (cond_a and cond_b) else 'baseline_rotation_35'

    print("\n" + "=" * 80)
    print("RULE 1.1 CONFIRMATION EVALUATION RESULT")
    print("=" * 80)
    print(f"Condition (a): Mean paired lenient diff = {mean_paired_lenient:+.2f} points (Threshold: >= +5.86) -> {'PASS' if cond_a else 'FAIL'}")
    print(f"Condition (b): Mean paired FA/hr change = {mean_paired_fa_pct:+.2f}% (Threshold: <= +10.0%) -> {'PASS' if cond_b else 'FAIL'}")
    print(f"ADOPTED RECIPE: {adopted_recipe}")

    # Save JSON report
    out_json = {
        'b70_seed_runs': b70_seed_runs,
        'per_seed_table': per_seed_table,
        'mean_paired_lenient_diff': mean_paired_lenient,
        'mean_paired_fa_diff': mean_paired_fa_diff,
        'mean_paired_fa_pct_diff': mean_paired_fa_pct,
        'per_fold_summary': per_fold_summary,
        'overall_fold_mean_diff': overall_fold_mean,
        'overall_fold_sd_diff': overall_fold_sd,
        'rule_1_1_evaluation': {
            'condition_a_lenient_ge_5_86': bool(cond_a),
            'condition_b_fa_not_worse_10pct': bool(cond_b),
            'adopted_recipe': adopted_recipe
        }
    }

    os.makedirs('reports/v2/diagnostics', exist_ok=True)
    with open('reports/v2/diagnostics/confirmation_report.json', 'w') as f:
        json.dump(out_json, f, indent=2)

    # Generate reports/v2/confirmation.md
    md_content = f"""# Confirmation Report: Rotation Recipe Generalization (Step 2)

**Evaluation Date:** 2026-11-16  
**Adopted Recipe Determination:** `{adopted_recipe}`  
**Pre-Registered Rule 1.1 Criteria:**
1. Mean paired LOFO lenient sensitivity difference (b_rotation_70 minus baseline, same seed) $\\ge +5.86$ points.
2. Mean paired FA/h change not worse than baseline by more than $+10.0\\%$.

---

## 1. Three-Seed Comparison Table

| Seed | Recipe | Window ROC-AUC | Window PR-AUC | LOFO Lenient Sens (%) | LOFO Strict Sens (%) | LOFO Relaxed Sens (%) | Median Delay (s) | LOFO FA/hr |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for row in per_seed_table:
        md_content += f"| {row['seed']} | Baseline (±35°) | {row['base_window_roc']:.4f} | {row['base_window_pr']:.4f} | {row['base_lenient']:.1f}% | {row['base_strict']:.1f}% | {row['base_relaxed']:.1f}% | {row['base_median_delay']:.2f}s | {row['base_fa_hr']:.1f} |\n"
        md_content += f"| {row['seed']} | b_rotation_70 (±70°) | {row['b70_window_roc']:.4f} | {row['b70_window_pr']:.4f} | {row['b70_lenient']:.1f}% | {row['b70_strict']:.1f}% | {row['b70_relaxed']:.1f}% | {row['b70_median_delay']:.2f}s | {row['b70_fa_hr']:.1f} |\n"

    md_content += f"""
---

## 2. Per-Seed Paired Differences (b_rotation_70 minus Baseline)

| Seed | Paired Lenient Diff (pts) | Paired FA/hr Diff | FA/hr Relative Change (%) |
|:---:|:---:|:---:|:---:|
"""
    for row in per_seed_table:
        md_content += f"| {row['seed']} | {row['paired_lenient_diff']:+.2f} | {row['paired_fa_diff']:+.1f} | {row['paired_fa_pct_diff']:+.1f}% |\n"

    md_content += f"""| **Mean** | **{mean_paired_lenient:+.2f}** | **{mean_paired_fa_diff:+.1f}** | **{mean_paired_fa_pct:+.1f}%** |

---

## 3. Per-Fold Paired Differences Across Folds

| Fold Index | Subjects in Fold | Mean Paired Lenient Diff (pts) | SD Across Seeds (pts) |
|:---:|:---|:---:|:---:|
"""
    for f in range(6):
        subjs_str = ", ".join(map(str, DEV_FOLDS[f]))
        md_content += f"| Fold {f} | Subjs [{subjs_str}] | {per_fold_summary[f]['mean_diff']:+.2f} | {per_fold_summary[f]['sd_diff']:.2f} |\n"

    md_content += f"""| **All Folds** | **All 18 Dev Subjects** | **{overall_fold_mean:+.2f}** | **{overall_fold_sd:.2f}** |

---

## 4. Rule 1.1 Adoption Decision

- **Condition (a) - Sensitivity Gain:** Mean paired lenient sensitivity difference is **{mean_paired_lenient:+.2f} points** (Required: $\\ge +5.86$ points) $\\to$ **{'MET' if cond_a else 'NOT MET'}**.
- **Condition (b) - False Alert Rate:** Mean relative FA/hr change is **{mean_paired_fa_pct:+.2f}%** (Required: $\\le +10.0\\%$) $\\to$ **{'MET' if cond_b else 'NOT MET'}**.
- **Formal Adoption Decision:**
  > **{'`b_rotation_70` (±70° rotation augmentation) is formally ADOPTED.' if (cond_a and cond_b) else 'Baseline recipe (±35° rotation augmentation) is RETAINED.'}**
"""

    with open('reports/v2/confirmation.md', 'w') as f:
        f.write(md_content)

    print("Wrote reports/v2/confirmation.md and confirmation_report.json successfully.")


if __name__ == '__main__':
    run_confirmation()

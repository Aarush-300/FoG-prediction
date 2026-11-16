"""
Computes nested LOFO (Leave-One-Fold-Out) operating point selection on dev data.
For each fold f in 0..5:
  1. Tune presets on the 5 other folds (aggregated across seeds 42, 43, 44)
  2. Evaluate the chosen preset on held-out fold f (across seeds 42, 43, 44)
Aggregate held-out performance across all 6 folds to obtain Nested LOFO metrics.
Compute the true optimism gap: In-Sample vs Nested LOFO.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_splits import DEV_FOLDS
from train_cv import load_dev_data
from run_phase2_experiments import build_dev_trials

class FastFSM:
    __slots__ = ['high', 'low', 'confirm', 'cooldown_ms', 'rearm_policy',
                 'state', 'smoothed_prob', 'consecutive_count', 'last_alert_time', 'alpha']
    def __init__(self, high, low, confirm, cooldown_ms, alpha, rearm_policy):
        self.high = high
        self.low = low
        self.confirm = confirm
        self.cooldown_ms = cooldown_ms
        self.alpha = alpha
        self.rearm_policy = rearm_policy
        self.reset()

    def reset(self):
        self.state = 0
        self.smoothed_prob = 0.0
        self.consecutive_count = 0
        self.last_alert_time = -999999

    def update(self, raw_prob, now_ms):
        self.smoothed_prob = (self.alpha * raw_prob) + ((1.0 - self.alpha) * self.smoothed_prob)
        fired_alert = False

        if self.state == 0:  # NORMAL
            if self.smoothed_prob >= self.high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirm:
                    self.state = 2  # ACTIVE_ALERT
                    self.last_alert_time = now_ms
                    fired_alert = True
                else:
                    self.state = 1  # WARNING
            else:
                self.consecutive_count = 0

        elif self.state == 1:  # WARNING
            if self.smoothed_prob >= self.high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirm:
                    self.state = 2  # ACTIVE_ALERT
                    self.last_alert_time = now_ms
                    fired_alert = True
            elif self.smoothed_prob < self.low:
                self.consecutive_count = 0
                self.state = 0  # NORMAL

        elif self.state == 2:  # ACTIVE_ALERT
            self.state = 3  # COOLDOWN

        elif self.state == 3:  # COOLDOWN
            time_since = now_ms - self.last_alert_time
            if time_since >= self.cooldown_ms:
                if self.rearm_policy == 'timeout':
                    self.consecutive_count = 0
                    self.state = 0
                elif self.rearm_policy == 'timeout_and_low':
                    if self.smoothed_prob < self.low:
                        self.consecutive_count = 0
                        self.state = 0

        return fired_alert

def eval_trials_fsm(trials, config):
    fsm = FastFSM(
        high=config['high'],
        low=config['low'],
        confirm=config['confirm'],
        cooldown_ms=config['cooldown_ms'],
        alpha=config['alpha'],
        rearm_policy=config['rearm_policy']
    )

    total_false_alerts = 0
    total_normal_s = 0.0
    total_episodes = 0
    caught_lenient = 0
    caught_strict = 0
    caught_relaxed = 0

    for tr in trials:
        fsm.reset()
        alerts = []
        for p, t_end in zip(tr['probs'], tr['end_times']):
            now_ms = int(round(t_end * 1000))
            if fsm.update(p, now_ms):
                alerts.append(t_end)

        for a in alerts:
            is_true = any(ep['onset'] - 3.0 <= a <= ep['offset'] for ep in tr['episodes'])
            if not is_true:
                total_false_alerts += 1
        total_normal_s += tr['normal_duration_s']

        for ep in tr['episodes']:
            total_episodes += 1
            if any(ep['onset'] - 3.0 <= a <= ep['offset'] for a in alerts):
                caught_lenient += 1
            if any(ep['onset'] - 3.0 <= a <= ep['onset'] - 0.5 for a in alerts):
                caught_strict += 1
            if any(ep['onset'] - 3.0 <= a <= ep['onset'] for a in alerts):
                caught_relaxed += 1

    norm_hr = total_normal_s / 3600.0
    fa_hr = float(total_false_alerts / norm_hr) if norm_hr > 0 else 0.0
    lenient_sens = float(caught_lenient / total_episodes) if total_episodes > 0 else 0.0
    strict_sens = float(caught_strict / total_episodes) if total_episodes > 0 else 0.0
    relaxed_sens = float(caught_relaxed / total_episodes) if total_episodes > 0 else 0.0

    return {
        'fa_hr': fa_hr,
        'lenient_sens': lenient_sens,
        'strict_sens': strict_sens,
        'relaxed_sens': relaxed_sens,
        'total_false_alerts': total_false_alerts,
        'total_normal_hours': norm_hr,
        'total_episodes': total_episodes,
        'caught_lenient': caught_lenient,
        'caught_strict': caught_strict,
        'caught_relaxed': caught_relaxed
    }

def main():
    X_dev_raw, y_bin_dev, y_multi_dev, meta_dev = load_dev_data()
    seeds = [42, 43, 44]
    trials_by_seed = {}

    for s in seeds:
        full_int8_probs = np.zeros(len(meta_dev), dtype=np.float32)
        for f in range(6):
            d = np.load(f'reports/v2/oof/b_rotation_35_seed{s}_fold{f}_int8.npz')
            idx = d['val_indices']
            full_int8_probs[idx] = d['int8_probs']
        trials = build_dev_trials(full_int8_probs, meta_dev)
        trials_by_seed[s] = trials

    # Define the 5,300 base configs (recue=0)
    confirms = [1, 2]
    alphas = [0.60, 0.70, 0.80, 0.90, 1.00]
    highs = [0.35, 0.40, 0.45, 0.49, 0.50, 0.55, 0.60, 0.65, 0.70]
    lows = [0.15, 0.20, 0.25, 0.29, 0.30, 0.35]
    cds = [2000, 3000, 4000, 5000, 6000]
    rearms = ['timeout', 'timeout_and_low']

    grid = []
    for c in confirms:
        for a in alphas:
            for h in highs:
                for l in lows:
                    if l >= h:
                        continue
                    for cd in cds:
                        for r_pol in rearms:
                            grid.append({
                                'confirm': c,
                                'alpha': a,
                                'high': h,
                                'low': l,
                                'cooldown_ms': cd,
                                'rearm_policy': r_pol,
                                'recue_interval_s': 0
                            })

    print(f"Total base configs in grid: {len(grid)}")

    # Pre-split trials by fold
    # trials_by_seed_fold[s][f] = list of trials
    trials_by_seed_fold = {s: {f: [] for f in range(6)} for s in seeds}
    for s in seeds:
        for tr in trials_by_seed[s]:
            trials_by_seed_fold[s][tr['fold_idx']].append(tr)

    # For each fold f held out:
    # Training folds: all k != f
    nested_results = {
        'LOW_ALARM': {'lofo_caught': 0, 'lofo_tot': 0, 'lofo_fa': 0, 'lofo_norm_hr': 0.0, 'chosen_configs': []},
        'BALANCED': {'lofo_caught': 0, 'lofo_tot': 0, 'lofo_fa': 0, 'lofo_norm_hr': 0.0, 'chosen_configs': []}
    }

    # Evaluate per fold
    print("\n--- Running Nested LOFO Selection across 6 Folds ---")
    t0 = time.time()

    for held_out_fold in range(6):
        print(f"\nFold {held_out_fold} held out (training on folds {[k for k in range(6) if k != held_out_fold]}):")
        train_trials_by_seed = {s: [] for s in seeds}
        test_trials_by_seed = {s: [] for s in seeds}
        for s in seeds:
            for f in range(6):
                if f == held_out_fold:
                    test_trials_by_seed[s].extend(trials_by_seed_fold[s][f])
                else:
                    train_trials_by_seed[s].extend(trials_by_seed_fold[s][f])

        # Evaluate grid on train_trials_by_seed
        # To make it fast, evaluate each config across seeds
        # We can score each config
        cfg_train_scores = []
        for cfg in grid:
            s_res = [eval_trials_fsm(train_trials_by_seed[s], cfg) for s in seeds]
            mean_lenient = np.mean([r['lenient_sens'] for r in s_res])
            mean_fa = np.mean([r['fa_hr'] for r in s_res])
            cfg_train_scores.append((cfg, mean_lenient, mean_fa))

        # Select LOW_ALARM: max lenient sens with mean_fa <= 30.0 (or fallback to min mean_fa)
        low_candidates = [item for item in cfg_train_scores if item[2] <= 30.0]
        if low_candidates:
            low_candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            chosen_low_cfg = low_candidates[0][0]
        else:
            cfg_train_scores.sort(key=lambda x: x[2])
            chosen_low_cfg = cfg_train_scores[0][0]

        # Select BALANCED: min mean_fa with mean_lenient >= 0.70 (or fallback to max mean_lenient)
        bal_candidates = [item for item in cfg_train_scores if item[1] >= 0.70]
        if bal_candidates:
            bal_candidates.sort(key=lambda x: x[2])
            chosen_bal_cfg = bal_candidates[0][0]
        else:
            cfg_train_scores.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            chosen_bal_cfg = cfg_train_scores[0][0]

        print(f"  Chosen LOW_ALARM cfg: {chosen_low_cfg}")
        print(f"  Chosen BALANCED cfg:  {chosen_bal_cfg}")

        nested_results['LOW_ALARM']['chosen_configs'].append(chosen_low_cfg)
        nested_results['BALANCED']['chosen_configs'].append(chosen_bal_cfg)

        # Evaluate on held-out fold
        for preset, cfg in [('LOW_ALARM', chosen_low_cfg), ('BALANCED', chosen_bal_cfg)]:
            for s in seeds:
                res = eval_trials_fsm(test_trials_by_seed[s], cfg)
                nested_results[preset]['lofo_caught'] += res['caught_lenient']
                nested_results[preset]['lofo_tot'] += res['total_episodes']
                nested_results[preset]['lofo_fa'] += res['total_false_alerts']
                nested_results[preset]['lofo_norm_hr'] += res['total_normal_hours']

    print(f"\nNested LOFO search finished in {time.time() - t0:.1f}s.")

    # Compute Final Nested LOFO numbers and compare to In-Sample
    # In-Sample numbers from operating_point_report.json
    with open('reports/v2/diagnostics/operating_point_report.json') as f:
        diag = json.load(f)

    report_out = {}
    for preset in ['LOW_ALARM', 'BALANCED']:
        p_data = diag['low_alarm_preset'] if preset == 'LOW_ALARM' else diag['balanced_preset']
        in_sample_sens = p_data['mean_lenient']  # in percent
        in_sample_fa = p_data['mean_fa_hr']

        lofo_sens = (nested_results[preset]['lofo_caught'] / nested_results[preset]['lofo_tot']) * 100.0
        lofo_fa = nested_results[preset]['lofo_fa'] / nested_results[preset]['lofo_norm_hr']

        gap_sens = in_sample_sens - lofo_sens
        gap_fa = lofo_fa - in_sample_fa

        report_out[preset] = {
            'in_sample_lenient': round(in_sample_sens, 2),
            'lofo_lenient': round(lofo_sens, 2),
            'gap_lenient': round(gap_sens, 2),
            'in_sample_fa': round(in_sample_fa, 2),
            'lofo_fa': round(lofo_fa, 2),
            'gap_fa': round(gap_fa, 2),
            'chosen_configs_per_fold': nested_results[preset]['chosen_configs']
        }
        print(f"\n=== {preset} RESULTS ===")
        print(f"  In-Sample Lenient Sens: {in_sample_sens:.2f}% | Nested LOFO: {lofo_sens:.2f}% | Gap: {gap_sens:+.2f}%")
        print(f"  In-Sample FA/hr:        {in_sample_fa:.2f}    | Nested LOFO: {lofo_fa:.2f}    | Gap: {gap_fa:+.2f}")

    with open('reports/v2/diagnostics/nested_lofo_results.json', 'w') as f:
        json.dump(report_out, f, indent=2)
    print("\nSaved reports/v2/diagnostics/nested_lofo_results.json")

if __name__ == '__main__':
    main()

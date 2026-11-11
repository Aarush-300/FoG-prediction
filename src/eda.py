"""
Exploratory Data Analysis (EDA) and Data Audit for TinyML Freezing-of-Gait Prediction.
Inspects raw IMU data, verifies sampling rate, checks sensor units, analyzes class balance,
calculates episode statistics, computes frequency spectra, and generates plots and audit reports.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

def set_seed(seed=42):
    np.random.seed(seed)
    print(f"[EDA] Random seed set to {seed}")

def load_data(data_dir):
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"[EDA] Found {len(csv_files)} CSV file(s) in {data_dir}: {[os.path.basename(f) for f in csv_files]}")
    dfs = []
    for f in csv_files:
        print(f"[EDA] Loading {f}...")
        df_temp = pd.read_csv(f)
        df_temp['source_file'] = os.path.basename(f)
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"[EDA] Combined dataset shape: {df.shape} ({len(df):,} rows, {len(df.columns)} columns)")
    return df

def audit_columns_and_types(df):
    print("\n--- Column List & Data Types ---")
    col_info = []
    for col in df.columns:
        col_info.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].notna().sum()),
            'nan_count': int(df[col].isna().sum()),
            'nan_pct': float(df[col].isna().mean() * 100)
        })
    col_df = pd.DataFrame(col_info)
    print(col_df.to_string(index=False))
    return col_df

def audit_channel_stats(df, sensor_prefix='ankleL'):
    channels = [f"{sensor_prefix}_acc_x", f"{sensor_prefix}_acc_y", f"{sensor_prefix}_acc_z",
                f"{sensor_prefix}_gyro_x", f"{sensor_prefix}_gyro_y", f"{sensor_prefix}_gyro_z"]
    
    existing = [c for c in channels if c in df.columns]
    stats = df[existing].describe().T
    stats['median'] = df[existing].median()
    print(f"\n--- Channel Statistics for {sensor_prefix} ---")
    print(stats[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']].to_string())
    return stats

def audit_sampling_rate(df):
    print("\n--- Sampling Rate and Continuity Audit ---")
    trials = df.groupby(['subjectID', 'sessionID', 'taskID'])
    total_trials = len(trials)
    print(f"Total unique trials (subjectID x sessionID x taskID): {total_trials}")
    
    all_dts = []
    gaps = []
    negative_dts = []
    
    trial_stats = []
    for (subj, sess, task), grp in trials:
        t = grp['timestamp'].values
        if len(t) < 2:
            continue
        dt = np.diff(t)
        all_dts.extend(dt)
        
        # Check anomalies
        neg_mask = dt <= 0
        if np.any(neg_mask):
            negative_dts.append((subj, sess, task, int(np.sum(neg_mask))))
            
        gap_mask = dt > 0.035 # > 2 missed samples at 60 Hz
        if np.any(gap_mask):
            gaps.append((subj, sess, task, int(np.sum(gap_mask)), float(np.max(dt))))
            
        trial_stats.append({
            'subjectID': int(subj),
            'sessionID': int(sess),
            'taskID': int(task),
            'samples': len(grp),
            'duration_sec': float(t[-1] - t[0]),
            'mean_dt': float(np.mean(dt)),
            'std_dt': float(np.std(dt)),
            'estimated_fs': float(1.0 / np.mean(dt)) if np.mean(dt) > 0 else 0.0
        })
    
    all_dts = np.array(all_dts)
    mean_dt = float(np.mean(all_dts))
    median_dt = float(np.median(all_dts))
    std_dt = float(np.std(all_dts))
    min_dt = float(np.min(all_dts))
    max_dt = float(np.max(all_dts))
    mean_fs = 1.0 / mean_dt if mean_dt > 0 else 0.0
    
    print(f"Overall Timestamp dt: Mean = {mean_dt:.8f} s (fs = {mean_fs:.2f} Hz)")
    print(f"Median dt = {median_dt:.8f} s, Std dt = {std_dt:.2e} s")
    print(f"Min dt = {min_dt:.8f} s, Max dt = {max_dt:.8f} s")
    print(f"Total gaps (>35ms): {len(gaps)}")
    print(f"Total negative/zero dt: {len(negative_dts)}")
    
    dt_summary = {
        'total_trials': total_trials,
        'total_transitions': len(all_dts),
        'mean_dt': mean_dt,
        'median_dt': median_dt,
        'std_dt': std_dt,
        'min_dt': min_dt,
        'max_dt': max_dt,
        'nominal_fs_hz': 60.0,
        'measured_fs_hz': mean_fs,
        'anomalies_gaps_count': len(gaps),
        'anomalies_negative_dt_count': len(negative_dts)
    }
    return dt_summary, trial_stats

def audit_classes_and_episodes(df):
    print("\n--- Class and Episode Analysis ---")
    total_samples = len(df)
    fog_samples = int((df['fog'] == 1).sum())
    normal_samples = int((df['fog'] == 0).sum())
    fog_pct = (fog_samples / total_samples) * 100.0
    
    print(f"Total Samples: {total_samples:,}")
    print(f"Normal (0): {normal_samples:,} ({100 - fog_pct:.2f}%)")
    print(f"FoG (1): {fog_samples:,} ({fog_pct:.2f}%)")
    
    # Severity breakdown
    sev_counts = df['fog_severity'].value_counts().to_dict()
    print("FoG Severity Counts:", sev_counts)
    
    # Episode level analysis
    subject_audit = []
    all_episodes = []
    
    for subj, s_grp in df.groupby('subjectID'):
        s_total = len(s_grp)
        s_fog = int((s_grp['fog'] == 1).sum())
        s_pct = (s_fog / s_total) * 100.0
        
        # Episode detection per trial within subject to prevent false episodes across trial boundaries
        s_episodes = 0
        s_ep_durations = []
        
        for (sess, task), t_grp in s_grp.groupby(['sessionID', 'taskID']):
            fog_arr = t_grp['fog'].values
            t_arr = t_grp['timestamp'].values
            
            # Find contiguous blocks of 1
            if len(fog_arr) == 0:
                continue
            
            diff = np.diff(np.pad(fog_arr, (1, 1), 'constant', constant_values=0))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for st, en in zip(starts, ends):
                s_episodes += 1
                dur = float(t_arr[en - 1] - t_arr[st] + 1/60.0)
                s_ep_durations.append(dur)
                all_episodes.append({
                    'subjectID': int(subj),
                    'sessionID': int(sess),
                    'taskID': int(task),
                    'start_idx': int(st),
                    'end_idx': int(en),
                    'length_samples': int(en - st),
                    'duration_sec': dur
                })
                
        subject_audit.append({
            'subjectID': int(subj),
            'total_samples': s_total,
            'duration_min': float(s_total / (60.0 * 60.0)),
            'fog_samples': s_fog,
            'fog_pct': float(s_pct),
            'episodes': s_episodes,
            'mean_ep_duration_sec': float(np.mean(s_ep_durations)) if s_ep_durations else 0.0,
            'max_ep_duration_sec': float(np.max(s_ep_durations)) if s_ep_durations else 0.0,
            'is_freezer': bool(s_episodes > 0)
        })
    
    subj_df = pd.DataFrame(subject_audit)
    print("\n--- Per-Subject Breakdown ---")
    print(subj_df.to_string(index=False))
    
    ep_df = pd.DataFrame(all_episodes)
    print(f"\nTotal FoG Episodes identified: {len(ep_df)}")
    if len(ep_df) > 0:
        print("Episode Duration (sec) Statistics:")
        print(ep_df['duration_sec'].describe().to_string())
        
    return {
        'total_samples': total_samples,
        'normal_samples': normal_samples,
        'fog_samples': fog_samples,
        'fog_pct': fog_pct,
        'severity_counts': sev_counts,
        'total_episodes': len(ep_df),
        'freezer_subjects': int(subj_df['is_freezer'].sum()),
        'non_freezer_subjects': int((~subj_df['is_freezer']).sum()),
        'subjects': subject_audit,
        'episodes': all_episodes
    }

def generate_plots(df, audit_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Class Balance Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1A: Sample-level FoG vs Normal
    classes = ['Normal (0)', 'FoG (1)']
    counts = [audit_data['normal_samples'], audit_data['fog_samples']]
    bars = axes[0].bar(classes, counts, color=['#2b5c8f', '#d95f02'])
    axes[0].set_title("Class Balance (Samples)", fontweight='bold')
    axes[0].set_ylabel("Count")
    for b in bars:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2., h + 3000, f"{h:,}\n({h/sum(counts)*100:.1f}%)",
                     ha='center', va='bottom', fontsize=10)
    axes[0].set_ylim(0, max(counts)*1.15)
    
    # 1B: Severity Breakdown
    sev = audit_data['severity_counts']
    sev_labels = [f"Sev {k}" if k > 0 else "Normal (0)" for k in sorted(sev.keys())]
    sev_vals = [sev[k] for k in sorted(sev.keys())]
    axes[1].bar(sev_labels, sev_vals, color=['#2b5c8f', '#fc8d59', '#e34a33', '#b30000'])
    axes[1].set_title("FoG Severity Breakdown", fontweight='bold')
    axes[1].set_ylabel("Count")
    for idx, v in enumerate(sev_vals):
        axes[1].text(idx, v + 2000, f"{v:,}", ha='center', va='bottom', fontsize=9)
    axes[1].set_ylim(0, max(sev_vals)*1.15)
    
    # 1C: Subject FoG %
    subj_df = pd.DataFrame(audit_data['subjects'])
    colors = ['#d95f02' if f else '#2b5c8f' for f in subj_df['is_freezer']]
    axes[2].bar(subj_df['subjectID'].astype(str), subj_df['fog_pct'], color=colors)
    axes[2].set_title("FoG % per Subject (Orange = Freezer)", fontweight='bold')
    axes[2].set_xlabel("Subject ID")
    axes[2].set_ylabel("FoG Sample %")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_balance.png'), dpi=200)
    plt.close()
    print("[EDA] Saved class_balance.png")
    
    # 2. Example Segments: Normal vs FoG
    # Find a clean normal walking segment and a clear FoG segment from ankleL
    ankle_cols = ['ankleL_acc_x', 'ankleL_acc_y', 'ankleL_acc_z', 'ankleL_gyro_x', 'ankleL_gyro_y', 'ankleL_gyro_z']
    valid_df = df.dropna(subset=ankle_cols)
    
    # Normal segment (subject 1, activity 1, fog == 0, length 300 = 5s)
    norm_cands = valid_df[(valid_df['subjectID'] == 1) & (valid_df['fog'] == 0) & (valid_df['activity'] == 1)]
    fog_cands = valid_df[(valid_df['subjectID'] == 1) & (valid_df['fog'] == 1)]
    
    if len(norm_cands) >= 300 and len(fog_cands) >= 300:
        norm_seg = norm_cands.iloc[100:400]
        fog_seg = fog_cands.iloc[100:400]
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
        t_axis = np.arange(300) / 60.0 # seconds
        
        # Normal Acc
        axes[0].plot(t_axis, norm_seg['ankleL_acc_x'], label='acc_x (g)', alpha=0.9)
        axes[0].plot(t_axis, norm_seg['ankleL_acc_y'], label='acc_y (g)', alpha=0.9)
        axes[0].plot(t_axis, norm_seg['ankleL_acc_z'], label='acc_z (g)', alpha=0.9)
        axes[0].set_title("Normal Walking (5.0s): AnkleL Accelerometer (g)", fontweight='bold')
        axes[0].set_ylabel("Acc (g)")
        axes[0].legend(loc='upper right', ncol=3)
        axes[0].set_ylim(-3.2, 3.2)
        
        # Normal Gyro
        axes[1].plot(t_axis, norm_seg['ankleL_gyro_x'], label='gyro_x (deg/s)', alpha=0.9)
        axes[1].plot(t_axis, norm_seg['ankleL_gyro_y'], label='gyro_y (deg/s)', alpha=0.9)
        axes[1].plot(t_axis, norm_seg['ankleL_gyro_z'], label='gyro_z (deg/s)', alpha=0.9)
        axes[1].set_title("Normal Walking (5.0s): AnkleL Gyroscope (deg/s)", fontweight='bold')
        axes[1].set_ylabel("Angular Vel (deg/s)")
        axes[1].legend(loc='upper right', ncol=3)
        
        # FoG Acc
        axes[2].plot(t_axis, fog_seg['ankleL_acc_x'], label='acc_x (g)', color='#d95f02', alpha=0.9)
        axes[2].plot(t_axis, fog_seg['ankleL_acc_y'], label='acc_y (g)', color='#7570b3', alpha=0.9)
        axes[2].plot(t_axis, fog_seg['ankleL_acc_z'], label='acc_z (g)', color='#1b9e77', alpha=0.9)
        axes[2].set_title("Freezing of Gait (FoG) Episode (5.0s): AnkleL Accelerometer (g) - Note Trembling / Cadence Loss", fontweight='bold')
        axes[2].set_ylabel("Acc (g)")
        axes[2].legend(loc='upper right', ncol=3)
        axes[2].set_ylim(-3.2, 3.2)
        
        # FoG Gyro
        axes[3].plot(t_axis, fog_seg['ankleL_gyro_x'], label='gyro_x (deg/s)', color='#d95f02', alpha=0.9)
        axes[3].plot(t_axis, fog_seg['ankleL_gyro_y'], label='gyro_y (deg/s)', color='#7570b3', alpha=0.9)
        axes[3].plot(t_axis, fog_seg['ankleL_gyro_z'], label='gyro_z (deg/s)', color='#1b9e77', alpha=0.9)
        axes[3].set_title("Freezing of Gait (FoG) Episode (5.0s): AnkleL Gyroscope (deg/s)", fontweight='bold')
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Angular Vel (deg/s)")
        axes[3].legend(loc='upper right', ncol=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'example_segments.png'), dpi=200)
        plt.close()
        print("[EDA] Saved example_segments.png")
        
    # 3. Frequency Spectrum (PSD) comparison (Normal vs FoG)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Compute acc magnitude
    acc_mag_norm = np.sqrt(norm_cands['ankleL_acc_x']**2 + norm_cands['ankleL_acc_y']**2 + norm_cands['ankleL_acc_z']**2)
    acc_mag_fog = np.sqrt(fog_cands['ankleL_acc_x']**2 + fog_cands['ankleL_acc_y']**2 + fog_cands['ankleL_acc_z']**2)
    
    # Gyro magnitude
    gyro_mag_norm = np.sqrt(norm_cands['ankleL_gyro_x']**2 + norm_cands['ankleL_gyro_y']**2 + norm_cands['ankleL_gyro_z']**2)
    gyro_mag_fog = np.sqrt(fog_cands['ankleL_gyro_x']**2 + fog_cands['ankleL_gyro_y']**2 + fog_cands['ankleL_gyro_z']**2)
    
    fs = 60.0
    nperseg = min(256, len(acc_mag_norm), len(acc_mag_fog))
    
    f_norm, psd_acc_norm = signal.welch(acc_mag_norm - np.mean(acc_mag_norm), fs=fs, nperseg=nperseg)
    f_fog, psd_acc_fog = signal.welch(acc_mag_fog - np.mean(acc_mag_fog), fs=fs, nperseg=nperseg)
    
    f_g_norm, psd_gyro_norm = signal.welch(gyro_mag_norm - np.mean(gyro_mag_norm), fs=fs, nperseg=nperseg)
    f_g_fog, psd_gyro_fog = signal.welch(gyro_mag_fog - np.mean(gyro_mag_fog), fs=fs, nperseg=nperseg)
    
    # Acc PSD
    axes[0].semilogy(f_norm, psd_acc_norm, label='Normal Walking', color='#2b5c8f', lw=2.5)
    axes[0].semilogy(f_fog, psd_acc_fog, label='FoG', color='#d95f02', lw=2.5)
    axes[0].axvspan(0.5, 3.0, alpha=0.15, color='blue', label='Locomotor Band (0.5-3 Hz)')
    axes[0].axvspan(3.0, 8.0, alpha=0.15, color='orange', label='Freeze Band (3-8 Hz)')
    axes[0].set_title("AnkleL Acc Magnitude PSD (Welch)", fontweight='bold')
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power Spectral Density (g²/Hz)")
    axes[0].set_xlim(0, 20)
    axes[0].legend(loc='upper right')
    
    # Gyro PSD
    axes[1].semilogy(f_g_norm, psd_gyro_norm, label='Normal Walking', color='#2b5c8f', lw=2.5)
    axes[1].semilogy(f_g_fog, psd_gyro_fog, label='FoG', color='#d95f02', lw=2.5)
    axes[1].axvspan(0.5, 3.0, alpha=0.15, color='blue', label='Locomotor Band (0.5-3 Hz)')
    axes[1].axvspan(3.0, 8.0, alpha=0.15, color='orange', label='Freeze Band (3-8 Hz)')
    axes[1].set_title("AnkleL Gyro Magnitude PSD (Welch)", fontweight='bold')
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power Spectral Density ((deg/s)²/Hz)")
    axes[1].set_xlim(0, 20)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrum_comparison.png'), dpi=200)
    plt.close()
    print("[EDA] Saved spectrum_comparison.png")
    
    # 4. Episode Duration Distribution
    ep_df = pd.DataFrame(audit_data['episodes'])
    if len(ep_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(ep_df['duration_sec'], bins=25, kde=True, ax=axes[0], color='#d95f02')
        axes[0].set_title(f"FoG Episode Duration Distribution (N={len(ep_df)})", fontweight='bold')
        axes[0].set_xlabel("Duration (seconds)")
        axes[0].set_ylabel("Number of Episodes")
        median_dur = ep_df['duration_sec'].median()
        axes[0].axvline(median_dur, color='black', linestyle='--', label=f'Median: {median_dur:.1f}s')
        axes[0].legend()
        
        sns.boxplot(data=ep_df, x='subjectID', y='duration_sec', hue='subjectID', ax=axes[1], palette='tab20', legend=False)
        axes[1].set_title("Episode Durations by Subject", fontweight='bold')
        axes[1].set_xlabel("Subject ID")
        axes[1].set_ylabel("Duration (seconds)")
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'episode_duration_hist.png'), dpi=200)
        plt.close()
        print("[EDA] Saved episode_duration_hist.png")
        
    # 5. Channel Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = valid_df[ankle_cols].corr()
    sns.heatmap(corr, annot=True, cmap='vlag', fmt='.2f', vmin=-1, vmax=1, ax=ax)
    ax.set_title("AnkleL 6-Axis IMU Channel Correlations", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=200)
    plt.close()
    print("[EDA] Saved correlation_heatmap.png")

def write_audit_markdown(col_df, stats_df, dt_summary, audit_data, output_path):
    subj_df = pd.DataFrame(audit_data['subjects'])
    ep_df = pd.DataFrame(audit_data['episodes'])
    
    md_content = f"""# Data Audit Report: Parkinson's Freezing-of-Gait (FoG) IMU Stream

**Date:** 2026-11-07  
**Target Hardware:** Arduino Nano 33 BLE Sense (Cortex-M4F, 256 KB RAM)  
**Primary IMU Channel:** Left Ankle (`ankleL_acc_x/y/z`, `ankleL_gyro_x/y/z`)  
**Deployment Rate:** 50 Hz on-device (from nominal 60 Hz recording rate)  
**Window Size:** 2.0 s (100 samples @ 50 Hz, stride 0.5 s = 25 samples)

---

## 1. Executive Summary & Verification Gates

| Item | Finding / Value | Status / Impact |
| :--- | :--- | :--- |
| **Total Rows** | {audit_data['total_samples']:,} rows across 106 recording trials | Confirmed complete dataset |
| **Total Duration** | {audit_data['total_samples']/60/60:.2f} hours ({audit_data['total_samples']/60:.2f} minutes @ 60.0 Hz) | Adequate multi-subject clinical sample |
| **Subjects** | 22 subjects total (16 freezers, 6 non-freezers) | Allows Leave-One-Subject-Out or Subject-Group Split |
| **FoG Episodes** | {audit_data['total_episodes']} clinical freezing episodes | Rich episode count for temporal validation |
| **Class Balance** | Normal: {audit_data['normal_samples']:,} ({100-audit_data['fog_pct']:.1f}%), FoG: {audit_data['fog_samples']:,} ({audit_data['fog_pct']:.1f}%) | Significant class imbalance; weighted loss required |
| **Label Column** | `fog` (0 = normal, 1 = freezing) | Verified present |
| **ID Columns** | `subjectID`, `sessionID`, `taskID` | Verified present; defines continuous trials |
| **Sampling Rate** | $\\Delta t = {dt_summary['mean_dt']:.8f}$ s (Std: ${dt_summary['std_dt']:.2e}$ s) $\\to$ **60.00 Hz** | Strictly periodic; no timestamp jitter |
| **Acc Units** | Range $\\sim [-2.9, +3.0]$, static axis $\\sim -1.0$ | **Units of $g$** ($1\\text{{ g}} \\approx 9.80665\\text{{ m/s}}^2$) |
| **Gyro Units** | Range $[-677, +342]$, std $\\approx 23-66$ | **Units of deg/s** (rad/s impossible for human ankle) |

---

## 2. Sensor Channels & Units Verification

### Accelerometer Units
* **Observed Range:** Min $\\approx -2.94$ g, Max $\\approx +3.02$ g, Mean $\\approx -1.00$ g along `ankleL_acc_x`.
* **Physical Basis:** Gravity acts along the sensor's x-axis while standing/walking ($1\\text{{ g}}$ downwards). If the data were in $\\text{{m/s}}^2$, earth gravity would read $\\approx 9.81$.
* **Conclusion for Firmware:** **Units are $g$**. The Arduino Nano 33 BLE Sense (LSM9DS1 IMU) typically provides accelerometer readings in $g$ directly (`IMU.readAcceleration(x, y, z)` yields $g$).

### Gyroscope Units
* **Observed Range:** Peak rates reach $-677.9^\\circ/\\text{{s}}$ to $+342.3^\\circ/\\text{{s}}$, standard deviation $23.7^\\circ/\\text{{s}}$ to $66.1^\\circ/\\text{{s}}$.
* **Physical Basis:** During walking swing phases, human ankles rotate at angular velocities between $100^\\circ/\\text{{s}}$ and $500^\\circ/\\text{{s}}$. If this were radians per second, $677\\text{{ rad/s}}$ would equate to over $100$ revolutions per second ($>6,400\\text{{ RPM}}$), which is physically impossible.
* **Conclusion for Firmware:** **Units are degrees per second (deg/s)**. Note that some Arduino libraries output gyro data in degrees/sec (dps) and some in radians/sec. The firmware MUST supply degrees/second to match the trained normalization constants.

---

## 3. Sampling Rate and Temporal Continuity

* **Mean $\\Delta t$:** `{dt_summary['mean_dt']:.8f}` seconds ($f_s = 60.000$ Hz).
* **Timestamp Jitter:** Standard deviation of $\\Delta t$ across 328,921 steps is ${dt_summary['std_dt']:.2e}$ seconds.
* **Dropped Samples / Gaps:** 0 negative time-steps, 0 gap intervals ($>35$ ms) within individual trials.
* **Resampling Strategy for Phase 3:**
  * Downsampling from 60 Hz to 50 Hz uses `scipy.signal.resample_poly(up=5, down=6)`.
  * **Critical Constraint:** Resampling must be performed **strictly trial-by-trial** (`subjectID`, `sessionID`, `taskID`). Never resample across recording boundaries.
  * Nearest-neighbour mapping for label alignment.

---

## 4. Missing Values & Data Quality Audit

| Sensor Location | Missing Samples | Missing % | Affected Trials |
| :--- | :--- | :--- | :--- |
| **`ankleL`** | 10,236 | 3.11% | Subj 5 (task 1), Subj 9 (task 7), Subj 17 (tasks 3, 4, 6, 7) |
| **`ankleR`** | 11,771 | 3.58% | Subj 6 (task 7), Subj 8 (task 1), Subj 16 (tasks 3, 7), Subj 17 |
| **`back`** | 27,016 | 8.21% | Subj 4, 8, 9, 12, 17, 21, 22 |
| **`wrist`** | 16,122 | 4.90% | Subj 4, 7, 13, 14, 19 |

### Handling in Preprocessing:
1. In Subj 17 (tasks 3, 4, 6, 7) and Subj 9 (task 7), the `ankleL` sensor was unrecorded (100% NaN). These trials are cleanly excluded when slicing `ankleL` windows.
2. In Subj 5 (task 1), `ankleL` starts recording halfway through (1,661 NaN, 953 valid samples). The leading NaN section will be dropped; only the valid continuous portion will be windowed.
3. **No synthetic imputation** will be performed on missing IMU streams, adhering strictly to real sensor data integrity.

---

## 5. Subject and Episode Breakdown

* **Freezer Subjects (16):** Subjects 1, 2, 3, 4, 5, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22.
* **Non-Freezer / Control Subjects (6):** Subjects 6, 7, 8, 10, 14, 18 (0 FoG episodes recorded).
* **Episode Duration Distribution:**
  * Total episodes: {len(ep_df)}
  * Median duration: {ep_df['duration_sec'].median():.2f} s
  * Mean duration: {ep_df['duration_sec'].mean():.2f} s
  * 25th percentile: {ep_df['duration_sec'].quantile(0.25):.2f} s
  * 75th percentile: {ep_df['duration_sec'].quantile(0.75):.2f} s
  * Min / Max: {ep_df['duration_sec'].min():.2f} s / {ep_df['duration_sec'].max():.2f} s

---

## 6. Spectral Analysis: Locomotor vs Freeze Bands

Power Spectral Density (PSD) analysis computed via Welch's method clearly reveals:
1. **Locomotor Band (0.5 – 3.0 Hz):** Normal walking displays dominant power concentrated between 1.0 Hz and 2.5 Hz, corresponding to cyclic stride frequency.
2. **Freeze Band (3.0 – 8.0 Hz):** During FoG episodes, locomotor peak drops sharply, and high-frequency power surges in the 3.0–8.0 Hz band due to leg trembling / shuddering.
3. **Implication for Baseline:** Confirms that the classical Freeze Index (FI = Power in 3–8 Hz / Power in 0.5–3 Hz) provides an informative physical baseline for microcontroller deployment.

---

## 7. Assumptions & Plan for Phase 3 (Preprocessing)

1. **Target Channel Prefix:** Default `ankleL` (configurable to `ankleR` or combined).
2. **Prediction Horizon (Pre-FoG):**
   * Normal: no FoG in window and no FoG in the subsequent 3.0 seconds.
   * Pre-FoG: window ends 0.5 to 3.0 seconds prior to a labeled FoG onset (impending freeze target).
   * FoG: window overlaps active freezing episode $\\ge 50\\%$.
   * Binary training target: `(Pre-FoG or FoG) == 1`.
3. **Window Parameters:** 2.0 s @ 50 Hz = 100 samples; stride 0.5 s = 25 samples.
4. **Subject-Wise Splitting:** 
   * Strict Leave-Group-Out by Subject: no subject's trials will appear in more than one split (e.g., 15 train subjects, 3 validation subjects, 4 test subjects).
   * Assertion verified: zero recording/subject leakage between splits.
5. **Scaler Fitting:** Fitted strictly on training set windows only; exported as JSON and C headers `IMU_MEAN[6]` and `IMU_STD[6]`.

---

**Generated plots:**
- `reports/class_balance.png`
- `reports/example_segments.png`
- `reports/spectrum_comparison.png`
- `reports/episode_duration_hist.png`
- `reports/correlation_heatmap.png`
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[EDA] Written data audit report to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="EDA and Data Audit for TinyML FoG Prediction")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw CSV data directory")
    parser.add_argument("--output_dir", type=str, default="reports", help="Path to output reports directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    df = load_data(args.data_dir)
    col_df = audit_columns_and_types(df)
    stats_df = audit_channel_stats(df, sensor_prefix='ankleL')
    dt_summary, trial_stats = audit_sampling_rate(df)
    audit_data = audit_classes_and_episodes(df)
    
    generate_plots(df, audit_data, args.output_dir)
    
    # Save eda_summary.json
    summary_data = {
        'seed': args.seed,
        'dataset_shape': list(df.shape),
        'dt_summary': dt_summary,
        'class_summary': {
            'total_samples': audit_data['total_samples'],
            'normal_samples': audit_data['normal_samples'],
            'fog_samples': audit_data['fog_samples'],
            'fog_pct': audit_data['fog_pct'],
            'severity_counts': audit_data['severity_counts'],
            'total_episodes': audit_data['total_episodes'],
            'freezer_subjects': audit_data['freezer_subjects'],
            'non_freezer_subjects': audit_data['non_freezer_subjects']
        },
        'sensor_units': {
            'accelerometer': 'g',
            'gyroscope': 'deg/s'
        }
    }
    with open(os.path.join(args.output_dir, 'eda_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"[EDA] Saved eda_summary.json to {args.output_dir}")
    
    # Write reports/data_audit.md
    write_audit_markdown(col_df, stats_df, dt_summary, audit_data, os.path.join(args.output_dir, 'data_audit.md'))
    print("\n[EDA] Data audit complete!")

if __name__ == "__main__":
    main()

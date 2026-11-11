"""
Preprocessing and Windowing Pipeline for ESP32 Freezing-of-Gait Prediction.
Resamples 60 Hz IMU streams to 50 Hz, generates 2.0s windows with 0.5s stride,
labels windows as Normal (0), Pre-FoG (1), FoG (2), or Transition (-1),
performs strictly subject-independent splitting, fits z-score scaler on training data only,
and exports scaler constants.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import signal

def set_seed(seed=42):
    np.random.seed(seed)
    print(f"[PREPROCESS] Random seed set to {seed}")

def resample_trial(trial_df, channels, source_fs=60.0, target_fs=50.0):
    """
    Resample a single continuous trial from source_fs to target_fs.
    Uses resample_poly for signal channels (up=5, down=6 for 60Hz -> 50Hz).
    Uses nearest-neighbor indexing for discrete labels.
    Never resamples across discontinuous recording trial boundaries.
    """
    valid_mask = trial_df[channels].notna().all(axis=1)
    if not valid_mask.any():
        return None, None, None
    
    first_valid = valid_mask.idxmax()
    last_valid = valid_mask[::-1].idxmax()
    df_clean = trial_df.loc[first_valid:last_valid].copy()
    
    if len(df_clean) < int(source_fs * 2.0):  # Need at least 2 seconds
        return None, None, None
        
    sig = df_clean[channels].values
    n_orig = len(sig)
    
    # 60 Hz to 50 Hz: up=5, down=6
    sig_resampled = signal.resample_poly(sig, up=5, down=6, axis=0)
    n_resampled = len(sig_resampled)
    
    t_orig = np.arange(n_orig) / source_fs
    t_new = np.arange(n_resampled) / target_fs
    
    # Nearest neighbor index for labels
    nn_indices = np.clip(np.round(t_new * source_fs).astype(int), 0, n_orig - 1)
    
    fog_resampled = df_clean['fog'].values[nn_indices]
    sev_resampled = df_clean['fog_severity'].values[nn_indices]
    
    return sig_resampled, fog_resampled, sev_resampled

def create_windows(sig, fog, window_size=100, stride=25, target_fs=50.0,
                   pre_fog_min_s=0.5, pre_fog_max_s=3.0, fog_overlap_thresh=0.5):
    """
    Slice 1D signals into overlapping windows and assign rigorous clinical labels:
    - Normal (0): no FoG in window AND next FoG onset is > pre_fog_max_s (3.0s) away.
    - Pre-FoG (1): no FoG in window AND next FoG onset is within [pre_fog_min_s, pre_fog_max_s] (0.5-3.0s).
    - FoG (2): active FoG in window with overlap >= fog_overlap_thresh (>= 50%).
    - Transition (-1): 
        * overlap > 0 but < fog_overlap_thresh (< 50% active FoG)
        * OR no FoG in window but next FoG onset is < pre_fog_min_s (< 0.5s immediately before freeze).
      In binary prediction (normal vs freeze risk), Transition is treated as positive (1).
    """
    n_samples = len(sig)
    pre_fog_min_samples = int(pre_fog_min_s * target_fs)  # 25 samples
    pre_fog_max_samples = int(pre_fog_max_s * target_fs)  # 150 samples
    
    # Locate all FoG onset indices in this trial
    fog_padded = np.pad(fog, (1, 1), 'constant', constant_values=0)
    onsets = np.where(np.diff(fog_padded) == 1)[0]  # indices in fog array
    
    windows = []
    labels_multi = []
    labels_binary = []
    meta = []
    
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        win_sig = sig[start:end]
        win_fog = fog[start:end]
        
        overlap_ratio = np.mean(win_fog == 1)
        
        # Check active FoG criterion (>= 50% active freeze)
        if overlap_ratio >= fog_overlap_thresh:
            multi_lbl = 2  # Active FoG
            bin_lbl = 1
        elif overlap_ratio > 0:
            # Active FoG entering window, but < 50% overlap -> Transition
            multi_lbl = -1  # Transition / Partial overlap
            bin_lbl = 1
        else:
            # No FoG in window: check future horizon for impending onset
            future_onsets = onsets[onsets >= end]
            if len(future_onsets) > 0:
                dist_to_onset = future_onsets[0] - end
                if dist_to_onset < pre_fog_min_samples:
                    # Under 0.5s before freeze onset -> Immediate pre-freeze transition
                    multi_lbl = -1  # Transition
                    bin_lbl = 1
                elif dist_to_onset <= pre_fog_max_samples:
                    # Strictly within [0.5, 3.0] s early warning horizon
                    multi_lbl = 1  # Pure Pre-FoG
                    bin_lbl = 1
                else:
                    # Next FoG is > 3.0s away -> Normal gait
                    multi_lbl = 0  # Normal
                    bin_lbl = 0
            else:
                # No future FoG in this trial -> Normal gait
                multi_lbl = 0  # Normal
                bin_lbl = 0
                
        windows.append(win_sig)
        labels_multi.append(multi_lbl)
        labels_binary.append(bin_lbl)
        meta.append({
            'start_sample': start,
            'end_sample': end,
            'start_time_s': start / target_fs,
            'end_time_s': end / target_fs,
            'overlap_ratio': float(overlap_ratio),
            'multi_label': int(multi_lbl),
            'binary_label': int(bin_lbl)
        })
        
    return np.array(windows, dtype=np.float32), np.array(labels_binary, dtype=np.int32), np.array(labels_multi, dtype=np.int32), meta

def export_scaler_constants(mean, std, output_json, output_header):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    scaler_dict = {
        'mean': [float(m) for m in mean],
        'std': [float(s) for s in std],
        'channels': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    }
    with open(output_json, 'w') as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"[PREPROCESS] Saved scaler JSON to {output_json}")
    
    header_content = f"""// Auto-generated scaler constants for ESP32 DevKit V1
// Channels: acc_x, acc_y, acc_z (g), gyro_x, gyro_y, gyro_z (deg/s)
#ifndef SCALER_CONSTANTS_H
#define SCALER_CONSTANTS_H

const float IMU_MEAN[6] = {{
    {mean[0]:.8f}f, {mean[1]:.8f}f, {mean[2]:.8f}f,
    {mean[3]:.8f}f, {mean[4]:.8f}f, {mean[5]:.8f}f
}};

const float IMU_STD[6] = {{
    {std[0]:.8f}f, {std[1]:.8f}f, {std[2]:.8f}f,
    {std[3]:.8f}f, {std[4]:.8f}f, {std[5]:.8f}f
}};

#endif // SCALER_CONSTANTS_H
"""
    with open(output_header, 'w') as f:
        f.write(header_content)
    print(f"[PREPROCESS] Saved scaler header to {output_header}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess IMU data for ESP32 FoG Prediction")
    parser.add_argument("--data_file", type=str, default="data/raw/sensor_data.csv", help="Path to raw CSV")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to processed output")
    parser.add_argument("--export_dir", type=str, default="export", help="Path to export directory")
    parser.add_argument("--sensor_prefix", type=str, default="ankleL", help="Sensor prefix (e.g. ankleL, ankleR)")
    parser.add_argument("--source_fs", type=float, default=60.0, help="Source sampling frequency (Hz)")
    parser.add_argument("--target_fs", type=float, default=50.0, help="Target sampling frequency (Hz)")
    parser.add_argument("--window_sec", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=0.5, help="Window stride in seconds")
    parser.add_argument("--pre_fog_min", type=float, default=0.5, help="Minimum pre-FoG horizon in seconds")
    parser.add_argument("--pre_fog_max", type=float, default=3.0, help="Maximum pre-FoG horizon in seconds")
    parser.add_argument("--fog_overlap", type=float, default=0.5, help="FoG overlap fraction threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.export_dir, exist_ok=True)
    
    print(f"[PREPROCESS] Loading raw data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    print(f"[PREPROCESS] Raw rows: {len(df):,}")
    
    channels = [f"{args.sensor_prefix}_acc_x", f"{args.sensor_prefix}_acc_y", f"{args.sensor_prefix}_acc_z",
                f"{args.sensor_prefix}_gyro_x", f"{args.sensor_prefix}_gyro_y", f"{args.sensor_prefix}_gyro_z"]
    print(f"[PREPROCESS] Selected channels: {channels}")
    
    window_samples = int(args.window_sec * args.target_fs)  # 100 samples
    stride_samples = int(args.stride_sec * args.target_fs)  # 25 samples
    
    # Subject-wise partition: Strictly subject-independent
    # 22 subjects: 15 Train, 3 Val, 4 Held-Out Test
    train_subjects = [1, 2, 3, 4, 5, 9, 11, 12, 13, 15, 16, 6, 7, 8, 10]
    val_subjects = [17, 19, 14]
    test_subjects = [20, 21, 22, 18]
    
    # Verification assertions
    assert set(train_subjects).isdisjoint(set(val_subjects)), "Train and Val subjects overlap!"
    assert set(train_subjects).isdisjoint(set(test_subjects)), "Train and Test subjects overlap!"
    assert set(val_subjects).isdisjoint(set(test_subjects)), "Val and Test subjects overlap!"
    all_assigned = set(train_subjects) | set(val_subjects) | set(test_subjects)
    assert all_assigned == set(df['subjectID'].unique()), "Not all subjects assigned to split!"
    
    print(f"[PREPROCESS] Subject assignment confirmed: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")
    
    split_data = {
        'train': {'windows': [], 'labels_bin': [], 'labels_multi': [], 'meta': []},
        'val': {'windows': [], 'labels_bin': [], 'labels_multi': [], 'meta': []},
        'test': {'windows': [], 'labels_bin': [], 'labels_multi': [], 'meta': []}
    }
    
    trials = df.groupby(['subjectID', 'sessionID', 'taskID'])
    skipped_trials = 0
    processed_trials = 0
    
    for (subj, sess, task), trial_df in trials:
        # Determine split
        if subj in train_subjects:
            split_key = 'train'
        elif subj in val_subjects:
            split_key = 'val'
        elif subj in test_subjects:
            split_key = 'test'
        else:
            raise ValueError(f"Unknown subject {subj}")
            
        sig_resampled, fog_resampled, sev_resampled = resample_trial(
            trial_df, channels, source_fs=args.source_fs, target_fs=args.target_fs
        )
        
        if sig_resampled is None:
            skipped_trials += 1
            continue
            
        wins, l_bin, l_multi, meta = create_windows(
            sig_resampled, fog_resampled,
            window_size=window_samples,
            stride=stride_samples,
            target_fs=args.target_fs,
            pre_fog_min_s=args.pre_fog_min,
            pre_fog_max_s=args.pre_fog_max,
            fog_overlap_thresh=args.fog_overlap
        )
        
        if len(wins) == 0:
            continue
            
        for m in meta:
            m['subjectID'] = int(subj)
            m['sessionID'] = int(sess)
            m['taskID'] = int(task)
            m['split'] = split_key
            
        split_data[split_key]['windows'].append(wins)
        split_data[split_key]['labels_bin'].append(l_bin)
        split_data[split_key]['labels_multi'].append(l_multi)
        split_data[split_key]['meta'].extend(meta)
        processed_trials += 1
        
    print(f"[PREPROCESS] Processed {processed_trials} trials, skipped {skipped_trials} unrecorded/short trials.")
    
    # Concatenate per split
    X_train_raw = np.concatenate(split_data['train']['windows'], axis=0)
    y_train_bin = np.concatenate(split_data['train']['labels_bin'], axis=0)
    y_train_multi = np.concatenate(split_data['train']['labels_multi'], axis=0)
    
    X_val_raw = np.concatenate(split_data['val']['windows'], axis=0)
    y_val_bin = np.concatenate(split_data['val']['labels_bin'], axis=0)
    y_val_multi = np.concatenate(split_data['val']['labels_multi'], axis=0)
    
    X_test_raw = np.concatenate(split_data['test']['windows'], axis=0)
    y_test_bin = np.concatenate(split_data['test']['labels_bin'], axis=0)
    y_test_multi = np.concatenate(split_data['test']['labels_multi'], axis=0)
    
    print("\n--- Window Statistics Before Normalization ---")
    print(f"Train: {X_train_raw.shape[0]:,} windows | Normal: {(y_train_bin==0).sum():,}, Risk/FoG: {(y_train_bin==1).sum():,} (Pre-FoG: {(y_train_multi==1).sum():,}, FoG: {(y_train_multi==2).sum():,}, Trans: {(y_train_multi==-1).sum():,})")
    print(f"Val:   {X_val_raw.shape[0]:,} windows | Normal: {(y_val_bin==0).sum():,}, Risk/FoG: {(y_val_bin==1).sum():,} (Pre-FoG: {(y_val_multi==1).sum():,}, FoG: {(y_val_multi==2).sum():,}, Trans: {(y_val_multi==-1).sum():,})")
    print(f"Test:  {X_test_raw.shape[0]:,} windows | Normal: {(y_test_bin==0).sum():,}, Risk/FoG: {(y_test_bin==1).sum():,} (Pre-FoG: {(y_test_multi==1).sum():,}, FoG: {(y_test_multi==2).sum():,}, Trans: {(y_test_multi==-1).sum():,})")
    
    # Fit scaler on TRAIN ONLY
    print("\n--- Fitting Scaler on Training Windows Only ---")
    mean = np.mean(X_train_raw, axis=(0, 1))
    std = np.std(X_train_raw, axis=(0, 1))
    # Prevent divide by zero
    std = np.where(std < 1e-6, 1.0, std)
    
    print("Channel Means: ", np.round(mean, 5).tolist())
    print("Channel Stds:  ", np.round(std, 5).tolist())
    
    # Export scaler constants
    export_scaler_constants(mean, std,
                            os.path.join(args.export_dir, "scaler.json"),
                            os.path.join(args.export_dir, "scaler_constants.h"))
    
    # Standardize windows
    X_train_norm = (X_train_raw - mean) / std
    X_val_norm = (X_val_raw - mean) / std
    X_test_norm = (X_test_raw - mean) / std
    
    # Save NPZ dataset with both normalized and raw windows
    dataset_path = os.path.join(args.output_dir, "fog_dataset.npz")
    np.savez_compressed(
        dataset_path,
        X_train=X_train_norm,
        y_train_bin=y_train_bin,
        y_train_multi=y_train_multi,
        X_train_raw=X_train_raw,
        X_val=X_val_norm,
        y_val_bin=y_val_bin,
        y_val_multi=y_val_multi,
        X_val_raw=X_val_raw,
        X_test=X_test_norm,
        y_test_bin=y_test_bin,
        y_test_multi=y_test_multi,
        X_test_raw=X_test_raw,
        mean=mean,
        std=std
    )
    print(f"[PREPROCESS] Saved dataset NPZ to {dataset_path} ({os.path.getsize(dataset_path):,} bytes)")
    
    # Save metadata JSONs
    for split_k in ['train', 'val', 'test']:
        meta_path = os.path.join(args.output_dir, f"{split_k}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(split_data[split_k]['meta'], f)
        print(f"[PREPROCESS] Saved {split_k} metadata to {meta_path} ({len(split_data[split_k]['meta']):,} windows)")
        
    # Save preprocessing config
    cfg_path = os.path.join(args.output_dir, "preprocess_config.json")
    with open(cfg_path, 'w') as f:
        json.dump({
            'source_fs': args.source_fs,
            'target_fs': args.target_fs,
            'window_sec': args.window_sec,
            'stride_sec': args.stride_sec,
            'window_samples': window_samples,
            'stride_samples': stride_samples,
            'pre_fog_min_s': args.pre_fog_min,
            'pre_fog_max_s': args.pre_fog_max,
            'fog_overlap_thresh': args.fog_overlap,
            'channels': channels,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects,
            'scaler_mean': [float(m) for m in mean],
            'scaler_std': [float(s) for s in std],
            'seed': args.seed
        }, f, indent=2)
    print(f"[PREPROCESS] Saved configuration to {cfg_path}")
    print("[PREPROCESS] Preprocessing completed successfully.")

if __name__ == '__main__':
    main()

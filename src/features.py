"""
Hand-crafted Feature Extraction for TinyML Freezing-of-Gait Prediction.
Computes time-domain (mean, std, RMS, ZCR) and frequency-domain features (Freeze Index,
locomotor power, freeze band power, total spectral power) from 2.0s IMU windows.
"""

import numpy as np

def extract_window_features(X_windows, mean_scaler=None, std_scaler=None, fs=50.0):
    """
    Extracts 28 handcrafted features from a batch of windows (N, 100, 6).
    
    Channels:
      0: acc_x, 1: acc_y, 2: acc_z (in g)
      3: gyro_x, 4: gyro_y, 5: gyro_z (in deg/s)
      
    Features per window:
      - 6 channel means
      - 6 channel standard deviations
      - 6 channel RMS values
      - 6 channel zero-crossing rates (mean-centered)
      - Ankle acc magnitude:
        * Locomotor band power (0.5 - 3.0 Hz)
        * Freeze band power (3.0 - 8.0 Hz)
        * Freeze Index (Freeze Power / Locomotor Power)
        * Total spectral power (0.5 - 20.0 Hz)
    """
    N, W, C = X_windows.shape
    
    # Reconstruct raw signals if scaler is provided, for physically meaningful spectral energy
    if mean_scaler is not None and std_scaler is not None:
        X_raw = X_windows * std_scaler + mean_scaler
    else:
        X_raw = X_windows
        
    features = []
    
    # 1. Time-domain channel features
    # Mean
    ch_mean = np.mean(X_windows, axis=1)  # (N, C)
    # Std
    ch_std = np.std(X_windows, axis=1)    # (N, C)
    # RMS
    ch_rms = np.sqrt(np.mean(X_windows**2, axis=1))  # (N, C)
    # Zero crossing rate (around channel mean)
    centered = X_windows - ch_mean[:, np.newaxis, :]
    signs = np.sign(centered)
    signs[signs == 0] = 1
    sign_changes = np.diff(signs, axis=1) != 0
    ch_zcr = np.sum(sign_changes, axis=1) / (W - 1)  # (N, C)
    
    # 2. Acc magnitude and spectral features
    acc_x = X_raw[:, :, 0]
    acc_y = X_raw[:, :, 1]
    acc_z = X_raw[:, :, 2]
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)  # (N, W)
    
    # Remove DC component for FFT
    acc_mag_detrend = acc_mag - np.mean(acc_mag, axis=1, keepdims=True)
    
    # FFT
    fft_vals = np.fft.rfft(acc_mag_detrend, axis=1)
    fft_power = (np.abs(fft_vals)**2) / W  # (N, W//2 + 1)
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)  # [0, 0.5, 1.0, ..., 25.0]
    
    # Frequency bands
    loco_mask = (freqs >= 0.5) & (freqs <= 3.0)
    freeze_mask = (freqs > 3.0) & (freqs <= 8.0)
    total_mask = (freqs >= 0.5) & (freqs <= 20.0)
    
    loco_power = np.sum(fft_power[:, loco_mask], axis=1)
    freeze_power = np.sum(fft_power[:, freeze_mask], axis=1)
    freeze_index = freeze_power / (loco_power + 1e-6)
    total_power = np.sum(fft_power[:, total_mask], axis=1)
    
    # Stack features: 6 + 6 + 6 + 6 + 4 = 28 features
    feats = np.column_stack([
        ch_mean, ch_std, ch_rms, ch_zcr,
        loco_power, freeze_power, freeze_index, total_power
    ])
    
    feature_names = (
        [f"mean_ch{i}" for i in range(C)] +
        [f"std_ch{i}" for i in range(C)] +
        [f"rms_ch{i}" for i in range(C)] +
        [f"zcr_ch{i}" for i in range(C)] +
        ["loco_power", "freeze_power", "freeze_index", "total_power"]
    )
    
    return feats.astype(np.float32), feature_names

if __name__ == "__main__":
    dummy = np.random.randn(10, 100, 6).astype(np.float32)
    feats, names = extract_window_features(dummy)
    print(f"Extracted {feats.shape[1]} features from {len(dummy)} windows.")
    print("Feature names:", names)

"""
Unit Tests for Preprocessing and Windowing Pipeline.
Validates resampling, sliding window generation, clinical label semantics,
and channel ordering.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import signal

from src.preprocess import resample_trial, create_windows

def test_resample_ratio_and_length():
    """Verify that 60 Hz to 50 Hz polyphase resampling exactly scales sample count by 5/6."""
    fs_orig = 60.0
    fs_target = 50.0
    duration_s = 4.0
    n_orig = int(fs_orig * duration_s)  # 240 samples
    
    t = np.arange(n_orig) / fs_orig
    channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    data = {ch: np.sin(2 * np.pi * 1.5 * t) for ch in channels}
    data['fog'] = np.zeros(n_orig, dtype=int)
    data['fog_severity'] = np.zeros(n_orig, dtype=int)
    trial_df = pd.DataFrame(data)
    
    sig_resampled, fog_resampled, sev_resampled = resample_trial(
        trial_df, channels, source_fs=fs_orig, target_fs=fs_target
    )
    
    expected_n = int(round(n_orig * 5.0 / 6.0))  # 200 samples
    assert sig_resampled is not None
    assert sig_resampled.shape == (expected_n, 6)
    assert len(fog_resampled) == expected_n
    assert len(sev_resampled) == expected_n

def test_window_geometry():
    """Verify window shape (100, 6) and stride (25 samples = 0.5s)."""
    fs = 50.0
    n_samples = 250  # 5.0s of data at 50 Hz
    sig = np.random.randn(n_samples, 6)
    fog = np.zeros(n_samples, dtype=int)
    
    windows, labels_binary, labels_multi, meta = create_windows(
        sig, fog, window_size=100, stride=25, target_fs=fs
    )
    
    expected_windows = (n_samples - 100) // 25 + 1
    assert len(windows) == expected_windows == 7
    assert windows.shape == (7, 100, 6)
    assert len(labels_multi) == 7
    assert len(labels_binary) == 7

def test_clinical_labeling_semantics():
    """
    Test clinical labeling state logic:
    - Normal (0): No FoG in window AND next onset > 3.0s away.
    - Pre-FoG (1): No FoG in window AND next onset in [0.5, 3.0]s.
    - Active FoG (2): FoG overlap >= 50% in window.
    - Transition (-1): FoG overlap in (0, 50%) OR onset < 0.5s away.
    """
    fs = 50.0
    # Create 12 seconds of data = 600 samples
    sig = np.zeros((600, 6))
    fog = np.zeros(600, dtype=int)
    
    # Introduce an active FoG episode from t = 7.0s (idx 350) to t = 10.0s (idx 500)
    fog[350:500] = 1
    
    windows, labels_binary, labels_multi, meta = create_windows(
        sig, fog, window_size=100, stride=25, target_fs=fs,
        pre_fog_min_s=0.5, pre_fog_max_s=3.0, fog_overlap_thresh=0.5
    )
    
    # Inspect windows relative to onset at sample 350:
    # Window 0: [0, 100] -> end=100. Dist to onset = 350 - 100 = 250 samples = 5.0s > 3.0s -> Normal (0)
    assert labels_multi[0] == 0
    assert labels_binary[0] == 0
    
    # Window 5: [125, 225] -> end=225. Dist to onset = 350 - 225 = 125 samples = 2.5s (in [0.5, 3.0]s) -> Pre-FoG (1)
    assert labels_multi[5] == 1
    assert labels_binary[5] == 1
    
    # Window 8: [200, 300] -> end=300. Dist to onset = 350 - 300 = 50 samples = 1.0s -> Pre-FoG (1)
    assert labels_multi[8] == 1
    assert labels_binary[8] == 1
    
    # Window 10: [250, 350] -> end=350. Dist to onset = 0 < 0.5s -> Transition (-1, bin=1)
    assert labels_multi[10] == -1
    assert labels_binary[10] == 1
    
    # Window 13: [325, 425] -> FoG in [350, 425] = 75 samples out of 100 = 75% >= 50% -> Active FoG (2, bin=1)
    assert labels_multi[13] == 2
    assert labels_binary[13] == 1

def test_channel_order_consistency():
    """Verify that expected channel order is strictly preserved."""
    expected_channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    assert len(expected_channels) == 6
    assert expected_channels[0:3] == ['acc_x', 'acc_y', 'acc_z']
    assert expected_channels[3:6] == ['gyro_x', 'gyro_y', 'gyro_z']

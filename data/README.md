# Freezing of Gait (FoG) Dataset Instructions

This directory houses the Parkinson's Disease Freezing of Gait sensor recordings.

## Raw Dataset Overview
The raw IMU recordings originate from wearable inertial measurement unit (IMU) sensors positioned on patients with Parkinson's Disease executing standardized mobility protocols (straight walking, 360-degree turning, obstacle crossing, and open hallway navigation).

- **Sensors:** 6-axis IMU (3-axis Accelerometer in $g$, 3-axis Gyroscope in $\text{deg/s}$) placed on the shank/ankle.
- **Original Sample Rate:** 60 Hz.
- **Resampled Target Rate:** 50 Hz (via polyphase filter, up=5, down=6).
- **Labels:**
  - `0`: Normal Gait (no freezing in window and > 3.0 s before any freeze).
  - `1`: Impending Pre-FoG Early Warning ([0.5, 3.0] s before freeze onset).
  - `2`: Active Freezing of Gait ($\ge 50\%$ active episode overlap).
  - `-1`: Transition / Boundary segment.

## Downloading or Placing Raw Data
Due to GitHub's file size limit (100 MB maximum per file), `sensor_data.csv` (119.6 MB) is not tracked directly in git history.

Place your raw dataset file at:
```text
data/raw/sensor_data.csv
```

### Verification Checksum
```text
File:   data/raw/sensor_data.csv
Size:   119,629,580 bytes (~114.1 MB)
SHA256: 888875133FEF386AFFDDD0A5864F122D527D8D7BC588A69905CC0871BEF53477
```

You can verify the file checksum using:
- **Windows (PowerShell):**
  ```powershell
  Get-FileHash -Path data/raw/sensor_data.csv -Algorithm SHA256
  ```
- **Linux / macOS:**
  ```bash
  sha256sum data/raw/sensor_data.csv
  ```

## Helper Script
To verify the dataset placement and integrity automatically, run:
```bash
python src/download_dataset.py --verify-only
```

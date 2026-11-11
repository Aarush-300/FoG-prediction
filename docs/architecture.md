# System Architecture & Technical Specifications

## 1. Dual-Core FreeRTOS Architecture (ESP32-WROOM-32)

The wearable system uses Espressif's dual-core 32-bit Xtensa LX6 microcontroller running FreeRTOS. Workloads are partitioned symmetrically between cores to guarantee zero dropped IMU samples while executing deep convolutional inference.

```mermaid
flowchart TD
    subgraph Core0["Core 0: High-Priority Sampling Task (Priority 3, 50 Hz)"]
        IMU["MPU6050 6-Axis IMU<br/>(I2C Fast Mode @ 400 kHz)"] -->|20 ms period| Read["Read Raw 16-bit Registers<br/>(Acc: ±2g, Gyro: ±500 dps)"]
        Read --> Calib["Apply Gyro Zero-Rate Bias Correction"]
        Calib --> Scale["Convert to Physical Units<br/>(Acc: g, Gyro: deg/s)"]
        Scale --> MutexTake["xSemaphoreTake(g_buffer_mutex)"]
        MutexTake --> RingBuf["Push Sample into 100x6 Circular Ring Buffer"]
        RingBuf --> MutexGive["xSemaphoreGive(g_buffer_mutex)"]
    end

    subgraph Core1["Core 1: Inference & Cueing Task (Priority 2, 500 ms Stride)"]
        CheckReady["Check if 25 New Samples (0.5s) Accumulated"] -->|Window Ready| PopWin["Extract 100x6 Window (2.0s)"]
        PopWin --> ZScore["Normalize with Offline Scaler Constants (Mean, Std)"]
        ZScore --> Quant["Quantize Float32 -> INT8 (Scale: 0.095929, ZP: 7)"]
        Quant --> TFLM["TensorFlow Lite Micro Interpreter<br/>Improved_ResCNN (43,393 Params, ~3.56 ms)"]
        TFLM --> Dequant["Dequantize Output -> P(FoG Impending / Active)"]
        Dequant --> FSM["4-State Temporal Post-Processing State Machine"]
        FSM -->|State == ACTIVE_ALERT| Haptic["LEDC PWM Motor Driver (GPIO 13)<br/>Rhythmic Cadence Cueing (3 Pulses @ 1 kHz)"]
    end

    RingBuf -.->|Shared Memory Protection| PopWin
```

---

## 2. Signal Preprocessing & Temporal Windowing Pipeline

### Resampling Strategy
The clinical laboratory data recorded at 60 Hz is resampled to 50 Hz using polyphase rational factor filtering (`scipy.signal.resample_poly(up=5, down=6)`):
$$\frac{f_{\text{target}}}{f_{\text{source}}} = \frac{50\text{ Hz}}{60\text{ Hz}} = \frac{5}{6}$$
Resampling is strictly applied within individual trial recordings to prevent artificial discontinuity boundary artifacts.

### 2.0-Second Sliding Windows
- **Window Length:** 100 samples ($2.0\text{ s} \times 50\text{ Hz}$).
- **Stride:** 25 samples ($0.5\text{ s} \times 50\text{ Hz}$).
- **Channels (6):** `acc_x`, `acc_y`, `acc_z` (in $g$), `gyro_x`, `gyro_y`, `gyro_z` (in $\text{deg/s}$).

### Labeling Taxonomy
Windows are classified into four mutually exclusive clinical regimes:
1. **Normal Gait (`0`):** No freeze within the window AND the next freeze onset is $> 3.0\text{ s}$ in the future.
2. **Pre-FoG Early Warning (`1`):** No active freeze in the window, but an impending freeze occurs within $[0.5, 3.0]\text{ s}$ after window termination.
3. **Active Freezing (`2`):** $\ge 50\%$ of the window overlaps an active clinical freeze episode.
4. **Transition (`-1`):** Window overlaps $< 50\%$ active freeze or falls within $< 0.5\text{ s}$ immediately preceding freeze onset.

---

## 3. Four-State Temporal Alarm Finite State Machine

To prevent transient noise spikes from causing false tactile alarms while ensuring rapid cues during impending freezes, predictions pass through a 4-state hysteresis state machine:

```mermaid
stateDiagram-v2
    [*] --> FOG_STATE_NORMAL
    
    FOG_STATE_NORMAL --> FOG_STATE_WARNING: P >= 0.49 (High Threshold)
    FOG_STATE_WARNING --> FOG_STATE_ACTIVE_ALERT: Consecutive High Count >= 2
    FOG_STATE_WARNING --> FOG_STATE_NORMAL: P < 0.29 (Low Threshold)
    
    FOG_STATE_ACTIVE_ALERT --> FOG_STATE_COOLDOWN: Trigger Haptic Cadence Cue
    
    FOG_STATE_COOLDOWN --> FOG_STATE_NORMAL: Elapsed Time >= 3000 ms AND P < 0.29
    FOG_STATE_COOLDOWN --> FOG_STATE_COOLDOWN: Ongoing Freeze or Cooldown Active
```

- **Exponential Moving Average (EMA):** $P_{\text{smooth}, t} = 0.60 \cdot P_{\text{raw}, t} + 0.40 \cdot P_{\text{smooth}, t-1}$
- **Hysteresis High Threshold:** $0.49$
- **Hysteresis Low Threshold:** $0.29$
- **Confirmation Filter:** 2 consecutive windows ($\ge 1.0\text{ s}$) above high threshold required to fire.
- **Cooldown Window:** $3,000\text{ ms}$ minimum lockout between distinct episodes.

# IMU Coordinate System, Axis Conventions & Register Mapping

This document specifies the exact coordinate frames, register mappings, physical units, matrix transformation conventions, and filtering parameters between the physical **InvenSense MPU6050** sensor hardware and the **TinyML Freezing-of-Gait (FoG)** inference pipeline.

---

## 1. Sensor Orientation & Anatomical Placement

The MPU6050 sensor is mounted securely on the lateral side of the left shank/ankle above the lateral malleolus.

```text
               Anatomical Shank Orientation
                     ▲ +Z (Anterior / Forward)
                     │
                     │
    (Lateral) +Y ────┼──── -Y (Medial)
                    ╱│
                   ╱ │
                  ▼  ▼
                 +X  -X (Inferior / Gravity Vector during Standing)
```

- **Nominal Stationary Gravity Reference:**
  When a patient stands upright at rest, earth gravity produces a static acceleration vector:
  $$\vec{g}_{\text{nominal}} = [-1.006\text{ g}, +0.132\text{ g}, +0.113\text{ g}]$$
  - Primary gravity vector points downwards along the $-X$ shank axis.
  - Slight positive $Y$ and $Z$ biases reflect standard lower-limb physiological valgus and forward tilt.

---

## 2. MPU6050 Register Map & Conversion to Physical Units

The ESP32 reads 14 consecutive bytes from register `0x3B` (`ACCEL_XOUT_H`) via a single I2C burst read at 50 Hz (every 20 ms).

| Register Address | Register Name | Data Format | Physical Quantity | Sensitivity Scale Factor | Model Channel Index |
|:---:|:---|:---:|:---|:---:|:---:|
| `0x3B` - `0x3C` | `ACCEL_XOUT_H/L` | 16-bit 2's complement | Accel X ($\pm 2\text{ g}$) | $16,384\text{ LSB} / g$ | `0: ankleL_acc_x` |
| `0x3D` - `0x3E` | `ACCEL_YOUT_H/L` | 16-bit 2's complement | Accel Y ($\pm 2\text{ g}$) | $16,384\text{ LSB} / g$ | `1: ankleL_acc_y` |
| `0x3F` - `0x40` | `ACCEL_ZOUT_H/L` | 16-bit 2's complement | Accel Z ($\pm 2\text{ g}$) | $16,384\text{ LSB} / g$ | `2: ankleL_acc_z` |
| `0x41` - `0x42` | `TEMP_OUT_H/L` | 16-bit 2's complement | Die Temperature | — (Unused by model) | — |
| `0x43` - `0x44` | `GYRO_XOUT_H/L` | 16-bit 2's complement | Gyro X ($\pm 500\text{ dps}$) | $65.5\text{ LSB} / (\text{deg/s})$ | `3: ankleL_gyro_x` |
| `0x45` - `0x46` | `GYRO_YOUT_H/L` | 16-bit 2's complement | Gyro Y ($\pm 500\text{ dps}$) | $65.5\text{ LSB} / (\text{deg/s})$ | `4: ankleL_gyro_y` |
| `0x47` - `0x48` | `GYRO_ZOUT_H/L` | 16-bit 2's complement | Gyro Z ($\pm 500\text{ dps}$) | $65.5\text{ LSB} / (\text{deg/s})$ | `5: ankleL_gyro_z` |

### Raw to Physical Conversion Formulas:
$$\vec{a}_{\text{raw}} = \begin{bmatrix} \text{rax} / 16384.0 \\ \text{ray} / 16384.0 \\ \text{raz} / 16384.0 \end{bmatrix} \quad (\text{in } g)$$

$$\vec{\omega}_{\text{raw}} = \begin{bmatrix} (\text{rgx} / 65.5) - \text{bias}_x \\ (\text{rgy} / 65.5) - \text{bias}_y \\ (\text{rgz} / 65.5) - \text{bias}_z \end{bmatrix} \quad (\text{in deg/s})$$

Gyroscope stationary zero-rate biases ($\text{bias}_x, \text{bias}_y, \text{bias}_z$) are calibrated during boot or on demand via serial command `c`.

---

## 3. Coordinate Rotation Matrix (`IMU_AXIS_ROTATION`)

To accommodate mechanical mounting variations (e.g. inverted mounting, 90-degree rotated enclosure brackets, or patient-specific anatomical strap angles) without altering training-set normalization or model weights, an orthogonal rotation matrix $R \in \mathrm{SO}(3)$ is applied in physical units ($g$, $\text{deg/s}$):

$$\vec{a}_{\text{model}} = R \cdot \vec{a}_{\text{raw}}$$
$$\vec{\omega}_{\text{model}} = R \cdot \vec{\omega}_{\text{raw}}$$

### Properties Enforced:
1. **Orthogonality:** $R^T R = I$ (preserves signal energy, vector norms, and relative angles).
2. **Proper Rotation:** $\det(R) = +1.0$ (right-handed chirality preserved; reflections strictly disallowed).

### Default (Nominal Identity):
```cpp
static const float IMU_AXIS_ROTATION[3][3] = {
    { 1.0f,  0.0f,  0.0f },
    { 0.0f,  1.0f,  0.0f },
    { 0.0f,  0.0f,  1.0f }
};
```

### Example: $90^\circ$ Yaw Rotation (Z-Axis):
```cpp
static const float IMU_AXIS_ROTATION[3][3] = {
    {  0.0f, -1.0f,  0.0f },
    {  1.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  1.0f }
};
```

---

## 4. On-Chip Digital Low-Pass Filter (DLPF) Configuration

The MPU6050 on-board hardware DLPF is programmed via register `0x1A` (`CONFIG`):
- **Setting:** `IMU_DLPF_CFG = 4`
- **Accelerometer Cutoff Frequency:** **21 Hz** (Group Delay: 8.5 ms)
- **Gyroscope Cutoff Frequency:** **20 Hz** (Group Delay: 8.3 ms)
- **Internal Gyro Sampling Rate:** 1 kHz

### Clinical & Hardware Rationale:
1. **Locomotion Spectrum Coverage:** Human walking and freezing frequencies fall strictly in the range of 0.5 Hz to 8.0 Hz (locomotor band 0.5–3.0 Hz, freezing band 3.0–8.0 Hz). A 21 Hz cutoff preserves $> 99\%$ of freeze energy.
2. **Anti-Aliasing:** Sampling at 50 Hz creates a Nyquist cutoff at 25 Hz. The 21 Hz hardware filter prevents foot impact shocks and high-frequency vibrations from aliasing into the 50 Hz digital signal.
3. **Actuator Noise Isolation:** Rhythmic haptic motor vibration is damped by the combination of DLPF attenuation and physical damping foam.

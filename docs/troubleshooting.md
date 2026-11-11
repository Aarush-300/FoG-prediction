# Troubleshooting & Operational Guide

## 1. Common Hardware & Sensor Issues

### Issue 1: `[ERROR] MPU6050 connection failed!` at boot
- **Cause:** ESP32 cannot communicate with the IMU over I2C at address `0x68`.
- **Diagnostics:**
  1. Verify VCC is connected to **3.3V** (do NOT connect to 5V unless the breakout board has an onboard 3.3V LDO regulator).
  2. Confirm SDA is wired to **GPIO 21** and SCL to **GPIO 22**.
  3. Ensure AD0 pin on the MPU6050 is connected to **GND** (pulling AD0 HIGH changes the address to `0x69`).
  4. Ensure I2C pull-up resistors ($4.7\text{ k}\Omega$) are present on SDA and SCL if using long jumper wires.

### Issue 2: IMU readings drift or trigger spurious alerts immediately after power-up
- **Cause:** Gyroscope bias was calibrated while the sensor was being moved.
- **Solution:** Keep the ankle strap completely motionless on a table or limb for 4 seconds after powering on or press `c` in the Serial Monitor to re-run stationary calibration.

---

## 2. Firmware Compilation & Flashing Issues

### Issue 3: `TFLite Micro: AllocateTensors() failed` or Internal SRAM overflow
- **Cause:** Tensor arena exceeds available contiguous internal memory.
- **Solution:** Verify `TENSOR_ARENA_SIZE_BYTES` in `firmware/ESP32_FoG/model_config.h` is set to `(220 * 1024)`. Ensure that you do not enable Wi-Fi or Bluetooth Classic simultaneously without configuring PSRAM, as the network stacks consume substantial SRAM.

### Issue 4: `ledcAttachChannel` or `ledcWrite` compilation errors
- **Cause:** ESP32 Arduino Core version difference between Core 2.x and Core 3.x.
- **Solution:** The provided `firmware/ESP32_FoG/haptic_feedback.cpp` includes preprocessor macros (`#if ESP_ARDUINO_VERSION_MAJOR >= 3`) that automatically select the correct LEDC API for both Core 2.x (`ledcSetup`, `ledcAttachPin`) and Core 3.x (`ledcAttachChannel`, `ledcWrite`).

---

## 3. On-Device Self-Test Verification

Typing `t` into the Serial Monitor at 115200 baud runs the built-in self-test against the 5 golden reference gait windows:
```text
--- RUNNING ON-DEVICE GOLDEN VECTOR SELF-TEST (100x6 Windows) ---
  Vector 1 [Normal (Walking)]: Expected P=0.484, Infer P=0.484 (Err: 0.0000, Tol: 0.05) Latency=3.52 ms -> PASS
  Vector 2 [Normal (Turning)]: Expected P=0.000, Infer P=0.000 (Err: 0.0000, Tol: 0.05) Latency=3.48 ms -> PASS
  Vector 3 [Pre-FoG (Onset Warning)]: Expected P=0.340, Infer P=0.340 (Err: 0.0000, Tol: 0.05) Latency=3.55 ms -> PASS
  Vector 4 [Active FoG (Tremor)]: Expected P=0.113, Infer P=0.113 (Err: 0.0000, Tol: 0.05) Latency=3.51 ms -> PASS
  Vector 5 [Active FoG (Akinetic)]: Expected P=0.531, Infer P=0.531 (Err: 0.0000, Tol: 0.05) Latency=3.50 ms -> PASS
--- SELF-TEST RESULT: 5 / 5 TEST VECTORS VERIFIED ---
```
If any vector fails, re-export the firmware headers using `python src/export_esp32.py`.

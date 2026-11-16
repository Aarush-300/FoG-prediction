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
- **Solution:** The firmware dynamically allocates a 16-byte aligned tensor arena using `heap_caps_malloc(..., MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)` with fallback sizes from 220 KB down to 100 KB. If allocation fails, ensure Wi-Fi and Bluetooth Classic are disabled to free internal SRAM blocks.

### Issue 4: `ledcAttach` or `ledcWrite` compilation errors
- **Cause:** ESP32 Arduino Core version difference between Core 2.x and Core 3.x.
- **Solution:** The provided `firmware/ESP32_FoG/haptic_feedback.cpp` includes preprocessor macros (`#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)`) that automatically select the correct LEDC API for both Core 2.x (`ledcSetup`, `ledcAttachPin`) and Core 3.x (`ledcAttach`, `ledcWrite`).

---

## 3. On-Device Diagnostic Commands

The firmware provides real-time interactive serial commands at 115200 baud:
- **`t` (Self-Test):** Executes inference across all 10 golden test vectors and verifies numerical output probability against recorded reference values within $\pm 0.05$ tolerance.
- **`b` (Benchmark):** Executes 100 consecutive inferences on golden vector #1, printing min/mean/max latency (via high-resolution `esp_timer`), dynamic tensor arena capacity and used bytes, free/minimum heap, and FreeRTOS task stack high-water marks.
- **`o` (Mounting Check):** Averages 2 seconds of stationary acceleration and computes the angular deviation from the training nominal coordinate frame `[-1.006, 0.132, 0.113] g`.
- **`c` (Calibrate):** Recalibrates gyroscope stationary zero-rate offset.
- **`s` (Status):** Prints inference counts, detected freeze events, last latency, and available free heap.

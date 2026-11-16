# ESP32 Firmware Build & Verification Guide

This document describes how to compile, verify, and flash the hardened real-time Parkinson's FoG detection firmware for ESP32.

---

## 1. Prerequisites & Toolchain Setup

The project uses [PlatformIO](https://platformio.org/) to provide deterministic, reproducible cross-compilation targeting the ESP32 DevKit V1 (Xtensa dual-core @ 240 MHz).

To avoid polluting your machine learning Python environment, install PlatformIO into a dedicated virtual environment:

```bash
# Create and activate a dedicated virtual environment
python -m venv .venv_pio
# Windows PowerShell:
.venv_pio\Scripts\Activate.ps1
# Linux/macOS:
source .venv_pio/bin/activate

# Install PlatformIO Core
pip install platformio
```

---

## 2. PlatformIO Configuration (`platformio.ini`)

The project supports both **Arduino-ESP32 Core 2.x** (production baseline) and **Arduino-ESP32 Core 3.x** (modern GCC 14 toolchain):

```ini
[platformio]
src_dir = firmware/ESP32_FoG
default_envs = esp32dev

[env:esp32dev]
platform = espressif32@6.5.0
board = esp32dev
framework = arduino
monitor_speed = 115200
build_src_filter = +<*> -<*_host_runner.*>
lib_deps =
    sebastianregelmann/tflite-micro @ 1.26.46
    Wire
build_flags =
    -DFSM_PRESET=1
    -O3

[env:esp32dev_core3]
platform = espressif32@8.0.0
board = esp32dev
framework = arduino
monitor_speed = 115200
build_src_filter = +<*> -<*_host_runner.*>
lib_deps =
    sebastianregelmann/tflite-micro @ 1.26.46
    Wire
build_flags =
    -DFSM_PRESET=1
    -O3
```

> [!NOTE]
> `build_src_filter` excludes host-side parity test runners (`*_host_runner.cpp`) which are compiled natively on the host machine for unit testing.

---

## 3. Compilation & Memory Footprint Verification

### Core 2.x (Arduino-ESP32 2.0.14 / ESP-IDF 4.4 / GCC 8.4)
```bash
pio run -e esp32dev
```
- **Build Result:** SUCCESS
- **Static RAM (DRAM):** 24,376 bytes (23.8 KB, **7.4%** of 320 KB total DRAM)
  - `.dram0.data`: 16,656 bytes
  - `.dram0.bss`: 7,720 bytes
  - **Free Internal SRAM Remaining:** > 303 KB available for heap and dynamic tensor arena.
- **Flash ROM:** 499,561 bytes (487.8 KB, **38.1%** of default 1.31 MB app partition)
  - `.flash.rodata`: 175,716 bytes (includes INT8 quantized weights `g_model_data`: 69,264 bytes)
  - `.flash.text`: 264,086 bytes
  - `.iram0.text`: 59,503 bytes

### Core 3.x (Arduino-ESP32 3.3.12 / ESP-IDF 5.1 / GCC 14.2)
```bash
pio run -e esp32dev_core3
```
- **Build Result:** SUCCESS
- **Static RAM (DRAM):** 26,208 bytes (25.6 KB, **8.0%** of 320 KB)
- **Flash ROM:** 587,191 bytes (573.4 KB, **44.8%** of app partition)

### Dynamic Tensor Arena Sizing
The earlier v1 firmware risk (allocating a static 220 KB array in BSS) was eliminated. In v2, the tensor arena is allocated dynamically via `heap_caps_malloc(c_size, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)` using an adaptive ladder (220 KB $\to$ 200 KB $\to$ 180 KB $\to$ 160 KB $\to$ 140 KB $\to$ 120 KB, floor 100 KB). With > 300 KB of free internal SRAM, the 220 KB arena allocation cleanly succeeds.

---

## 4. Hardware Flashing & Serial Monitor

Connect the ESP32 DevKit V1 via Micro-USB, identify the COM/serial port, and run:

```bash
# Upload binary to ESP32
pio run -t upload

# Open serial monitor at 115200 baud
pio device monitor
```

Upon boot, the firmware logs:
1. IMU I2C connection and WHO_AM_I handshake (`0x68`).
2. DLPF configuration (Register 26 set to `0x04` = 21 Hz cutoff, 8.5 ms group delay).
3. TFLM MicroInterpreter boot allocation on 16-byte aligned arena.
4. Golden vector verification (10 reference vectors matching Python model output within 0.005).
5. Dual-core FreeRTOS task launch (Core 0: 50 Hz IMU acquisition, Core 1: 2 Hz inference).

---

## 5. Known Pitfalls & Solutions

1. **Path Characters on Windows (Apostrophe/Spaces):**
   If the workspace folder contains special characters like apostrophes (e.g. `Parkinson's FoG`), SCons/GCC preprocessor command lines may fail during `.ino` conversion. Workaround:
   ```powershell
   subst P: "C:\path\to\Parkinson's FoG"
   cd P:\
   pio run -e esp32dev
   ```
2. **TFLite Micro Header Location:**
   Different versions of TFLM place `schema_generated.h` at either `tensorflow/lite/schema/schema_generated.h` or `tensorflow/lite/micro/schema/schema_generated.h`. The firmware headers use `#if __has_include(...)` to guarantee transparent cross-version compatibility.
3. **Internal SRAM vs PSRAM:**
   Do not allocate the tensor arena in external SPI PSRAM unless cache latency jitter is acceptable. Internal SRAM (`MALLOC_CAP_INTERNAL`) guarantees sub-4 ms inference execution.

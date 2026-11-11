# Hardware Wiring & Setup Guide

## 1. Bill of Materials (BOM)

| Component | Quantity | Description / Model | Approximate Cost |
| :--- | :--- | :--- | :--- |
| **Microcontroller** | 1 | ESP32 DevKit V1 (ESP32-WROOM-32, 240 MHz, 4 MB Flash) | ~$4.50 |
| **IMU Sensor** | 1 | InvenSense MPU6050 6-Axis IMU Module (GY-521 breakout) | ~$1.80 |
| **Haptic Actuator** | 1 | 3V DC Coin Vibration Motor (10 mm diameter, 2.7–3.3V) | ~$1.00 |
| **Switching Transistor**| 1 | 2N7000 or BSS138 N-Channel MOSFET (or 2N2222 NPN) | ~$0.20 |
| **Flyback Diode** | 1 | 1N4148 or 1N5819 Schottky Diode (protects against inductive kick) | ~$0.10 |
| **Resistors** | 2 | $1\text{ k}\Omega$ (Gate pull-down), $100\ \Omega$ (Gate series) | ~$0.10 |
| **Battery** | 1 | 3.7V 500 mAh LiPo Battery + TP4056 USB-C Charger Module | ~$5.00 |
| **Enclosure** | 1 | 3D-printed ankle strap enclosure or elastic hook-and-loop band | ~$2.00 |

---

## 2. Pinout & Electrical Connections

### MPU6050 to ESP32 DevKit V1 (I2C Bus)
```text
  MPU6050 (GY-521)              ESP32 DevKit V1
  +-------------+               +---------------+
  |         VCC | ------------> | 3.3V          |
  |         GND | ------------> | GND           |
  |         SCL | ------------> | GPIO 22 (SCL) |
  |         SDA | ------------> | GPIO 21 (SDA) |
  |         AD0 | ------------> | GND (Addr 0x68)|
  |         INT | ------------> | NC / GPIO 27  |
  +-------------+               +---------------+
```

### Haptic Motor Low-Side Driver Circuit (GPIO 13)
```text
  +3.3V / V_BAT
       |
       +-------+
       |       |
     [MTR]   [DIODE] (1N4148 Cathode to +, Anode to Drain)
       |       |
       +-------+
       |
     Drain
  GPIO 13 ---[100 Ohm]--- Gate (N-MOSFET)
                            |
                         [10k Ohm]
                            |
     Source                 |
       |--------------------+
       |
      GND
```

---

## 3. Sensor Ankle Mounting Instructions

1. **Placement Location:** Attach the MPU6050 securely to the lateral side of the shank, approximately 3–5 cm above the lateral malleolus of the ankle.
2. **Orientation Reference:**
   - **X-axis:** Longitudinal (aligned along the shank, pointing downward toward the foot: nominal $+1g$ or $-1g$ static gravity).
   - **Y-axis:** Anteroposterior (pointing forward along direction of forward progression).
   - **Z-axis:** Mediolateral (pointing perpendicular to sagittal plane).
3. **Rigid Coupling:** Ensure rigid mechanical contact with the limb using an elastic strap to prevent sensor resonance artifacts during vigorous foot strikes.
4. **Gyroscope Calibration:** At boot, remain stationary for 4 seconds while the firmware samples 200 readings to estimate and remove zero-rate bias.

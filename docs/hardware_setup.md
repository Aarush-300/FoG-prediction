# Hardware Wiring & Setup Guide

## 1. Bill of Materials (BOM)

| Component | Quantity | Description / Model | Approximate Cost |
| :--- | :--- | :--- | :--- |
| **Microcontroller** | 1 | ESP32 DevKit V1 (ESP32-WROOM-32, 240 MHz, 4 MB Flash) | ~$4.50 |
| **IMU Sensor** | 1 | InvenSense MPU6050 6-Axis IMU Module (GY-521 breakout) | ~$1.80 |
| **Haptic Actuator** | 1 | 3V DC Coin Vibration Motor (10 mm diameter, 2.7–3.3V) | ~$1.00 |
| **Logic-Level MOSFET**| 1 | **IRLML2502, IRLML6344, or AO3400** N-Channel SOT-23 MOSFET | ~$0.25 |
| **BJT Alternative** | 1 | 2N2222 or SS8050 NPN BJT (requires $1\text{ k}\Omega$ base resistor) | ~$0.15 |
| **Flyback Diode** | 1 | 1N4148 or 1N5819 Schottky Diode (snubs inductive voltage spikes) | ~$0.10 |
| **Gate Resistors** | 2 | $10\text{ k}\Omega$ (Gate pull-down), $100\ \Omega$ (Gate series current limiter) | ~$0.10 |
| **Power Decoupling** | 2 | $100\ \mu\text{F}$ electrolytic (motor supply rail), $0.1\ \mu\text{F}$ ceramic (MPU6050) | ~$0.20 |
| **Battery** | 1 | 3.7V 500 mAh LiPo Battery + TP4056 USB-C Charger Module | ~$5.00 |
| **Enclosure & Damping**| 1 | Ankle strap enclosure + vibration damping foam/silicone pad | ~$2.50 |

---

## 2. Low-Side Driver Circuit Design & Transistor Selection

### Why 2N7000 and BSS138 Must NOT Be Used
Standard signal MOSFETs like the **2N7000** or **BSS138** have gate threshold voltages ($V_{GS(th)}$) that can reach up to **$3.0\text{ V}$** (or $2.8\text{ V}$). When driven by the ESP32's **$3.3\text{ V}$** logic output, they operate barely above threshold in their high-resistance linear region with an on-resistance ($R_{DS(on)}$) of several ohms. This causes a severe voltage drop across the transistor, starving the motor of current, resulting in sluggish spin-up, weak vibration intensity, or complete failure to actuate.

**Recommended Logic-Level MOSFETs:**
- **IRLML2502:** $V_{GS(th)} \le 1.2\text{ V}$, $R_{DS(on)} < 0.045\ \Omega$ at $V_{GS} = 2.5\text{ V}$.
- **AO3400:** $V_{GS(th)} \le 1.4\text{ V}$, $R_{DS(on)} < 0.033\ \Omega$ at $V_{GS} = 2.5\text{ V}$.
- **IRLML6344:** $V_{GS(th)} \le 1.1\text{ V}$, ultra-low $R_{DS(on)}$.

These fully saturate at $3.3\text{ V}$ gate drive, passing full motor operating current (up to 80–100 mA) with negligible voltage drop ($< 5\text{ mV}$).

### Driver Circuit Schematic
```text
  +3.3V / V_BAT (Capped at ~3.0V RMS)
       |
       +-------------------+-----------------+
       |                   |                 |
     + |                 [MTR]           [DIODE] (1N4148 / 1N5819)
   [100 uF]           (Vibration)      (Cathode to +, Anode to Drain)
       |                   |                 |
       |                   +-----------------+
       |                   |
       |                 Drain
       |   GPIO 13 ---[100 Ohm]--- Gate (Logic-Level N-MOSFET)
       |                             |
       |                          [10k Ohm] (Gate Pull-Down)
       |                             |
       |                 Source      |
       +-------------------+---------+
                           |
                          GND
```

### Driver Component Rules:
1. **Gate Pull-Down Resistor ($10\text{ k}\Omega$):** Must be placed between Gate and GND. Holds the gate at $0\text{ V}$ during ESP32 bootloader and reset phases when GPIO 13 is in high-impedance tri-state mode, preventing spurious motor buzzing on power-up.
2. **NPN Transistor Option:** If using a BJT (such as 2N2222 or SS8050), replace the $100\ \Omega$ resistor with a **$1\text{ k}\Omega$ base resistor**. This limits base current from the $3.3\text{ V}$ GPIO to approximately $2.6\text{ mA}$ while ensuring sufficient base overdrive ($I_B \approx I_C / 30$) to saturate the transistor.
3. **Decoupling Capacitors:**
   - **$100\ \mu\text{F}$ Electrolytic:** Placed directly across the motor supply rail (+ to GND) near the motor leads. Absorbs transient inductive inrush currents when the motor starts, preventing voltage dips on the $3.3\text{ V}$ rail that cause ESP32 brownout resets.
   - **$0.1\ \mu\text{F}$ Ceramic:** Placed right at the MPU6050 VCC/GND pins to filter high-frequency switching noise from the I2C bus and motor driver.
4. **Motor Supply Voltage Capping:** Miniature coin vibration motors are rated for **$2.7\text{ V} - 3.3\text{ V}$** (nominal $3.0\text{ V}$). If powered directly from a single-cell LiPo battery ($3.7\text{ V}$ nominal, up to $4.2\text{ V}$ fully charged), average voltage must be capped at $\approx 3.0\text{ V}$ to avoid overheating and rapid brush failure. In firmware, the LEDC PWM duty cycle is capped via `HAPTIC_DEFAULT_DUTY_CYCLE 220` ($220/255 \approx 86\%$, yielding $4.2\text{V} \times 0.86 \approx 3.6\text{V}$ peak, $\approx 3.0\text{V}$ effective average). Alternatively, insert a silicon diode (e.g. 1N4001, $\approx 0.7\text{ V}$ drop) in series with the motor supply.

---

## 3. Mechanical Damping & Sensor Isolation

**Critical Physical Requirement:** Do NOT glue or rigidly bolt the vibration motor to the same PCB or rigid mount as the MPU6050 sensor.
- **Problem:** Miniature coin motors spin at 10,000–14,000 RPM, generating mechanical vibration between 150 Hz and 240 Hz. When rigidly coupled, these high-amplitude accelerations overwhelm the accelerometer silicon, causing signal saturation, baseline shift, and severe false freeze alarms.
- **Solution:** 
  1. Separate the motor mechanically from the IMU by at least 3–5 cm on the ankle strap.
  2. Mount the motor on a vibration-damping carrier (e.g. silicone pad, dense EVA foam tape, or rubber grommet).
  3. Firmware includes a **$1000\text{ ms}$ haptic blanking window** (`HAPTIC_BLANKING_MS 1000`) that holds the FSM state while the motor pulses and for 1 second afterwards to allow limb dynamics to settle.

---

## 4. Ankle Mounting & Coordinate Frame Alignment

### Coordinate Frame Orientation
The neural network was trained on leg IMU data recorded with the following nominal resting gravitational orientation:
- **$X$-axis:** Nominal $-1.006\text{ g}$ (points **UPWARD** along the shank, opposite to gravity).
- **$Y$-axis:** Nominal $+0.132\text{ g}$ (points forward along the progression direction).
- **$Z$-axis:** Nominal $+0.113\text{ g}$ (points lateral/outward from the limb).

```text
       ^  +X (Upward along shank, reads ~ -1.0g gravity)
       |
       |   +Y (Forward progression)
       +--->
      /
     v  +Z (Lateral / outward)
```

### On-Device Orientation Verification Command (`"o"`)
The firmware includes a dedicated mounting alignment utility accessible via the Serial Monitor (115200 baud):
1. Secure the strap to the lateral shank of the ankle.
2. Ask the user to stand stationary and still for 2 seconds.
3. Send the character `o` in the Serial Monitor.
4. The ESP32 collects 100 samples at 50 Hz, averages static acceleration, and prints:
   - Measured static acceleration vector $[a_x, a_y, a_z]$ in g.
   - Angular deviation relative to the training nominal vector $[-1.006, 0.132, 0.113]\text{ g}$.
5. **Acceptance Criteria:**
   - **$\le 15^\circ$:** Excellent alignment with training coordinate frame.
   - **$15^\circ - 30^\circ$:** Acceptable. Model retains detection capability.
   - **$> 30^\circ$:** High tilt. Adjust strap angle to prevent false alarms or missed episodes.

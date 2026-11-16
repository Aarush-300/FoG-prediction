# Scientific Diagnosis: Subject 22 Misdetection & Kinematic Analysis

**Investigation Objective:** Scientifically isolate the physical and physiological factors causing orientation-sensitive models to misdetect Freezing-of-Gait in Subject 22.

---

## 1. Verified Experimental Findings

1. **Rest-Estimated Static Gravity Alignment:**
   - **Population Nominal:** Gravity is oriented vertically along the leg: X = -0.987 g, Y = 0.111 g, Z = 0.113 g.
   - **Subject 20:** Gravity X = -0.994 g, Y = 0.097 g, Z = -0.012 g (Nominal vertical alignment).
   - **Subject 21:** Gravity X = -0.936 g, Y = 0.203 g, Z = 0.263 g (Nominal vertical alignment).
   - **Subject 22:** Gravity X = -0.960 g, Y = -0.253 g, Z = 0.115 g.
   - **Roll Tilt Misalignment:** Subject 22 displays an **extreme -65.6° roll offset** in the sensor frame. Lateral acceleration carries a persistent gravitational component of -0.253 g rather than nominal +0.111 g.

2. **Spectral Tremor Power & Freeze Index:**
   - In continuous trials, Subject 22 exhibits profound high-frequency trembling during freezing episodes:
     - Locomotion Power (0.5–3.0 Hz): Drops during freeze.
     - Freeze Band Power (3.0–8.0 Hz): Surges substantially.
     - Freeze Index (FI): Rises from 0.403 in normal gait to 1.584 during FoG (3.93x elevation).
   - **Conclusion:** The failure is **NOT** physiological akinetic freezing. Subject 22 produces intense trembling tremor (mean duration 5.94 s), but non-rotation-invariant convolution kernels failed because the gravity vector and tremor axes were projected onto orthogonal sensor channels.

3. **Remediation Strategy:**
   - Applying wide 3D rotation augmentation in physical Euclidean sensor coordinates before normalization.
   - Employing recurrent temporal tracking (e.g. GRU) to decouple gait phase and tremor oscillations from static sensor tilt.

// Configuration and Constants for ESP32 Freezing-of-Gait (FoG) Detector (v2 Hardened)
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdint.h>

// ==========================================
// Hardware Pinout (ESP32 DevKit V1)
// ==========================================
#define MPU6050_SDA_PIN          21   // ESP32 default I2C SDA
#define MPU6050_SCL_PIN          22   // ESP32 default I2C SCL
#define PIN_I2C_SDA              MPU6050_SDA_PIN
#define PIN_I2C_SCL              MPU6050_SCL_PIN

#define HAPTIC_MOTOR_PIN         13   // PWM output to logic MOSFET driver (IRLML2502/AO3400)
#define CALIBRATE_BUTTON_PIN     0    // On-board BOOT button (active LOW) for manual re-calibration
#define STATUS_LED_PIN           2    // On-board LED (GPIO 2 on ESP32 DevKit V1)

// ==========================================
// Sensor & Windowing Parameters
// ==========================================
#define IMU_SAMPLE_RATE_HZ       50   // 50 Hz target sampling frequency
#define MODEL_SAMPLE_RATE_HZ     IMU_SAMPLE_RATE_HZ
#define IMU_SAMPLE_PERIOD_MS     20   // 20 ms interval
#define MODEL_WINDOW_SAMPLES     100  // 2.0 seconds = 100 samples
#define MODEL_STRIDE_SAMPLES     25   // 0.5 seconds = 25 samples
#define MODEL_NUM_CHANNELS       6    // [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

// MPU6050 I2C Configuration & Filter
#define MPU6050_I2C_ADDR         0x68 // Default I2C address (AD0 to GND)
#define MPU6050_ACCEL_RANGE_G    2    // +/- 2g range (16384 LSB/g)
#define MPU6050_GYRO_RANGE_DPS   500  // +/- 500 deg/s range (65.5 LSB/dps)
#define MPU6050_ACCEL_LSB_PER_G  16384.0f
#define MPU6050_GYRO_LSB_PER_DPS 65.5f

// Digital Low-Pass Filter Configuration (Step 3.5: DLPF 4 = 21 Hz cutoff, 8.5 ms delay)
#define IMU_DLPF_CFG             4

// Compile-Time IMU Axis Rotation Matrix (Step 3.5: default identity, determinant +1)
// Applied to [acc_x, acc_y, acc_z] and [gyro_x, gyro_y, gyro_z]
static const float IMU_AXIS_ROTATION[3][3] = {
    { 1.0f,  0.0f,  0.0f },
    { 0.0f,  1.0f,  0.0f },
    { 0.0f,  0.0f,  1.0f }
};

// Nominal Training Orientation Reference Vector in g (Step 3.5)
static const float IMU_NOMINAL_GRAVITY_REF[3] = { -1.006f, 0.132f, 0.113f };

// ==========================================
// Z-Score Scaler Constants (v2 Retrained Offline)
// Channels: acc_x, acc_y, acc_z (in g), gyro_x, gyro_y, gyro_z (in deg/s)
// Source: export/v2/scaler.json
// ==========================================
static const float IMU_MEAN[6] = {
    -1.00663269f, 0.13048260f, 0.10839508f,
     2.39226532f, 0.97329688f, -0.59430683f
};

static const float IMU_STD[6] = {
    0.18871723f, 0.30785435f, 0.18671834f,
    45.17684174f, 22.88062096f, 62.82246780f
};

// ==========================================
// Model INT8 Quantization Constants
// Source: export/v2/quantization_constants.json
// ==========================================
#define MODEL_INPUT_SCALE        0.10611961f
#define MODEL_INPUT_ZERO_POINT   2
#define MODEL_OUTPUT_SCALE       0.00390625f
#define MODEL_OUTPUT_ZERO_POINT  -128

// ==========================================
// Temporal Alarm Post-Processing Presets (Step 3.1)
// Compile-time switch: FSM_PRESET (1=BALANCED, 2=LOW_ALARM)
// ==========================================
#define FSM_PRESET_BALANCED      1
#define FSM_PRESET_LOW_ALARM     2

#ifndef FSM_PRESET
#define FSM_PRESET FSM_PRESET_BALANCED
#endif

// Re-arm policy enumeration
#define REARM_POLICY_TIMEOUT          0
#define REARM_POLICY_TIMEOUT_AND_LOW  1

#if (FSM_PRESET == FSM_PRESET_BALANCED)
    // Preset: BALANCED (Default Deployed Configuration)
    #define FSM_PRESET_NAME              "BALANCED"
    #define FSM_CONFIRM_CONSECUTIVE      2
    #define FSM_EMA_ALPHA                0.60f
    #define FSM_HYSTERESIS_HIGH          0.55f
    #define FSM_HYSTERESIS_LOW           0.35f
    #define FSM_COOLDOWN_MS              6000
    #define FSM_REARM_POLICY             REARM_POLICY_TIMEOUT_AND_LOW
    #define FSM_RECUE_INTERVAL_S         0
#elif (FSM_PRESET == FSM_PRESET_LOW_ALARM)
    // Preset: LOW_ALARM (Fallback Configuration for High Alert-Fatigue Sensitivity)
    #define FSM_PRESET_NAME              "LOW_ALARM"
    #define FSM_CONFIRM_CONSECUTIVE      2
    #define FSM_EMA_ALPHA                0.60f
    #define FSM_HYSTERESIS_HIGH          0.70f
    #define FSM_HYSTERESIS_LOW           0.15f
    #define FSM_COOLDOWN_MS              6000
    #define FSM_REARM_POLICY             REARM_POLICY_TIMEOUT_AND_LOW
    #define FSM_RECUE_INTERVAL_S         0
#else
    #error "Invalid FSM_PRESET selected. Choose FSM_PRESET_BALANCED or FSM_PRESET_LOW_ALARM."
#endif

// Backward compatibility defines mapped to active preset
#define MODEL_HYSTERESIS_HIGH        FSM_HYSTERESIS_HIGH
#define MODEL_HYSTERESIS_LOW         FSM_HYSTERESIS_LOW
#define EMA_SMOOTHING_ALPHA          FSM_EMA_ALPHA
#define CONSECUTIVE_CONFIRMATIONS    FSM_CONFIRM_CONSECUTIVE
#define FOG_CONFIRM_CONSECUTIVE      FSM_CONFIRM_CONSECUTIVE
#define ALARM_COOLDOWN_MS            FSM_COOLDOWN_MS
#define FOG_COOLDOWN_MS              ALARM_COOLDOWN_MS

// Documented: Deprecated raw window threshold from v1 exploratory phase; unused in FSM post-processing.
#define DEFAULT_DECISION_THRESHOLD   0.44f

// ==========================================
// Haptic Cueing & Blanking Parameters (Step 3.4)
// ==========================================
#define HAPTIC_PWM_CHANNEL           0      // ESP32 LEDC PWM channel
#define HAPTIC_PWM_FREQ_HZ           1000   // PWM frequency (1 kHz)
#define HAPTIC_PWM_RESOLUTION_BITS   8      // 8-bit resolution (0-255)
#define HAPTIC_DEFAULT_DUTY_CYCLE    220    // ~86% power intensity (capped for 3V motor limit)
#define HAPTIC_PULSE_DURATION_MS     120    // Cadence pulse duration: 120 ms
#define HAPTIC_PULSE_PAUSE_MS        420    // Inter-pulse pause: 420 ms
#define HAPTIC_PULSES_PER_CUE        8      // 8 pulses rhythmic cue
#define HAPTIC_BLANKING_MS           1000   // Hold FSM state during vibration and 1000 ms afterwards

// ==========================================
// Dynamic TFLM Runtime Arena Size Bounds (Step 3.2)
// ==========================================
#define TENSOR_ARENA_INITIAL_BYTES   (220 * 1024)
#define TENSOR_ARENA_FLOOR_BYTES     (100 * 1024)

#endif // MODEL_CONFIG_H

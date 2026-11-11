// Configuration and Constants for ESP32 Freezing-of-Gait (FoG) Predictor
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

#define HAPTIC_MOTOR_PIN         13   // PWM output to MOSFET/transistor driver for vibration motor
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

// MPU6050 I2C Configuration
#define MPU6050_I2C_ADDR         0x68 // Default I2C address (AD0 to GND)
#define MPU6050_ACCEL_RANGE_G    2    // +/- 2g range (16384 LSB/g)
#define MPU6050_GYRO_RANGE_DPS   500  // +/- 500 deg/s range (65.5 LSB/dps)
#define MPU6050_ACCEL_LSB_PER_G  16384.0f
#define MPU6050_GYRO_LSB_PER_DPS 65.5f

// ==========================================
// Z-Score Scaler Constants (Trained Offline)
// ==========================================
// Channels: acc_x, acc_y, acc_z (in g), gyro_x, gyro_y, gyro_z (in deg/s)
const float IMU_MEAN[6] = {
    -1.00592959f, 0.13175330f, 0.11280084f,
     2.14400911f, 1.10965145f, -0.69512039f
};

const float IMU_STD[6] = {
    0.19035448f, 0.31152907f, 0.18985811f,
    45.32674789f, 23.22983932f, 63.61863708f
};

// ==========================================
// Temporal Alarm Post-Processing Parameters
// ==========================================
#define DEFAULT_DECISION_THRESHOLD   0.44f  // Clinically tuned on validation PR-curve
#define MODEL_HYSTERESIS_HIGH        0.49f  // Triggers alert
#define MODEL_HYSTERESIS_LOW         0.29f  // Resets alert state
#define EMA_SMOOTHING_ALPHA          0.60f  // Weight for exponential moving average
#define CONSECUTIVE_CONFIRMATIONS    2      // Consecutive inferences >= high_th before alarm
#define FOG_CONFIRM_CONSECUTIVE      CONSECUTIVE_CONFIRMATIONS
#define ALARM_COOLDOWN_MS            3000   // Cooldown period between distinct episodes
#define FOG_COOLDOWN_MS              ALARM_COOLDOWN_MS

// ==========================================
// Haptic Cueing Parameters
// ==========================================
#define HAPTIC_PWM_CHANNEL           0      // ESP32 LEDC PWM channel
#define HAPTIC_PWM_FREQ_HZ           1000   // PWM frequency (1 kHz)
#define HAPTIC_PWM_RESOLUTION_BITS   8      // 8-bit resolution (0-255)
#define HAPTIC_DEFAULT_DUTY_CYCLE    220    // ~86% power intensity
#define HAPTIC_PULSE_DURATION_MS     150    // Cadence pulse duration
#define HAPTIC_PULSE_PAUSE_MS        100    // Inter-pulse pause
#define HAPTIC_PULSES_PER_CUE        3      // Triple-pulse rhythmic auditory/tactile cue

// ==========================================
// TFLM Runtime Arena Size for ESP32
// ==========================================
// Estimated arena: 199.2 KB. Allocated with 10% safety margin in ESP32 internal SRAM.
#define TENSOR_ARENA_SIZE_BYTES      (220 * 1024) 

#endif // MODEL_CONFIG_H

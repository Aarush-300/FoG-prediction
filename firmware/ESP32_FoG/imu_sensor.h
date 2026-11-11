#ifndef IMU_SENSOR_H
#define IMU_SENSOR_H

#include <Arduino.h>
#include <Wire.h>
#include "model_config.h"

/**
 * @brief Register-level Driver for InvenSense MPU6050 6-Axis IMU on ESP32.
 * 
 * Configures I2C Fast Mode (400 kHz), DLPF 44 Hz, 50 Hz sample rate,
 * verifies device identity (WHO_AM_I == 0x68), calibrates stationary gyroscope bias,
 * and outputs physical units: Accel in g, Gyro in deg/s.
 */
class IMUSensor {
public:
    IMUSensor();
    
    // Initialize I2C, verify WHO_AM_I, and configure registers
    bool begin(int sda_pin = MPU6050_SDA_PIN, int scl_pin = MPU6050_SCL_PIN);
    
    // Stationary gyroscope zero-rate bias calibration
    bool calibrate(int num_samples = 200);
    bool calibrateGyro(int num_samples = 200) { return calibrate(num_samples); }
    
    // Read 6-axis IMU sample in physical units: [ax, ay, az, gx, gy, gz]
    // Accel in g (+/- 2g range), Gyro in deg/s (+/- 500 dps range)
    bool readSample(float* out_sample6);
    
    // Status queries
    bool isCalibrated() const { return m_calibrated; }
    float getGyroBiasX() const { return m_gyro_bias_x; }
    float getGyroBiasY() const { return m_gyro_bias_y; }
    float getGyroBiasZ() const { return m_gyro_bias_z; }
    
private:
    bool writeRegister(uint8_t reg, uint8_t value);
    bool readRegisters(uint8_t reg, uint8_t count, uint8_t* dest);
    
    uint8_t m_i2c_addr;
    bool m_initialized;
    bool m_calibrated;
    
    float m_gyro_bias_x;
    float m_gyro_bias_y;
    float m_gyro_bias_z;
};

#endif // IMU_SENSOR_H

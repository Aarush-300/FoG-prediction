#ifndef IMU_SENSOR_H
#define IMU_SENSOR_H

#include <Arduino.h>
#include <Wire.h>
#include "model_config.h"

class IMUSensor {
public:
    IMUSensor();
    
    // Initialize I2C interface, verify WHO_AM_I, set DLPF (21 Hz) and sample rates
    bool begin(int sda_pin = PIN_I2C_SDA, int scl_pin = PIN_I2C_SCL);
    
    // Stationary gyroscope bias calibration
    bool calibrateGyro(int num_samples = 200);
    
    // Read calibrated and axis-rotated 6-channel IMU vector
    // [acc_x, acc_y, acc_z] in g, [gyro_x, gyro_y, gyro_z] in deg/s
    // Returns true on successful I2C read, false on bus error
    bool readSample(float* out_sample6);
    
    // Measure stationary orientation over 2 seconds (100 samples at 50 Hz)
    // Computes average acceleration vector and angle vs training reference [-1.006, 0.132, 0.113] g
    bool measureStationaryOrientation(float out_acc[3], float* out_angle_deg, int num_samples = 100);
    
    // Sensor status queries
    bool isInitialized() const { return m_initialized; }
    bool isCalibrated() const { return m_calibrated; }
    
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

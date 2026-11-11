#include "imu_sensor.h"

// MPU6050 Register Map
#define MPU6050_REG_SMPLRT_DIV    0x19
#define MPU6050_REG_CONFIG        0x1A
#define MPU6050_REG_GYRO_CONFIG   0x1B
#define MPU6050_REG_ACCEL_CONFIG  0x1C
#define MPU6050_REG_ACCEL_XOUT_H  0x3B
#define MPU6050_REG_PWR_MGMT_1    0x6B
#define MPU6050_REG_WHO_AM_I      0x75

IMUSensor::IMUSensor()
    : m_i2c_addr(MPU6050_I2C_ADDR),
      m_initialized(false),
      m_calibrated(false),
      m_gyro_bias_x(0.0f),
      m_gyro_bias_y(0.0f),
      m_gyro_bias_z(0.0f)
{
}

bool IMUSensor::begin(int sda_pin, int scl_pin) {
    Wire.begin(sda_pin, scl_pin, 400000); // 400 kHz Fast Mode

    // 1. Verify WHO_AM_I device identity
    uint8_t who_am_i = 0;
    if (!readRegisters(MPU6050_REG_WHO_AM_I, 1, &who_am_i)) {
        Serial.println(F("[IMU] Error: Failed to communicate with MPU6050 over I2C!"));
        return false;
    }
    if (who_am_i != 0x68 && who_am_i != 0x69 && who_am_i != 0x70 && who_am_i != 0x72) {
        Serial.printf("[IMU] Error: Unexpected WHO_AM_I ID: 0x%02X (Expected 0x68)\n", who_am_i);
        return false;
    }
    Serial.printf("[IMU] Device identified: WHO_AM_I = 0x%02X\n", who_am_i);

    // 2. Wake up device (clear SLEEP bit, select PLL with X axis gyroscope reference)
    if (!writeRegister(MPU6050_REG_PWR_MGMT_1, 0x01)) return false;
    delay(10);

    // 3. Configure Digital Low-Pass Filter (DLPF_CFG = 3 -> 44 Hz Accel / 42 Hz Gyro bandwidth)
    // Prevents high-frequency aliasing at 50 Hz sampling frequency
    if (!writeRegister(MPU6050_REG_CONFIG, 0x03)) return false;

    // 4. Configure Sample Rate Divider (SMPLRT_DIV = 19 -> 1000 Hz / (1 + 19) = 50 Hz internal sampling)
    if (!writeRegister(MPU6050_REG_SMPLRT_DIV, 0x13)) return false;

    // 5. Configure Accelerometer Full Scale (+/- 2g -> 16,384 LSB/g)
    if (!writeRegister(MPU6050_REG_ACCEL_CONFIG, 0x00)) return false;

    // 6. Configure Gyroscope Full Scale (+/- 500 deg/s -> 65.5 LSB/deg/s)
    if (!writeRegister(MPU6050_REG_GYRO_CONFIG, 0x08)) return false;

    m_initialized = true;
    Serial.println(F("[IMU] MPU6050 configured successfully (+/-2g, +/-500dps, DLPF 44Hz, 50Hz ODR)."));
    return true;
}

bool IMUSensor::calibrate(int num_samples) {
    if (!m_initialized) {
        Serial.println(F("[IMU] Error: Cannot calibrate uninitialized sensor!"));
        return false;
    }

    Serial.printf("[IMU] Calibrating gyroscope bias (%d samples)... Keep sensor stationary!\n", num_samples);
    float sum_gx = 0.0f, sum_gy = 0.0f, sum_gz = 0.0f;
    int valid = 0;

    for (int i = 0; i < num_samples; i++) {
        uint8_t raw[14];
        if (readRegisters(MPU6050_REG_ACCEL_XOUT_H, 14, raw)) {
            int16_t rgx = (int16_t)((raw[8]  << 8) | raw[9]);
            int16_t rgy = (int16_t)((raw[10] << 8) | raw[11]);
            int16_t rgz = (int16_t)((raw[12] << 8) | raw[13]);

            sum_gx += (float)rgx / MPU6050_GYRO_LSB_PER_DPS;
            sum_gy += (float)rgy / MPU6050_GYRO_LSB_PER_DPS;
            sum_gz += (float)rgz / MPU6050_GYRO_LSB_PER_DPS;
            valid++;
        }
        delay(10);
    }

    if (valid < num_samples / 2) {
        Serial.println(F("[IMU] Error: Calibration failed due to excessive I2C read errors!"));
        return false;
    }

    m_gyro_bias_x = sum_gx / (float)valid;
    m_gyro_bias_y = sum_gy / (float)valid;
    m_gyro_bias_z = sum_gz / (float)valid;
    m_calibrated = true;

    Serial.printf("[IMU] Calibration complete. Gyro Biases (deg/s): X=%.2f, Y=%.2f, Z=%.2f\n",
                  m_gyro_bias_x, m_gyro_bias_y, m_gyro_bias_z);
    return true;
}

bool IMUSensor::readSample(float* out_sample6) {
    if (!m_initialized || !out_sample6) return false;

    // Burst read 14 bytes: AccX (2), AccY (2), AccZ (2), Temp (2), GyroX (2), GyroY (2), GyroZ (2)
    uint8_t raw[14];
    if (!readRegisters(MPU6050_REG_ACCEL_XOUT_H, 14, raw)) {
        return false;
    }

    int16_t rax = (int16_t)((raw[0]  << 8) | raw[1]);
    int16_t ray = (int16_t)((raw[2]  << 8) | raw[3]);
    int16_t raz = (int16_t)((raw[4]  << 8) | raw[5]);

    int16_t rgx = (int16_t)((raw[8]  << 8) | raw[9]);
    int16_t rgy = (int16_t)((raw[10] << 8) | raw[11]);
    int16_t rgz = (int16_t)((raw[12] << 8) | raw[13]);

    // Accelerometer: convert raw LSB to g
    out_sample6[0] = (float)rax / MPU6050_ACCEL_LSB_PER_G;
    out_sample6[1] = (float)ray / MPU6050_ACCEL_LSB_PER_G;
    out_sample6[2] = (float)raz / MPU6050_ACCEL_LSB_PER_G;

    // Gyroscope: convert raw LSB to deg/s and subtract stationary bias
    out_sample6[3] = ((float)rgx / MPU6050_GYRO_LSB_PER_DPS) - m_gyro_bias_x;
    out_sample6[4] = ((float)rgy / MPU6050_GYRO_LSB_PER_DPS) - m_gyro_bias_y;
    out_sample6[5] = ((float)rgz / MPU6050_GYRO_LSB_PER_DPS) - m_gyro_bias_z;

    return true;
}

bool IMUSensor::writeRegister(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(m_i2c_addr);
    Wire.write(reg);
    Wire.write(value);
    return (Wire.endTransmission() == 0);
}

bool IMUSensor::readRegisters(uint8_t reg, uint8_t count, uint8_t* dest) {
    Wire.beginTransmission(m_i2c_addr);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;

    uint8_t bytes_read = Wire.requestFrom(m_i2c_addr, count);
    if (bytes_read != count) return false;

    for (uint8_t i = 0; i < count; i++) {
        dest[i] = Wire.read();
    }
    return true;
}

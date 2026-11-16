/*
 * ======================================================================================
 * PARKINSON'S DISEASE FREEZING OF GAIT (FoG) REAL-TIME DETECTION SYSTEM (v2 Hardened)
 * Target Hardware: ESP32 DevKit V1 (ESP32-WROOM-32 @ 240MHz) + InvenSense MPU6050
 * Operating System: FreeRTOS Dual-Core Real-Time Architecture
 * 
 * Core 0: Real-time 50 Hz IMU acquisition, 21 Hz DLPF, axis rotation, and ring buffer
 * Core 1: INT8 TFLite Micro inference (500 ms stride), 4-state FoG FSM,
 *         thread-safe interpreter mutex, and haptic cueing with 1000 ms blanking.
 * ======================================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <esp_timer.h>
#include "model_config.h"
#include "model_data.h"
#include "imu_sensor.h"
#include "preprocessing.h"
#include "model_inference.h"
#include "fog_detection.h"
#include "haptic_feedback.h"
#include "golden_vectors.h"

// System Components
static IMUSensor            g_imu;
static Preprocessing        g_buffer;
static ModelInferenceEngine g_engine;
static FoGDetectionEngine   g_fog_detector;
static HapticFeedback       g_haptic;

// FreeRTOS Synchronization Handles
static TaskHandle_t      g_sensor_task_handle = nullptr;
static TaskHandle_t      g_inference_task_handle = nullptr;
static SemaphoreHandle_t g_buffer_mutex = nullptr;
static SemaphoreHandle_t g_predict_mutex = nullptr;

// Diagnostic Statistics
static volatile uint32_t      g_total_inferences = 0;
static volatile uint32_t      g_total_fog_events = 0;
static volatile unsigned long g_last_inference_latency_us = 0;
static volatile uint32_t      g_i2c_consecutive_errors = 0;

// Forward Declarations
void taskSensorAcquisition(void* pvParameters);
void taskInferenceAndControl(void* pvParameters);
void runOnDeviceSelfTest();
void runBenchmarkCommand();
void runOrientationCheckCommand();

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println();
    Serial.println(F("=========================================================="));
    Serial.println(F("  ESP32 Parkinson's Freezing-of-Gait Detector (v2 Hardened)"));
    Serial.printf( "  Preset: %s | DLPF: 21 Hz | Arena: Dynamic 16-Byte Aligned\n", FSM_PRESET_NAME);
    Serial.println(F("=========================================================="));

    // 1. Initialize FreeRTOS Mutexes
    g_buffer_mutex = xSemaphoreCreateMutex();
    g_predict_mutex = xSemaphoreCreateMutex();
    if (g_buffer_mutex == nullptr || g_predict_mutex == nullptr) {
        Serial.println(F("[ERROR] Failed to create FreeRTOS mutexes!"));
        while (1) delay(1000);
    }

    // 2. Initialize Haptic Driver (LEDC PWM on GPIO 13, 8 pulses default, 1000 ms blanking)
    g_haptic.begin();

    // 3. Initialize Preprocessing Ring Buffer
    g_buffer.begin();

    // 4. Initialize FoG Detection State Machine
    g_fog_detector.begin(
        FSM_HYSTERESIS_HIGH,
        FSM_HYSTERESIS_LOW,
        FSM_CONFIRM_CONSECUTIVE,
        FSM_COOLDOWN_MS,
        FSM_REARM_POLICY
    );

    // 5. Initialize I2C (400 kHz) and MPU6050 IMU
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    Serial.println(F("[INIT] Initializing MPU6050 IMU sensor..."));
    if (!g_imu.begin(PIN_I2C_SDA, PIN_I2C_SCL)) {
        Serial.println(F("[ERROR] MPU6050 connection failed! Check wiring:"));
        Serial.printf("  VCC->3.3V, GND->GND, SDA->GPIO%d, SCL->GPIO%d\n", PIN_I2C_SDA, PIN_I2C_SCL);
        g_haptic.triggerEmergencyAlert();
        while (1) delay(1000);
    }

    // Calibrate stationary gyroscope bias
    Serial.println(F("[CALIB] Keep sensor still on ankle for gyroscope bias calibration..."));
    g_imu.calibrateGyro(200);

    // 6. Initialize TensorFlow Lite Micro Model
    Serial.println(F("[INIT] Loading TFLM INT8 Neural Network Model..."));
    if (!g_engine.begin(g_fog_model_data, g_fog_model_data_len)) {
        Serial.println(F("[ERROR] Failed to initialize TFLite Micro model!"));
        while (1) delay(1000);
    }

    // 7. Execute On-Device Self-Test using 10 Golden Vectors
    runOnDeviceSelfTest();

    // 8. Launch Dual-Core FreeRTOS Tasks
    Serial.println(F("[SYSTEM] Launching dual-core real-time tasks:"));
    
    // Core 0: 50 Hz IMU Acquisition Task (Higher Priority: 3)
    xTaskCreatePinnedToCore(
        taskSensorAcquisition,
        "SensorTask",
        4096,
        nullptr,
        3,
        &g_sensor_task_handle,
        0   // Core 0
    );
    Serial.println(F("  - Sensor Acquisition Task pinned to Core 0 (Priority 3, 50 Hz)"));

    // Core 1: Inference & Haptic Control Task (Priority: 2)
    xTaskCreatePinnedToCore(
        taskInferenceAndControl,
        "InferenceTask",
        8192,
        nullptr,
        2,
        &g_inference_task_handle,
        1   // Core 1
    );
    Serial.println(F("  - Inference & Control Task pinned to Core 1 (Priority 2, 500 ms stride)"));

    Serial.println(F("[SYSTEM] System fully armed and monitoring gait dynamics."));
    Serial.println(F("Commands: 't' -> Self-Test | 'b' -> Benchmark | 'o' -> Mounting Check | 'c' -> Calib | 's' -> Status"));
    Serial.println(F("----------------------------------------------------------"));
}

void loop() {
    // Interactive Serial Monitor Commands & Diagnostics
    if (Serial.available()) {
        char cmd = (char)Serial.read();
        if (cmd == 't' || cmd == 'T') {
            runOnDeviceSelfTest();
        } else if (cmd == 'b' || cmd == 'B') {
            runBenchmarkCommand();
        } else if (cmd == 'o' || cmd == 'O') {
            runOrientationCheckCommand();
        } else if (cmd == 'c' || cmd == 'C') {
            Serial.println(F("[CALIB] Re-calibrating gyro bias... Keep still!"));
            g_imu.calibrateGyro(200);
        } else if (cmd == 's' || cmd == 'S') {
            Serial.printf("[STATUS] Preset: %s | Inferences: %u | FoG Events: %u | Last Latency: %.2f ms | Free Heap: %u bytes\n",
                          FSM_PRESET_NAME,
                          g_total_inferences, g_total_fog_events,
                          g_last_inference_latency_us / 1000.0f,
                          esp_get_free_heap_size());
        }
    }
    vTaskDelay(pdMS_TO_TICKS(100));
}

/**
 * FreeRTOS Task on Core 0: High-Precision 50 Hz IMU Sampling
 * Executes precisely every 20 ms using FreeRTOS vTaskDelayUntil.
 * Includes I2C error detection and failsafe emergency alert.
 */
void taskSensorAcquisition(void* pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(1000 / MODEL_SAMPLE_RATE_HZ); // 20 ms
    
    float sample[6];

    while (true) {
        // Read calibrated and axis-rotated physical units (g and deg/s)
        if (g_imu.readSample(sample)) {
            g_i2c_consecutive_errors = 0;
            // Protect ring buffer write with mutex
            if (xSemaphoreTake(g_buffer_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                g_buffer.pushSample(sample);
                xSemaphoreGive(g_buffer_mutex);
            }
        } else {
            g_i2c_consecutive_errors++;
            if (g_i2c_consecutive_errors >= 25) { // 500 ms continuous I2C failure
                Serial.println(F("[ERROR] I2C bus stall detected! Triggering emergency alert..."));
                g_haptic.triggerEmergencyAlert();
                g_i2c_consecutive_errors = 0;
            }
        }
        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

/**
 * FreeRTOS Task on Core 1: Periodic 500 ms Inference & Haptic Control
 * Evaluates the 100x6 temporal window and actuates haptic motor on FoG alert.
 * Implements 1000 ms blanking to ignore motor vibration artifacts.
 */
void taskInferenceAndControl(void* pvParameters) {
    static int8_t window_input_quantized[MODEL_WINDOW_SAMPLES * MODEL_NUM_CHANNELS];
    
    while (true) {
        // Update non-blocking haptic pulse state machine
        g_haptic.update();

        // Check if 25 new samples (0.5 s stride) have accumulated
        bool ready_for_inference = false;
        if (xSemaphoreTake(g_buffer_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            if (g_buffer.isWindowReady()) {
                g_buffer.getQuantizedWindow(window_input_quantized);
                g_buffer.consumeStride();
                ready_for_inference = true;
            }
            xSemaphoreGive(g_buffer_mutex);
        }

        if (ready_for_inference) {
            unsigned long latency_us = 0;
            float raw_prob = 0.0f;

            // Thread-safe inference invocation
            if (xSemaphoreTake(g_predict_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                raw_prob = g_engine.predict(window_input_quantized, &latency_us);
                xSemaphoreGive(g_predict_mutex);
            }

            g_last_inference_latency_us = latency_us;
            g_total_inferences++;

            // If haptic blanking is active, hold FSM state and suppress new alerts
            if (g_haptic.isBlankingActive()) {
                // Blanking active: motor is vibrating or refractory settling in progress
            } else {
                // Process probability through dual-threshold hysteresis & cooldown
                FoGState state = g_fog_detector.update(raw_prob);
                float smooth_prob = g_fog_detector.getSmoothedProbability();

                if (state == FOG_STATE_ACTIVE_ALERT) {
                    g_total_fog_events++;
                    // Actuate 8-pulse rhythmic cadence cueing
                    g_haptic.triggerCadenceCue();
                    Serial.printf(">>> [ALERT] FREEZING OF GAIT DETECTED! Prob: %.3f (Raw: %.3f) | Latency: %.2f ms <<<\n",
                                  smooth_prob, raw_prob, latency_us / 1000.0f);
                } else if (state == FOG_STATE_COOLDOWN) {
                    // Refractory cooldown
                } else {
                    // Normal gait
                    if (g_total_inferences % 10 == 0) {
                        Serial.printf("[GAIT] Normal | P(FoG): %.3f | Latency: %.2f ms | Free Heap: %u bytes\n",
                                      smooth_prob, latency_us / 1000.0f, esp_get_free_heap_size());
                    }
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

/**
 * Self-Test Routine: Validates TFLM engine against all 10 Golden Vectors
 */
void runOnDeviceSelfTest() {
    Serial.println(F("\n--- RUNNING ON-DEVICE GOLDEN VECTOR SELF-TEST (10 Vectors) ---"));
    int passed = 0;
    for (int i = 0; i < 10; i++) {
        unsigned long t_us = 0;
        float p = 0.0f;
        if (xSemaphoreTake(g_predict_mutex, portMAX_DELAY) == pdTRUE) {
            p = g_engine.predict(g_golden_vectors[i].window_data, &t_us);
            xSemaphoreGive(g_predict_mutex);
        }
        float err = fabsf(p - g_golden_vectors[i].expected_prob);
        bool match = (err <= g_golden_vectors[i].tolerance);
        
        Serial.printf("  Vector %2d [%s]: Expected P=%.3f, Infer P=%.3f (Err: %.4f, Tol: %.2f) Latency=%.2f ms -> %s\n",
                      i + 1, g_golden_vectors[i].description,
                      g_golden_vectors[i].expected_prob, p, err, g_golden_vectors[i].tolerance,
                      t_us / 1000.0f, match ? "PASS" : "FAIL");
        if (match) {
            passed++;
        }
    }
    Serial.printf("--- SELF-TEST RESULT: %d / 10 TEST VECTORS VERIFIED ---\n\n", passed);
}

/**
 * Step 3.6: Benchmark Command ('b')
 * Runs 100 inferences on golden vector, measures latency, arena usage, heap, and stack high water marks.
 */
void runBenchmarkCommand() {
    Serial.println(F("\n--- RUNNING 100-INFERENCE BENCHMARK ('b') ---"));
    const int N_BENCH = 100;
    int64_t total_us = 0;
    int64_t min_us = 9999999;
    int64_t max_us = 0;

    for (int i = 0; i < N_BENCH; i++) {
        int64_t start = esp_timer_get_time();
        if (xSemaphoreTake(g_predict_mutex, portMAX_DELAY) == pdTRUE) {
            g_engine.predict(g_golden_vectors[0].window_data);
            xSemaphoreGive(g_predict_mutex);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        total_us += elapsed;
        if (elapsed < min_us) min_us = elapsed;
        if (elapsed > max_us) max_us = elapsed;
        vTaskDelay(pdMS_TO_TICKS(5));
    }

    float mean_ms = (total_us / (float)N_BENCH) / 1000.0f;
    float min_ms = min_us / 1000.0f;
    float max_ms = max_us / 1000.0f;

    Serial.printf("Inference Latency (100 runs): Min: %.2f ms | Mean: %.2f ms | Max: %.2f ms\n",
                  min_ms, mean_ms, max_ms);
    Serial.printf("TFLM Arena Allocated: %u bytes | Used: %u bytes\n",
                  (unsigned int)g_engine.getArenaSize(),
                  (unsigned int)g_engine.getArenaUsedBytes());
    Serial.printf("Free Heap: %u bytes | Minimum Free Heap: %u bytes\n",
                  esp_get_free_heap_size(),
                  esp_get_minimum_free_heap_size());

    if (g_sensor_task_handle != nullptr) {
        Serial.printf("Sensor Task Stack High-Water Mark: %u bytes\n",
                      uxTaskGetStackHighWaterMark(g_sensor_task_handle) * 4);
    }
    if (g_inference_task_handle != nullptr) {
        Serial.printf("Inference Task Stack High-Water Mark: %u bytes\n",
                      uxTaskGetStackHighWaterMark(g_inference_task_handle) * 4);
    }
    Serial.println(F("--- BENCHMARK COMPLETE ---\n"));
}

/**
 * Step 3.5: Orientation Check Command ('o')
 * Averages 2 s of stationary acceleration (100 samples) and computes angle vs training mean [-1.006, 0.132, 0.113] g.
 */
void runOrientationCheckCommand() {
    Serial.println(F("\n[ORIENTATION] Measuring 2 seconds stationary acceleration... Keep still!"));
    float meas_acc[3] = {0.0f, 0.0f, 0.0f};
    float angle_deg = 0.0f;

    if (g_imu.measureStationaryOrientation(meas_acc, &angle_deg, 100)) {
        Serial.printf("Measured Static Accel (g): X=%.3f, Y=%.3f, Z=%.3f (Norm: %.3f g)\n",
                      meas_acc[0], meas_acc[1], meas_acc[2],
                      sqrtf(meas_acc[0]*meas_acc[0] + meas_acc[1]*meas_acc[1] + meas_acc[2]*meas_acc[2]));
        Serial.printf("Training Reference   (g): X=%.3f, Y=%.3f, Z=%.3f\n",
                      IMU_NOMINAL_GRAVITY_REF[0], IMU_NOMINAL_GRAVITY_REF[1], IMU_NOMINAL_GRAVITY_REF[2]);
        Serial.printf("Mounting Angle Deviation: %.1f degrees\n", angle_deg);

        if (angle_deg <= 15.0f) {
            Serial.println(F("[ORIENTATION] -> Status: EXCELLENT alignment with training coordinate frame."));
        } else if (angle_deg <= 30.0f) {
            Serial.println(F("[ORIENTATION] -> Status: ACCEPTABLE. Minor tilt within baseline tolerance."));
        } else {
            Serial.println(F("[ORIENTATION] -> WARNING: High tilt (>30 deg)! Check wearable strap and orientation."));
        }
    } else {
        Serial.println(F("[ORIENTATION] Error: Measurement failed due to I2C errors."));
    }
    Serial.println();
}

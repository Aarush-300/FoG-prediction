/*
 * ======================================================================================
 * PARKINSON'S DISEASE FREEZING OF GAIT (FoG) REAL-TIME DETECTION SYSTEM
 * Target Hardware: ESP32 DevKit V1 (ESP32-WROOM-32 @ 240MHz) + InvenSense MPU6050
 * Operating System: FreeRTOS Dual-Core Real-Time Architecture
 * 
 * Core 0: Real-time 50 Hz IMU acquisition, I2C fast-mode reading, and ring buffer
 * Core 1: INT8 TFLite Micro inference (500 ms stride), 4-state FoG FSM,
 *         and non-blocking LEDC PWM rhythmic haptic cueing.
 * ======================================================================================
 */

#include <Arduino.h>
#include <Wire.h>
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

// Diagnostic Statistics
static volatile uint32_t      g_total_inferences = 0;
static volatile uint32_t      g_total_fog_events = 0;
static volatile unsigned long g_last_inference_latency_us = 0;

// Forward Declarations
void taskSensorAcquisition(void* pvParameters);
void taskInferenceAndControl(void* pvParameters);
void runOnDeviceSelfTest();

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println();
    Serial.println(F("=========================================================="));
    Serial.println(F("  ESP32 Parkinson's Freezing-of-Gait Detection System     "));
    Serial.println(F("  Target: ESP32-WROOM-32 @ 240MHz | Sensor: MPU6050      "));
    Serial.println(F("=========================================================="));

    // 1. Initialize FreeRTOS Mutex for Ring Buffer protection
    g_buffer_mutex = xSemaphoreCreateMutex();
    if (g_buffer_mutex == nullptr) {
        Serial.println(F("[ERROR] Failed to create FreeRTOS mutex!"));
        while (1) delay(1000);
    }

    // 2. Initialize Haptic Driver (LEDC PWM on GPIO 13)
    g_haptic.begin();

    // 3. Initialize Preprocessing Ring Buffer
    g_buffer.begin();

    // 4. Initialize FoG Detection State Machine
    g_fog_detector.begin(
        MODEL_HYSTERESIS_HIGH,
        MODEL_HYSTERESIS_LOW,
        FOG_CONFIRM_CONSECUTIVE,
        FOG_COOLDOWN_MS
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

    // 7. Execute On-Device Self-Test using 5 Full 100x6 Golden Vectors
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
    Serial.println(F("Commands: 't' -> Run Self-Test | 'c' -> Re-calibrate Gyro | 's' -> Diagnostics"));
    Serial.println(F("----------------------------------------------------------"));
}

void loop() {
    // Core 1 Interactive Serial Monitor Commands & Diagnostics
    if (Serial.available()) {
        char cmd = (char)Serial.read();
        if (cmd == 't' || cmd == 'T') {
            runOnDeviceSelfTest();
        } else if (cmd == 'c' || cmd == 'C') {
            Serial.println(F("[CALIB] Re-calibrating gyro bias... Keep still!"));
            g_imu.calibrateGyro(200);
        } else if (cmd == 's' || cmd == 'S') {
            Serial.printf("[STATUS] Inferences: %u | FoG Events: %u | Last Latency: %.2f ms | Free Heap: %u bytes\n",
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
 */
void taskSensorAcquisition(void* pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(1000 / MODEL_SAMPLE_RATE_HZ); // 20 ms
    
    float sample[6];

    while (true) {
        // Read calibrated physical units (g and deg/s)
        if (g_imu.readSample(sample)) {
            // Protect ring buffer write with mutex
            if (xSemaphoreTake(g_buffer_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                g_buffer.pushSample(sample);
                xSemaphoreGive(g_buffer_mutex);
            }
        }
        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

/**
 * FreeRTOS Task on Core 1: Periodic 500 ms Inference & Haptic Control
 * Evaluates the 100x6 temporal window and actuates haptic motor on FoG alert.
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
            float raw_prob = g_engine.predict(window_input_quantized, &latency_us);
            g_last_inference_latency_us = latency_us;
            g_total_inferences++;

            // Process probability through dual-threshold hysteresis & cooldown
            FoGState state = g_fog_detector.update(raw_prob);
            float smooth_prob = g_fog_detector.getSmoothedProbability();

            if (state == FOG_STATE_ACTIVE_ALERT) {
                g_total_fog_events++;
                // Actuate non-blocking rhythmic cadence cueing
                g_haptic.triggerCadenceCue();
                Serial.printf(">>> [ALERT] FREEZING OF GAIT DETECTED! Prob: %.3f (Raw: %.3f) | Latency: %.2f ms <<<\n",
                              smooth_prob, raw_prob, latency_us / 1000.0f);
            } else if (state == FOG_STATE_COOLDOWN) {
                // FoG ongoing or cooling down
            } else {
                // Normal gait
                if (g_total_inferences % 10 == 0) {
                    Serial.printf("[GAIT] Normal | P(FoG): %.3f | Latency: %.2f ms | Free Heap: %u bytes\n",
                                  smooth_prob, latency_us / 1000.0f, esp_get_free_heap_size());
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

/**
 * Self-Test Routine: Validates TFLM engine against Full 100x6 Golden Vectors
 * Uses strict numerical error tolerance check: |prob - expected_prob| <= tolerance.
 */
void runOnDeviceSelfTest() {
    Serial.println(F("\n--- RUNNING ON-DEVICE GOLDEN VECTOR SELF-TEST (100x6 Windows) ---"));
    int passed = 0;
    for (int i = 0; i < 5; i++) {
        unsigned long t_us = 0;
        float p = g_engine.predict(g_golden_vectors[i].window_data, &t_us);
        float err = fabsf(p - g_golden_vectors[i].expected_prob);
        bool match = (err <= g_golden_vectors[i].tolerance);
        
        Serial.printf("  Vector %d [%s]: Expected P=%.3f, Infer P=%.3f (Err: %.4f, Tol: %.2f) Latency=%.2f ms -> %s\n",
                      i + 1, g_golden_vectors[i].description,
                      g_golden_vectors[i].expected_prob, p, err, g_golden_vectors[i].tolerance,
                      t_us / 1000.0f, match ? "PASS" : "FAIL");
        if (match) {
            passed++;
        }
    }
    Serial.printf("--- SELF-TEST RESULT: %d / 5 TEST VECTORS VERIFIED ---\n\n", passed);
}

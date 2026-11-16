#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_config.h"

#ifdef ESP32
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#endif

class ModelInferenceEngine {
public:
    ModelInferenceEngine();
    ~ModelInferenceEngine();
    
    // Initialize TFLM runtime, register ops, and dynamically allocate 16-byte aligned arena
    bool begin(const unsigned char* model_data, size_t model_data_len);
    
    // Run INT8 inference on a 100x6 window (600 int8 bytes)
    // Thread-safe: protected by FreeRTOS mutex across Core 0/Core 1 tasks & loop()
    float predict(const int8_t* window_input_600b, unsigned long* out_latency_us = nullptr);
    
    // Diagnostic queries
    bool isReady() const { return m_ready; }
    size_t getArenaSize() const { return m_arena_allocated_size; }
    size_t getArenaUsedBytes() const;
    int getInputElements() const { return MODEL_WINDOW_SAMPLES * MODEL_NUM_CHANNELS; }
    
private:
    uint8_t* m_tensor_arena_raw;
    uint8_t* m_tensor_arena_aligned;
    size_t m_arena_allocated_size;

    tflite::MicroInterpreter* m_interpreter;
    TfLiteTensor* m_input_tensor;
    TfLiteTensor* m_output_tensor;
    bool m_ready;
    
    float m_output_scale;
    int m_output_zero_point;

#ifdef ESP32
    SemaphoreHandle_t m_predict_mutex;
#endif
};

#endif // MODEL_INFERENCE_H

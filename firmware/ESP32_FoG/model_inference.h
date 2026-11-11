#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_config.h"

class ModelInferenceEngine {
public:
    ModelInferenceEngine();
    
    // Initialize TFLM runtime, register ops, and allocate tensor arena
    bool begin(const unsigned char* model_data, size_t model_data_len);
    
    // Run INT8 inference on a 100x6 window (600 int8 bytes)
    // Returns probability of Freezing of Gait P(FoG) in [0.0, 1.0]
    float predict(const int8_t* window_input_600b, unsigned long* out_latency_us = nullptr);
    
    // Diagnostic queries
    bool isReady() const { return m_ready; }
    size_t getArenaSize() const { return TENSOR_ARENA_SIZE_BYTES; }
    size_t getArenaUsedBytes() const;
    int getInputElements() const { return MODEL_WINDOW_SAMPLES * MODEL_NUM_CHANNELS; }
    
private:
    uint8_t m_tensor_arena[TENSOR_ARENA_SIZE_BYTES];
    tflite::MicroInterpreter* m_interpreter;
    TfLiteTensor* m_input_tensor;
    TfLiteTensor* m_output_tensor;
    bool m_ready;
    
    float m_output_scale;
    int m_output_zero_point;
};

#endif // MODEL_INFERENCE_H

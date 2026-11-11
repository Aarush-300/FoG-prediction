#include "model_inference.h"
#include <esp_timer.h>

ModelInferenceEngine::ModelInferenceEngine()
    : m_interpreter(nullptr),
      m_input_tensor(nullptr),
      m_output_tensor(nullptr),
      m_ready(false),
      m_output_scale(MODEL_OUTPUT_SCALE),
      m_output_zero_point(MODEL_OUTPUT_ZERO_POINT)
{
    memset(m_tensor_arena, 0, sizeof(m_tensor_arena));
}

bool ModelInferenceEngine::begin(const unsigned char* model_data, size_t model_data_len) {
    if (!model_data || model_data_len == 0) {
        Serial.println(F("[TFLM] Error: NULL or empty model data pointer!"));
        return false;
    }

    // 1. Map model schema
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TFLM] Error: Model schema version %d does not match supported version %d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // 2. Register all required TFLite Micro operators
    static tflite::MicroMutableOpResolver<26> op_resolver;
    op_resolver.AddConv2D();
    op_resolver.AddDepthwiseConv2D();
    op_resolver.AddFullyConnected();
    op_resolver.AddMaxPool2D();
    op_resolver.AddAveragePool2D();
    op_resolver.AddRelu();
    op_resolver.AddRelu6();
    op_resolver.AddReshape();
    op_resolver.AddAdd();
    op_resolver.AddSub();
    op_resolver.AddMul();
    op_resolver.AddTanh();
    op_resolver.AddLogistic();
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();
    op_resolver.AddPad();
    op_resolver.AddPack();
    op_resolver.AddUnpack();
    op_resolver.AddSplit();
    op_resolver.AddSplitV();
    op_resolver.AddStridedSlice();
    op_resolver.AddTranspose();
    op_resolver.AddExpandDims();
    op_resolver.AddFill();
    op_resolver.AddShape();
    op_resolver.AddMean();

    // 3. Instantiate MicroInterpreter
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, m_tensor_arena, sizeof(m_tensor_arena)
    );
    m_interpreter = &static_interpreter;

    // 4. Allocate memory for model tensors from tensor arena
    TfLiteStatus allocate_status = m_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println(F("[TFLM] Error: AllocateTensors() failed! Increase TENSOR_ARENA_SIZE_BYTES."));
        return false;
    }

    // 5. Query and validate input/output tensors
    m_input_tensor = m_interpreter->input(0);
    m_output_tensor = m_interpreter->output(0);

    if (!m_input_tensor || !m_output_tensor) {
        Serial.println(F("[TFLM] Error: Failed to resolve input/output tensors!"));
        return false;
    }

    // Validate tensor types
    if (m_input_tensor->type != kTfLiteInt8) {
        Serial.println(F("[TFLM] Warning: Input tensor is not INT8!"));
    }

    // Extract dynamic quantization parameters if available
    if (m_output_tensor->params.scale > 0.0f) {
        m_output_scale = m_output_tensor->params.scale;
        m_output_zero_point = m_output_tensor->params.zero_point;
    }

    m_ready = true;
    Serial.printf("[TFLM] Initialized successfully. Arena used: %u / %u bytes (%.1f%%)\n",
                  m_interpreter->arena_used_bytes(),
                  sizeof(m_tensor_arena),
                  (float)m_interpreter->arena_used_bytes() / sizeof(m_tensor_arena) * 100.0f);
    return true;
}

float ModelInferenceEngine::predict(const int8_t* window_input_600b, unsigned long* out_latency_us) {
    if (!m_ready || !m_interpreter || !window_input_600b) {
        return 0.0f;
    }

    // Copy 600 quantized INT8 bytes into TFLM input tensor
    memcpy(m_input_tensor->data.int8, window_input_600b, MODEL_WINDOW_SAMPLES * MODEL_NUM_CHANNELS);

    int64_t t_start = esp_timer_get_time();

    // Execute on-device neural network inference
    TfLiteStatus invoke_status = m_interpreter->Invoke();

    int64_t t_end = esp_timer_get_time();
    if (out_latency_us) {
        *out_latency_us = (unsigned long)(t_end - t_start);
    }

    if (invoke_status != kTfLiteOk) {
        Serial.println(F("[TFLM] Error: Invoke() failed!"));
        return 0.0f;
    }

    // Dequantize INT8 output to float probability P(FoG)
    int8_t raw_q_output = m_output_tensor->data.int8[0];
    float prob = (static_cast<float>(raw_q_output) - static_cast<float>(m_output_zero_point)) * m_output_scale;

    // Numerical clamping to [0.0, 1.0]
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

    return prob;
}

size_t ModelInferenceEngine::getArenaUsedBytes() const {
    if (m_interpreter) {
        return m_interpreter->arena_used_bytes();
    }
    return 0;
}

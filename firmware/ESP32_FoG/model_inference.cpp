#include "model_inference.h"
#include <esp_timer.h>

ModelInferenceEngine::ModelInferenceEngine()
    : m_tensor_arena_raw(nullptr),
      m_tensor_arena_aligned(nullptr),
      m_arena_allocated_size(0),
      m_interpreter(nullptr),
      m_input_tensor(nullptr),
      m_output_tensor(nullptr),
      m_ready(false),
      m_output_scale(MODEL_OUTPUT_SCALE),
      m_output_zero_point(MODEL_OUTPUT_ZERO_POINT)
{
#ifdef ESP32
    m_predict_mutex = xSemaphoreCreateMutex();
#endif
}

ModelInferenceEngine::~ModelInferenceEngine() {
#ifdef ESP32
    if (m_predict_mutex != nullptr) {
        vSemaphoreDelete(m_predict_mutex);
        m_predict_mutex = nullptr;
    }
    if (m_tensor_arena_raw != nullptr) {
        heap_caps_free(m_tensor_arena_raw);
        m_tensor_arena_raw = nullptr;
        m_tensor_arena_aligned = nullptr;
    }
#else
    if (m_tensor_arena_raw != nullptr) {
        free(m_tensor_arena_raw);
        m_tensor_arena_raw = nullptr;
        m_tensor_arena_aligned = nullptr;
    }
#endif
}

bool ModelInferenceEngine::begin(const unsigned char* model_data, size_t model_data_len) {
    if (!model_data || model_data_len == 0) {
        Serial.println(F("[TFLM] Error: NULL or empty model data pointer!"));
        return false;
    }

    // 1. Map model schema
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TFLM] Error: Model schema version %u does not match supported version %u\n",
                      (unsigned int)model->version(), (unsigned int)TFLITE_SCHEMA_VERSION);
        return false;
    }

    // 2. Register required TFLite Micro operators
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

    // 3. Dynamic Tensor Arena Allocation (Step 3.2: 16-byte aligned with fallbacks)
    static const size_t candidate_sizes[] = {
        220 * 1024,
        200 * 1024,
        180 * 1024,
        160 * 1024,
        140 * 1024,
        120 * 1024
    };
    const size_t arena_floor = TENSOR_ARENA_FLOOR_BYTES; // 100 KB floor

    m_tensor_arena_raw = nullptr;
    m_tensor_arena_aligned = nullptr;
    m_arena_allocated_size = 0;

    for (size_t c_size : candidate_sizes) {
        if (c_size < arena_floor) break;
#ifdef ESP32
        m_tensor_arena_raw = (uint8_t*)heap_caps_malloc(c_size + 16, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
#else
        m_tensor_arena_raw = (uint8_t*)malloc(c_size + 16);
#endif
        if (m_tensor_arena_raw != nullptr) {
            m_arena_allocated_size = c_size;
            // 16-byte alignment
            uintptr_t addr = (uintptr_t)m_tensor_arena_raw;
            m_tensor_arena_aligned = (uint8_t*)((addr + 15) & ~((uintptr_t)15));
            break;
        }
    }

    if (m_tensor_arena_aligned == nullptr) {
        Serial.println(F("[TFLM] Error: Failed to allocate internal SRAM for tensor arena!"));
        return false;
    }

    // 4. Instantiate MicroInterpreter on aligned arena
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, m_tensor_arena_aligned, m_arena_allocated_size
    );
    m_interpreter = &static_interpreter;

    // 5. Allocate tensors
    TfLiteStatus allocate_status = m_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println(F("[TFLM] Error: AllocateTensors() failed! Increase arena capacity."));
        return false;
    }

    // 6. Query and validate input/output tensors
    m_input_tensor = m_interpreter->input(0);
    m_output_tensor = m_interpreter->output(0);

    if (!m_input_tensor || !m_output_tensor) {
        Serial.println(F("[TFLM] Error: Failed to resolve input/output tensors!"));
        return false;
    }

    if (m_input_tensor->type != kTfLiteInt8) {
        Serial.println(F("[TFLM] Warning: Input tensor is not INT8!"));
    }

    if (m_output_tensor->params.scale > 0.0f) {
        m_output_scale = m_output_tensor->params.scale;
        m_output_zero_point = m_output_tensor->params.zero_point;
    }

    m_ready = true;
    Serial.printf("[TFLM] Boot Initialized. Arena capacity: %u bytes | Used: %u bytes (%.1f%%)\n",
                  (unsigned int)m_arena_allocated_size,
                  (unsigned int)m_interpreter->arena_used_bytes(),
                  (float)m_interpreter->arena_used_bytes() / m_arena_allocated_size * 100.0f);
    return true;
}

float ModelInferenceEngine::predict(const int8_t* window_input_600b, unsigned long* out_latency_us) {
    if (!m_ready || !m_interpreter || !window_input_600b) {
        return 0.0f;
    }

#ifdef ESP32
    if (m_predict_mutex != nullptr) {
        xSemaphoreTake(m_predict_mutex, portMAX_DELAY);
    }
#endif

    int64_t start_time = esp_timer_get_time();

    // Copy 600 bytes (100 samples * 6 channels) into INT8 input tensor
    memcpy(m_input_tensor->data.int8, window_input_600b, MODEL_WINDOW_SAMPLES * MODEL_NUM_CHANNELS);

    // Run inference
    TfLiteStatus invoke_status = m_interpreter->Invoke();
    int64_t end_time = esp_timer_get_time();

    if (out_latency_us) {
        *out_latency_us = (unsigned long)(end_time - start_time);
    }

    if (invoke_status != kTfLiteOk) {
#ifdef ESP32
        if (m_predict_mutex != nullptr) {
            xSemaphoreGive(m_predict_mutex);
        }
#endif
        return 0.0f;
    }

    // Dequantize INT8 output to float probability P(FoG) in [0.0, 1.0]
    int8_t raw_q_out = m_output_tensor->data.int8[0];
    float prob = (static_cast<float>(raw_q_out) - static_cast<float>(m_output_zero_point)) * m_output_scale;

    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

#ifdef ESP32
    if (m_predict_mutex != nullptr) {
        xSemaphoreGive(m_predict_mutex);
    }
#endif

    return prob;
}

size_t ModelInferenceEngine::getArenaUsedBytes() const {
    if (!m_interpreter) return 0;
    return m_interpreter->arena_used_bytes();
}

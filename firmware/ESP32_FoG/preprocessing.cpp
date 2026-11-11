#include "preprocessing.h"

Preprocessing::Preprocessing()
    : m_head_index(0),
      m_total_samples(0),
      m_stride_counter(0),
      m_window_ready(false)
{
    memset(m_ring_buffer, 0, sizeof(m_ring_buffer));
}

void Preprocessing::begin() {
    reset();
}

void Preprocessing::reset() {
    m_head_index = 0;
    m_total_samples = 0;
    m_stride_counter = 0;
    m_window_ready = false;
    memset(m_ring_buffer, 0, sizeof(m_ring_buffer));
}

inline int8_t Preprocessing::quantizeVal(float normalized_val, float scale, int zero_point) {
    int32_t q = static_cast<int32_t>(roundf(normalized_val / scale)) + zero_point;
    if (q < -128) return -128;
    if (q > 127)  return 127;
    return static_cast<int8_t>(q);
}

bool Preprocessing::pushSample(const float* sample6) {
    if (!sample6) return false;

    // 1. Z-Score Normalization and INT8 Quantization per channel
    for (int c = 0; c < MODEL_NUM_CHANNELS; c++) {
        float norm_val = (sample6[c] - IMU_MEAN[c]) / IMU_STD[c];
        m_ring_buffer[m_head_index][c] = quantizeVal(norm_val, MODEL_INPUT_SCALE, MODEL_INPUT_ZERO_POINT);
    }

    // 2. Advance circular buffer head
    m_head_index = (m_head_index + 1) % MODEL_WINDOW_SAMPLES;
    m_total_samples++;
    m_stride_counter++;

    // 3. Check if ready for inference
    if (m_total_samples >= MODEL_WINDOW_SAMPLES) {
        if (m_stride_counter >= MODEL_STRIDE_SAMPLES) {
            m_window_ready = true;
            return true;
        }
    }
    return false;
}

void Preprocessing::getQuantizedWindow(int8_t* dest_buffer) {
    if (!dest_buffer) return;

    // Unroll circular buffer into strictly chronological order [Oldest -> Newest]
    // The oldest sample is at m_head_index
    int read_idx = m_head_index;
    for (int i = 0; i < MODEL_WINDOW_SAMPLES; i++) {
        for (int c = 0; c < MODEL_NUM_CHANNELS; c++) {
            dest_buffer[i * MODEL_NUM_CHANNELS + c] = m_ring_buffer[read_idx][c];
        }
        read_idx = (read_idx + 1) % MODEL_WINDOW_SAMPLES;
    }
}

void Preprocessing::consumeStride() {
    m_stride_counter = 0;
    m_window_ready = false;
}

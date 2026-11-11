#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <Arduino.h>
#include "model_config.h"

/**
 * @brief Circular Ring Buffer and Quantized Feature Pipeline for ESP32 FoG Detection.
 * 
 * Ingests 50 Hz 6-axis IMU samples [ax, ay, az, gx, gy, gz] in physical units (g, deg/s),
 * applies training-set Z-score standardization, quantizes to INT8,
 * maintains a 100-sample temporal window, and triggers inference every 25 samples (0.5 s).
 */
class Preprocessing {
public:
    Preprocessing();
    
    // Initialize buffer and counters
    void begin();
    
    // Reset buffer and sample counters
    void reset();
    
    // Ingest a 6-channel sample: [ax, ay, az, gx, gy, gz]
    // Returns true if an inference window (25-sample stride) is ready
    bool pushSample(const float* sample6);
    
    // Check if at least 100 samples have been collected
    bool isWindowReady() const { return m_window_ready; }
    
    // Copy chronological 100x6 window (600 bytes) into destination buffer
    void getQuantizedWindow(int8_t* dest_buffer);
    
    // Consume stride (resets stride counter for next 25 samples)
    void consumeStride();
    
    // Diagnostic stats
    int getTotalSamples() const { return m_total_samples; }
    
private:
    inline int8_t quantizeVal(float normalized_val, float scale, int zero_point);
    
    int8_t m_ring_buffer[MODEL_WINDOW_SAMPLES][MODEL_NUM_CHANNELS];
    int m_head_index;
    int m_total_samples;
    int m_stride_counter;
    bool m_window_ready;
};

#endif // PREPROCESSING_H

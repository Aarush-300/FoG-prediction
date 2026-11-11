#ifndef FOG_DETECTION_H
#define FOG_DETECTION_H

#include <Arduino.h>
#include "model_config.h"

/**
 * @brief Clinical State Machine for Freezing of Gait Detection.
 * 
 * Implements: NORMAL -> WARNING -> ACTIVE_ALERT -> COOLDOWN -> NORMAL.
 * Uses Exponential Moving Average (EMA) smoothing, dual-threshold hysteresis,
 * consecutive inference confirmations, refractory cooldown, and episode counting.
 */
enum FoGState {
    FOG_STATE_NORMAL = 0,
    FOG_STATE_WARNING = 1,
    FOG_STATE_ACTIVE_ALERT = 2,
    FOG_STATE_COOLDOWN = 3
};

class FoGDetectionEngine {
public:
    FoGDetectionEngine(float high_threshold = MODEL_HYSTERESIS_HIGH,
                       float low_threshold = MODEL_HYSTERESIS_LOW,
                       int consecutive_confirmations = FOG_CONFIRM_CONSECUTIVE,
                       unsigned long cooldown_ms = FOG_COOLDOWN_MS,
                       float ema_alpha = EMA_SMOOTHING_ALPHA);
                       
    // Initialize or re-configure state machine parameters
    bool begin(float high_threshold = MODEL_HYSTERESIS_HIGH,
               float low_threshold = MODEL_HYSTERESIS_LOW,
               int consecutive_confirmations = FOG_CONFIRM_CONSECUTIVE,
               unsigned long cooldown_ms = FOG_COOLDOWN_MS);
               
    // Update state machine with latest inference probability [0.0, 1.0]
    // Returns current state. ACTIVE_ALERT indicates that a haptic cue should fire.
    FoGState update(float raw_probability);
    
    // Status queries
    FoGState getState() const { return m_state; }
    const char* getStateString() const;
    float getSmoothedProbability() const { return m_smoothed_prob; }
    float getRawProbability() const { return m_last_raw_prob; }
    int getEpisodeCount() const { return m_episode_count; }
    
    // Manual reset
    void reset();
    
private:
    float m_high_threshold;
    float m_low_threshold;
    int m_consecutive_needed;
    unsigned long m_cooldown_ms;
    float m_ema_alpha;
    
    FoGState m_state;
    float m_smoothed_prob;
    float m_last_raw_prob;
    int m_consecutive_count;
    int m_episode_count;
    unsigned long m_last_alert_time;
};

#endif // FOG_DETECTION_H

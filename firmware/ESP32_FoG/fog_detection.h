#ifndef FOG_DETECTION_H
#define FOG_DETECTION_H

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <stdint.h>
#include <stdbool.h>
extern unsigned long millis();
#endif

#include "model_config.h"

/**
 * @brief Clinical Finite State Machine for Freezing of Gait Detection.
 * 
 * Implements: NORMAL (0) -> WARNING (1) -> ACTIVE_ALERT (2) -> COOLDOWN (3) -> NORMAL (0).
 * Hardened with:
 * - Truly consecutive confirmations (streak resets to 0 if smoothed_prob < high).
 * - Pre-registered re-arm policies (timeout vs timeout_and_low).
 * - Deterministic timestamp ingestion for unit testing / verification.
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
                       float ema_alpha = EMA_SMOOTHING_ALPHA,
                       int rearm_policy = FSM_REARM_POLICY);
                       
    bool begin(float high_threshold = MODEL_HYSTERESIS_HIGH,
               float low_threshold = MODEL_HYSTERESIS_LOW,
               int consecutive_confirmations = FOG_CONFIRM_CONSECUTIVE,
               unsigned long cooldown_ms = FOG_COOLDOWN_MS,
               int rearm_policy = FSM_REARM_POLICY);
               
    // Update state machine with latest inference probability [0.0, 1.0]
    // If now_ms is 0, calls millis() internally.
    FoGState update(float raw_probability, unsigned long now_ms = 0);
    
    // Status queries
    FoGState getState() const { return m_state; }
    const char* getStateString() const;
    float getSmoothedProbability() const { return m_smoothed_prob; }
    float getRawProbability() const { return m_last_raw_prob; }
    int getEpisodeCount() const { return m_episode_count; }
    int getConsecutiveCount() const { return m_consecutive_count; }
    
    // Manual reset
    void reset();
    
private:
    float m_high_threshold;
    float m_low_threshold;
    int m_consecutive_needed;
    unsigned long m_cooldown_ms;
    float m_ema_alpha;
    int m_rearm_policy;
    
    FoGState m_state;
    float m_smoothed_prob;
    float m_last_raw_prob;
    int m_consecutive_count;
    int m_episode_count;
    unsigned long m_last_alert_time;
};

#endif // FOG_DETECTION_H

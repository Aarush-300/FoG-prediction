#include "fog_detection.h"

FoGDetectionEngine::FoGDetectionEngine(float high_threshold,
                                       float low_threshold,
                                       int consecutive_confirmations,
                                       unsigned long cooldown_ms,
                                       float ema_alpha)
    : m_high_threshold(high_threshold),
      m_low_threshold(low_threshold),
      m_consecutive_needed(consecutive_confirmations),
      m_cooldown_ms(cooldown_ms),
      m_ema_alpha(ema_alpha),
      m_state(FOG_STATE_NORMAL),
      m_smoothed_prob(0.0f),
      m_last_raw_prob(0.0f),
      m_consecutive_count(0),
      m_episode_count(0),
      m_last_alert_time(0)
{
}

bool FoGDetectionEngine::begin(float high_threshold,
                              float low_threshold,
                              int consecutive_confirmations,
                              unsigned long cooldown_ms)
{
    m_high_threshold = high_threshold;
    m_low_threshold = low_threshold;
    m_consecutive_needed = consecutive_confirmations;
    m_cooldown_ms = cooldown_ms;
    reset();
    return true;
}

void FoGDetectionEngine::reset() {
    m_state = FOG_STATE_NORMAL;
    m_smoothed_prob = 0.0f;
    m_last_raw_prob = 0.0f;
    m_consecutive_count = 0;
    m_last_alert_time = 0;
}

FoGState FoGDetectionEngine::update(float raw_probability) {
    m_last_raw_prob = raw_probability;

    // 1. Exponential Moving Average (EMA) smoothing
    m_smoothed_prob = (m_ema_alpha * raw_probability) + ((1.0f - m_ema_alpha) * m_smoothed_prob);

    unsigned long now = millis();

    // 2. State Machine Transitions
    switch (m_state) {
        case FOG_STATE_NORMAL:
            if (m_smoothed_prob >= m_high_threshold) {
                m_consecutive_count++;
                if (m_consecutive_count >= m_consecutive_needed) {
                    m_state = FOG_STATE_ACTIVE_ALERT;
                    m_episode_count++;
                    m_last_alert_time = now;
                } else {
                    m_state = FOG_STATE_WARNING;
                }
            } else {
                m_consecutive_count = 0;
            }
            break;

        case FOG_STATE_WARNING:
            if (m_smoothed_prob >= m_high_threshold) {
                m_consecutive_count++;
                if (m_consecutive_count >= m_consecutive_needed) {
                    m_state = FOG_STATE_ACTIVE_ALERT;
                    m_episode_count++;
                    m_last_alert_time = now;
                }
            } else if (m_smoothed_prob < m_low_threshold) {
                m_consecutive_count = 0;
                m_state = FOG_STATE_NORMAL;
            }
            break;

        case FOG_STATE_ACTIVE_ALERT:
            // Alert has been triggered; immediately transition to cooldown refractory period
            // to avoid repeated jarring pulses while the patient is actively breaking the freeze
            m_state = FOG_STATE_COOLDOWN;
            break;

        case FOG_STATE_COOLDOWN:
            // Remain in cooldown until both:
            // 1. The refractory period has elapsed (e.g. 3.0 seconds)
            // 2. The smoothed probability drops below the low reset threshold
            if ((now - m_last_alert_time >= m_cooldown_ms) && (m_smoothed_prob < m_low_threshold)) {
                m_consecutive_count = 0;
                m_state = FOG_STATE_NORMAL;
            }
            break;
    }

    return m_state;
}

const char* FoGDetectionEngine::getStateString() const {
    switch (m_state) {
        case FOG_STATE_NORMAL:       return "NORMAL";
        case FOG_STATE_WARNING:      return "WARNING";
        case FOG_STATE_ACTIVE_ALERT: return "ACTIVE_ALERT";
        case FOG_STATE_COOLDOWN:     return "COOLDOWN";
        default:                     return "UNKNOWN";
    }
}

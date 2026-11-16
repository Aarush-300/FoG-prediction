#include "fog_detection.h"

FoGDetectionEngine::FoGDetectionEngine(float high_threshold,
                                       float low_threshold,
                                       int consecutive_confirmations,
                                       unsigned long cooldown_ms,
                                       float ema_alpha,
                                       int rearm_policy)
    : m_high_threshold(high_threshold),
      m_low_threshold(low_threshold),
      m_consecutive_needed(consecutive_confirmations),
      m_cooldown_ms(cooldown_ms),
      m_ema_alpha(ema_alpha),
      m_rearm_policy(rearm_policy),
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
                              unsigned long cooldown_ms,
                              int rearm_policy)
{
    m_high_threshold = high_threshold;
    m_low_threshold = low_threshold;
    m_consecutive_needed = consecutive_confirmations;
    m_cooldown_ms = cooldown_ms;
    m_rearm_policy = rearm_policy;
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

FoGState FoGDetectionEngine::update(float raw_probability, unsigned long now_ms) {
    m_last_raw_prob = raw_probability;

    // 1. Exponential Moving Average (EMA) smoothing
    m_smoothed_prob = (m_ema_alpha * raw_probability) + ((1.0f - m_ema_alpha) * m_smoothed_prob);

    unsigned long now = (now_ms > 0) ? now_ms : millis();

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
            } else {
                // Confirmations must be strictly consecutive:
                // If probability drops below high, the streak is broken!
                m_consecutive_count = 0;
                if (m_smoothed_prob < m_low_threshold) {
                    m_state = FOG_STATE_NORMAL;
                }
            }
            break;

        case FOG_STATE_ACTIVE_ALERT:
            // Alert has fired; transition to cooldown refractory period immediately
            m_state = FOG_STATE_COOLDOWN;
            break;

        case FOG_STATE_COOLDOWN: {
            bool rearm = false;
            unsigned long elapsed = now - m_last_alert_time;
            if (m_rearm_policy == REARM_POLICY_TIMEOUT) {
                rearm = (elapsed >= m_cooldown_ms);
            } else { // REARM_POLICY_TIMEOUT_AND_LOW
                rearm = (elapsed >= m_cooldown_ms) && (m_smoothed_prob < m_low_threshold);
            }

            if (rearm) {
                m_consecutive_count = 0;
                m_state = FOG_STATE_NORMAL;
            }
            break;
        }
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

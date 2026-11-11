#include "haptic_feedback.h"

HapticFeedback::HapticFeedback(int motor_pin, int pwm_channel, int default_intensity)
    : m_motor_pin(motor_pin),
      m_pwm_channel(pwm_channel),
      m_intensity(default_intensity),
      m_active(false),
      m_is_vibrating(false),
      m_target_pulses(0),
      m_current_pulse(0),
      m_pulse_ms(HAPTIC_PULSE_DURATION_MS),
      m_pause_ms(HAPTIC_PULSE_PAUSE_MS),
      m_state_timer(0)
{
}

bool HapticFeedback::begin() {
    pinMode(m_motor_pin, OUTPUT);
    digitalWrite(m_motor_pin, LOW);

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
    bool ok = ledcAttach(m_motor_pin, HAPTIC_PWM_FREQ_HZ, HAPTIC_PWM_RESOLUTION_BITS);
    if (!ok) {
        Serial.println(F("[HAPTIC] Error: ledcAttach failed on GPIO!"));
        return false;
    }
#else
    ledcSetup(m_pwm_channel, HAPTIC_PWM_FREQ_HZ, HAPTIC_PWM_RESOLUTION_BITS);
    ledcAttachPin(m_motor_pin, m_pwm_channel);
#endif

    setMotorPower(0);
    Serial.printf("[HAPTIC] Initialized on GPIO %d (Channel %d, Freq %d Hz)\n",
                  m_motor_pin, m_pwm_channel, HAPTIC_PWM_FREQ_HZ);
    return true;
}

void HapticFeedback::setMotorPower(uint8_t duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
    ledcWrite(m_motor_pin, duty);
#else
    ledcWrite(m_pwm_channel, duty);
#endif
}

void HapticFeedback::triggerCadenceCue(int num_pulses, int pulse_ms, int pause_ms) {
    m_target_pulses = num_pulses;
    m_pulse_ms = pulse_ms;
    m_pause_ms = pause_ms;
    m_current_pulse = 0;
    m_active = true;
    m_is_vibrating = true;
    m_state_timer = millis();
    setMotorPower(m_intensity);
}

void HapticFeedback::triggerEmergencyAlert() {
    // SOS rhythm: 5 distinct fast bursts
    triggerCadenceCue(5, 80, 80);
}

void HapticFeedback::stop() {
    m_active = false;
    m_is_vibrating = false;
    m_current_pulse = 0;
    setMotorPower(0);
}

void HapticFeedback::update() {
    if (!m_active) return;

    unsigned long now = millis();

    if (m_is_vibrating) {
        if (now - m_state_timer >= (unsigned long)m_pulse_ms) {
            setMotorPower(0);
            m_is_vibrating = false;
            m_state_timer = now;
            m_current_pulse++;

            if (m_current_pulse >= m_target_pulses) {
                m_active = false;
            }
        }
    } else {
        if (now - m_state_timer >= (unsigned long)m_pause_ms) {
            setMotorPower(m_intensity);
            m_is_vibrating = true;
            m_state_timer = now;
        }
    }
}

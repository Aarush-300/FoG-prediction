#ifndef HAPTIC_FEEDBACK_H
#define HAPTIC_FEEDBACK_H

#include <Arduino.h>
#include "model_config.h"

class HapticFeedback {
public:
    HapticFeedback(int motor_pin = HAPTIC_MOTOR_PIN,
                   int pwm_channel = HAPTIC_PWM_CHANNEL,
                   int default_intensity = HAPTIC_DEFAULT_DUTY_CYCLE);

    bool begin();
    
    // Trigger rhythmic cadence cueing: 8 pulses, 120 ms on, 420 ms off
    void triggerCadenceCue(int num_pulses = HAPTIC_PULSES_PER_CUE,
                           int pulse_ms = HAPTIC_PULSE_DURATION_MS,
                           int pause_ms = HAPTIC_PULSE_PAUSE_MS);

    // Fast warning SOS pattern for critical system errors
    void triggerEmergencyAlert();

    // Immediately halt vibration motor
    void stop();

    // Periodic state machine update (call from Core 1 task loop)
    void update();

    // Query status
    bool isActive() const { return m_active; }
    bool isVibrating() const { return m_is_vibrating; }
    
    // Blanking query: true while motor is actively running AND for HAPTIC_BLANKING_MS (1000 ms) afterwards
    bool isBlankingActive() const;

private:
    void setMotorPower(uint8_t duty);

    int m_motor_pin;
    int m_pwm_channel;
    int m_intensity;

    bool m_active;
    bool m_is_vibrating;
    int m_target_pulses;
    int m_current_pulse;
    int m_pulse_ms;
    int m_pause_ms;
    unsigned long m_state_timer;
    unsigned long m_vibration_end_time;
};

#endif // HAPTIC_FEEDBACK_H

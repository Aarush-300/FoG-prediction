#ifndef HAPTIC_FEEDBACK_H
#define HAPTIC_FEEDBACK_H

#include <Arduino.h>
#include "model_config.h"

/**
 * @brief Non-blocking Haptic Tactile Cueing Driver for ESP32.
 * 
 * Drives an eccentric rotating mass (ERM) or linear resonant actuator (LRA)
 * vibration motor via an N-channel MOSFET on GPIO 13.
 * Supports ESP32 Arduino Core 2.x and 3.x LEDC PWM architectures.
 */
class HapticFeedback {
public:
    HapticFeedback(int motor_pin = HAPTIC_MOTOR_PIN,
                   int pwm_channel = HAPTIC_PWM_CHANNEL,
                   int default_intensity = HAPTIC_DEFAULT_DUTY_CYCLE);
                   
    // Initialize GPIO and LEDC PWM peripheral
    bool begin();
    
    // Trigger rhythmic sensory cueing pattern (non-blocking)
    void triggerCadenceCue(int num_pulses = HAPTIC_PULSES_PER_CUE,
                           int pulse_ms = HAPTIC_PULSE_DURATION_MS,
                           int pause_ms = HAPTIC_PULSE_PAUSE_MS);
                           
    // Trigger high-urgency SOS emergency alert (e.g. on sensor disconnect)
    void triggerEmergencyAlert();
    
    // Advance non-blocking pulse state machine
    void update();
    
    // Immediate stop
    void stop();
    
    // Intensity configuration (0 to 255)
    void setIntensity(uint8_t duty_cycle) { m_intensity = duty_cycle; }
    bool isVibrating() const { return m_is_vibrating; }
    
private:
    void setMotorPower(uint8_t duty);
    
    int m_motor_pin;
    int m_pwm_channel;
    uint8_t m_intensity;
    
    bool m_active;
    bool m_is_vibrating;
    int m_target_pulses;
    int m_current_pulse;
    int m_pulse_ms;
    int m_pause_ms;
    unsigned long m_state_timer;
};

#endif // HAPTIC_FEEDBACK_H

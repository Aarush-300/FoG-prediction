#include <iostream>
#include <vector>
#include <iomanip>
#include "fog_detection.h"

// Shim millis() for host runner
static unsigned long g_simulated_millis = 0;
unsigned long millis() {
    return g_simulated_millis;
}

int main(int argc, char** argv) {
    // Config: high=0.55, low=0.35, confirm=2, cooldown_ms=6000, alpha=0.60, rearm=REARM_POLICY_TIMEOUT_AND_LOW
    float high = 0.55f;
    float low = 0.35f;
    int confirm = 2;
    unsigned long cooldown_ms = 6000;
    float alpha = 0.60f;
    int rearm = REARM_POLICY_TIMEOUT_AND_LOW;

    if (argc >= 7) {
        high = std::stof(argv[1]);
        low = std::stof(argv[2]);
        confirm = std::stoi(argv[3]);
        cooldown_ms = std::stoul(argv[4]);
        alpha = std::stof(argv[5]);
        rearm = std::stoi(argv[6]);
    }

    FoGDetectionEngine fsm(high, low, confirm, cooldown_ms, alpha, rearm);

    // Reads line by line from stdin: prob now_ms
    float prob;
    unsigned long now_ms;
    while (std::cin >> prob >> now_ms) {
        g_simulated_millis = now_ms;
        FoGState state = fsm.update(prob, now_ms);
        std::cout << (int)state << " " 
                  << std::fixed << std::setprecision(6) << fsm.getSmoothedProbability() << " " 
                  << fsm.getConsecutiveCount() << "\n";
    }

    return 0;
}

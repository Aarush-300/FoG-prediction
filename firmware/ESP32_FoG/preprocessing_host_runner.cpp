#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include "preprocessing.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_raw_float_bin> <output_quant_int8_bin>\n";
        return 1;
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open input: " << argv[1] << "\n";
        return 1;
    }

    std::ofstream out(argv[2], std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output: " << argv[2] << "\n";
        return 1;
    }

    // Read total number of windows
    uint32_t num_windows = 0;
    in.read(reinterpret_cast<char*>(&num_windows), sizeof(num_windows));

    std::vector<float> window_buf(MODEL_WINDOW_SAMPLES * MODEL_NUM_CHANNELS);
    std::vector<int8_t> quant_out(MODEL_WINDOW_SAMPLES * MODEL_NUM_CHANNELS);

    Preprocessing prep;

    for (uint32_t w = 0; w < num_windows; w++) {
        in.read(reinterpret_cast<char*>(window_buf.data()), window_buf.size() * sizeof(float));
        
        prep.reset();
        for (int i = 0; i < MODEL_WINDOW_SAMPLES; i++) {
            const float* sample6 = &window_buf[i * MODEL_NUM_CHANNELS];
            prep.pushSample(sample6);
        }

        prep.getQuantizedWindow(quant_out.data());
        out.write(reinterpret_cast<const char*>(quant_out.data()), quant_out.size() * sizeof(int8_t));
    }

    return 0;
}

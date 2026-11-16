#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_bin> <output_bin>\n";
        return 1;
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in.is_open()) return 1;

    std::ofstream out(argv[2], std::ios::binary);
    if (!out.is_open()) return 1;

    // Read 3x3 rotation matrix (9 floats)
    float R[3][3];
    in.read(reinterpret_cast<char*>(R), 9 * sizeof(float));

    // Read number of test 3D vectors
    uint32_t num_vectors = 0;
    in.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));

    for (uint32_t n = 0; n < num_vectors; n++) {
        float v_in[3];
        float v_out[3];
        in.read(reinterpret_cast<char*>(v_in), 3 * sizeof(float));

        // Exact matrix multiplication from imu_sensor.cpp
        for (int i = 0; i < 3; i++) {
            v_out[i] = R[i][0] * v_in[0] + R[i][1] * v_in[1] + R[i][2] * v_in[2];
        }

        out.write(reinterpret_cast<const char*>(v_out), 3 * sizeof(float));
    }

    return 0;
}

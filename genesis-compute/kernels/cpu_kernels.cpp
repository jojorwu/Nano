#include <iostream>
#include <vector>
#include <cstdint>

extern "C" {
    // Structure equivalent to NeuronsSoA pointers for Zero-Copy FFI
    struct NeuronsFFI {
        int32_t* potential;
        int32_t* proximal_potential;
        int32_t* distal_potential;
        int32_t* apical_potential;
        int32_t* basal_potential;
        int32_t* threshold;
        int32_t* decay;
        int32_t* dendritic_gate;
        uint32_t len;
    };

    void cpp_propagate_spikes(
        const uint32_t* source_indices,
        const uint32_t* target_indices,
        const int32_t* weights,
        const uint8_t* compartments,
        uint32_t synapse_count,
        const bool* previous_spikes,
        NeuronsFFI neurons
    ) {
        for (uint32_t i = 0; i < synapse_count; ++i) {
            if (previous_spikes[source_indices[i]]) {
                uint32_t target = target_indices[i];
                int32_t weight = weights[i];
                uint8_t comp = compartments[i];

                switch (comp) {
                    case 0: neurons.proximal_potential[target] += weight; break;
                    case 1: neurons.distal_potential[target] += weight; break;
                    case 2: neurons.apical_potential[target] += weight; break;
                    case 3: neurons.basal_potential[target] += weight; break;
                }
            }
        }
    }
}

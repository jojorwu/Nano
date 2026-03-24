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

    // Integer sigmoid approximation
    inline int32_t sigmoid_gate_approx(int32_t input, int32_t theta) {
        int32_t diff = input - theta;
        if (diff > 512) return 1024;
        int32_t val = diff + 512;
        if (val < 64) return 64;
        return val;
    }

    void cpp_update_neurons(
        NeuronsFFI neurons,
        uint32_t current_tick,
        bool* new_spikes,
        int32_t ip_inc,
        int32_t ip_dec,
        int32_t noise_amp
    ) {
        // Use a single loop with minimal branching to encourage SIMD auto-vectorization
        #pragma omp parallel for
        for (uint32_t i = 0; i < neurons.len; ++i) {
            int32_t prox = neurons.proximal_potential[i];
            int32_t dist = neurons.distal_potential[i];

            int32_t dist_gain = sigmoid_gate_approx(prox, 512);
            int32_t dist_gated = (int64_t(dist) * dist_gain) >> 10;

            int32_t current_pot = neurons.potential[i] + prox + dist_gated;
            int32_t decayed_pot = (int64_t(current_pot) * (1024 - neurons.decay[i])) >> 10;

            bool fired = decayed_pot >= neurons.threshold[i];
            new_spikes[i] = fired;

            // Conditional updates without explicit branching where possible
            neurons.potential[i] = fired ? 0 : decayed_pot;

            int32_t target_thresh = fired ? (neurons.threshold[i] + ip_inc) :
                                   (neurons.threshold[i] > 1024 ? neurons.threshold[i] - ip_dec : neurons.threshold[i]);
            neurons.threshold[i] = target_thresh;

            // Clear compartments
            neurons.proximal_potential[i] = 0;
            neurons.distal_potential[i] = 0;
            neurons.apical_potential[i] = 0;
            neurons.basal_potential[i] = 0;
        }
    }
}

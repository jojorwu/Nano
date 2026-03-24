#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

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
        int32_t* gate_threshold;
        int32_t* adaptation_current;
        int32_t* refractory_timer;
        uint32_t* last_spike_tick;
        int32_t* backprop_signal;
        int32_t* activity_ema;
        int32_t* base_threshold;
        int32_t* liquid_current;
        uint8_t* is_excitatory;
        uint32_t len;
    };

    struct SynapsesFFI {
        const uint32_t* source_index;
        const uint32_t* target_index;
        int32_t* weight;
        const uint8_t* delay;
        int32_t* stp_resources;
        int32_t* stp_calcium;
        const uint8_t* compartment;
        uint32_t len;
    };

    // Physics Helpers (must match physics.rs exactly)
    inline int32_t sigmoid_gate_approx(int32_t input, int32_t theta) {
        int32_t diff = input - theta;
        if (diff > 512) return 1024;
        int32_t val = diff + 512;
        if (val < 64) return 64;
        return val;
    }

    inline int32_t calculate_dynamic_decay(int32_t base_decay, int32_t proximal, int32_t distal) {
        int32_t liquid_mod = ((std::abs(proximal) + std::abs(distal)) * 10) >> 10;
        int32_t res = base_decay - liquid_mod;
        return res > 1 ? res : 1;
    }

    inline int32_t saturating_add(int32_t a, int32_t b) {
        int64_t res = (int64_t)a + b;
        if (res > 2147483647) return 2147483647;
        if (res < -2147483648) return -2147483648;
        return (int32_t)res;
    }

    inline int32_t saturating_sub(int32_t a, int32_t b) {
        int64_t res = (int64_t)a - b;
        if (res > 2147483647) return 2147483647;
        if (res < -2147483648) return -2147483648;
        return (int32_t)res;
    }

    // Fast Xorshift PRNG for parity with Rust
    static uint32_t xorshift_seed = 0xDEADBEEF;
    inline int32_t get_noise(int32_t noise_amp) {
        if (noise_amp <= 0) return 0;
        xorshift_seed ^= xorshift_seed << 13;
        xorshift_seed ^= xorshift_seed >> 17;
        xorshift_seed ^= xorshift_seed << 5;
        return (int32_t)(xorshift_seed % (noise_amp * 2)) - noise_amp;
    }

    void cpp_propagate_spikes(
        SynapsesFFI synapses,
        const bool* previous_spikes,
        NeuronsFFI neurons
    ) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < synapses.len; ++i) {
            uint32_t src = synapses.source_index[i];
            if (previous_spikes[src]) {
                uint32_t target = synapses.target_index[i];
                int32_t gate = neurons.dendritic_gate[target];
                if (gate < 8) continue; // Match Rust CpuBackend check

                int32_t weight = synapses.weight[i];
                uint8_t comp = synapses.compartment[i];

                // STP Implementation
                int32_t u_facilitation = synapses.stp_calcium[i];
                int32_t r_depression = synapses.stp_resources[i];

                int32_t stp_weight = (int64_t(weight) * r_depression) >> 10;
                stp_weight = (int64_t(stp_weight) * (1024 + u_facilitation)) >> 10;

                // Consumption
                synapses.stp_resources[i] = (int64_t(synapses.stp_resources[i]) * 800) >> 10;
                int32_t new_calcium = synapses.stp_calcium[i] + 200;
                synapses.stp_calcium[i] = (new_calcium > 1024) ? 1024 : new_calcium;

                int32_t gated_weight = (int64_t(stp_weight) * gate) >> 10;

                switch (comp) {
                    case 0:
                        #pragma omp atomic update
                        neurons.proximal_potential[target] += gated_weight;
                        break;
                    case 1:
                        #pragma omp atomic update
                        neurons.distal_potential[target] += gated_weight;
                        break;
                    case 2:
                        #pragma omp atomic update
                        neurons.apical_potential[target] += gated_weight;
                        break;
                    case 3:
                        #pragma omp atomic update
                        neurons.basal_potential[target] += gated_weight;
                        break;
                }
            }
        }
    }

    void cpp_recover_stp(SynapsesFFI synapses) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < synapses.len; ++i) {
            synapses.stp_resources[i] = (int64_t(synapses.stp_resources[i]) * 99 + 1024) / 100;
            synapses.stp_calcium[i] = (int64_t(synapses.stp_calcium[i]) * 95) / 100;
        }
    }

    void cpp_update_neurons(
        NeuronsFFI neurons,
        uint32_t current_tick,
        uint8_t* new_spikes,
        int32_t ip_inc,
        int32_t ip_dec,
        int32_t noise_amp
    ) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < neurons.len; ++i) {
            int32_t pot = neurons.potential[i];
            int32_t prox = neurons.proximal_potential[i];
            int32_t dist = neurons.distal_potential[i];
            int32_t apical = neurons.apical_potential[i];
            int32_t basal = neurons.basal_potential[i];
            int32_t g_thresh = neurons.gate_threshold[i];
            int32_t liquid = neurons.liquid_current[i];
            int32_t adaptation = neurons.adaptation_current[i];
            int32_t refr = neurons.refractory_timer[i];

            int32_t dist_gain = sigmoid_gate_approx(prox, g_thresh);
            int32_t dist_gated = (int64_t(dist) * dist_gain) >> 10;

            int32_t apical_gain = sigmoid_gate_approx(dist_gated, g_thresh);
            int32_t apical_gated = (int64_t(apical) * apical_gain) >> 10;

            int32_t mod_factor = (basal < 0) ? 800 : ((basal > 512) ? 1200 : 1024);
            int32_t noise = get_noise(noise_amp);

            int32_t current_pot = pot;
            current_pot = saturating_add(current_pot, prox);
            current_pot = saturating_add(current_pot, dist_gated);
            current_pot = saturating_add(current_pot, apical_gated);
            current_pot = saturating_add(current_pot, liquid);
            current_pot = saturating_add(current_pot, noise);
            current_pot = saturating_sub(current_pot, adaptation);

            current_pot = (int64_t(current_pot) * mod_factor) >> 10;

            int32_t d_val = calculate_dynamic_decay(neurons.decay[i], prox, dist);
            int32_t final_pot = (int64_t(current_pot) * (1024 - d_val)) >> 10;

            int32_t refr_mult = (refr > 0) ? (1 + (1 << refr)) : 1;
            int64_t effective_threshold = (int64_t)neurons.threshold[i] * refr_mult;

            bool fired = final_pot >= effective_threshold;
            new_spikes[i] = fired ? 1 : 0;

            if (fired) {
                neurons.potential[i] = 0;
                neurons.threshold[i] = saturating_add(neurons.threshold[i], ip_inc);
                neurons.adaptation_current[i] = saturating_add(neurons.adaptation_current[i], 100);
                neurons.refractory_timer[i] = 4;
                neurons.last_spike_tick[i] = current_tick;
                neurons.backprop_signal[i] = 1024;
            } else {
                neurons.potential[i] = final_pot;
                if (neurons.threshold[i] > neurons.base_threshold[i]) {
                    neurons.threshold[i] = saturating_sub(neurons.threshold[i], ip_dec);
                }
                neurons.adaptation_current[i] = (int64_t(neurons.adaptation_current[i]) * 95) / 100;
                if (refr > 0) neurons.refractory_timer[i]--;
                neurons.backprop_signal[i] = (int64_t(neurons.backprop_signal[i]) * 800) >> 10;
            }

            neurons.activity_ema[i] = (int64_t(neurons.activity_ema[i]) * 990 + (fired ? 1000 : 0)) / 1000;
            int32_t error = neurons.activity_ema[i] - 100;
            if (error > 0) {
                int32_t rate = (std::abs(error) > 100) ? 2 : 1;
                neurons.base_threshold[i] = saturating_add(neurons.base_threshold[i], rate);
            } else if (error < 0 && neurons.base_threshold[i] > 512) {
                neurons.base_threshold[i] = saturating_sub(neurons.base_threshold[i], 1);
            }

            neurons.proximal_potential[i] = 0;
            neurons.distal_potential[i] = 0;
            neurons.apical_potential[i] = 0;
            neurons.basal_potential[i] = 0;
        }
    }

    void cpp_plugin_tick(int32_t* bus_ptr, uint32_t bus_size, uint32_t tick) {
        if (tick % 5 == 0) {
            for (uint32_t i = 0; i < 10 && i < bus_size; ++i) {
                bus_ptr[i] += 500;
            }
        }
    }
}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

#define SCALE 1024

// FFI structures
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
    const uint32_t* offsets;
    const size_t* indices_flat;
};

// --- Physics Helpers (Device only) ---
__device__ inline int32_t sigmoid_gate_gpu(int32_t input, int32_t theta) {
    int32_t diff = input - theta;
    if (diff > 512) return 1024;
    int32_t val = diff + 512;
    if (val < 64) return 64;
    return val;
}

__device__ inline int32_t dynamic_decay_gpu(int32_t base_decay, int32_t proximal, int32_t distal) {
    int32_t liquid_mod = ((abs(proximal) + abs(distal)) * 10) >> 10;
    int32_t res = base_decay - liquid_mod;
    return res > 1 ? res : 1;
}

// --- Kernels ---

__global__ void propagate_spikes_csr_kernel(
    SynapsesFFI synapses,
    const bool* previous_spikes,
    NeuronsFFI neurons
) {
    uint32_t src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src < neurons.len) {
        if (previous_spikes[src]) {
            // CSR lookup for active source neuron
            uint32_t start = synapses.offsets[src * 16];
            uint32_t end = synapses.offsets[src * 16 + 1];

            for (uint32_t i = start; i < end; ++i) {
                size_t syn_idx = synapses.indices_flat[i];
                uint32_t target = synapses.target_index[syn_idx];

                int32_t gate = neurons.dendritic_gate[target];
                if (gate < 8) continue;

                int32_t weight = synapses.weight[syn_idx];
                uint8_t comp = synapses.compartment[syn_idx];

                // Short-Term Plasticity (STP)
                int32_t u_facil = synapses.stp_calcium[syn_idx];
                int32_t r_depr = synapses.stp_resources[syn_idx];
                int32_t stp_weight = (int64_t(weight) * r_depr) >> 10;
                stp_weight = (int64_t(stp_weight) * (1024 + u_facil)) >> 10;

                // STP Consumption
                atomicExch(&synapses.stp_resources[syn_idx], (int32_t)((int64_t(synapses.stp_resources[syn_idx]) * 800) >> 10));
                int32_t new_calc = synapses.stp_calcium[syn_idx] + 200;
                atomicExch(&synapses.stp_calcium[syn_idx], (new_calc > 1024) ? 1024 : new_calc);

                int32_t gated_weight = (int64_t(stp_weight) * gate) >> 10;

                // Atomic aggregation into compartments
                switch (comp) {
                    case 0: atomicAdd(&neurons.proximal_potential[target], gated_weight); break;
                    case 1: atomicAdd(&neurons.distal_potential[target], gated_weight); break;
                    case 2: atomicAdd(&neurons.apical_potential[target], gated_weight); break;
                    case 3: atomicAdd(&neurons.basal_potential[target], gated_weight); break;
                }
            }
        }
    }
}

__global__ void update_neurons_full_kernel(
    NeuronsFFI neurons,
    uint32_t current_tick,
    bool* new_spikes,
    int32_t ip_inc,
    int32_t ip_dec
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < neurons.len) {
        int32_t pot = neurons.potential[i];
        int32_t prox = neurons.proximal_potential[i];
        int32_t dist = neurons.distal_potential[i];
        int32_t apical = neurons.apical_potential[i];
        int32_t basal = neurons.basal_potential[i];
        int32_t g_thresh = neurons.gate_threshold[i];
        int32_t liquid = neurons.liquid_current[i];
        int32_t adaptation = neurons.adaptation_current[i];
        int32_t refr = neurons.refractory_timer[i];

        // Gated Coincidence Detection
        int32_t dist_gain = sigmoid_gate_gpu(prox, g_thresh);
        int32_t dist_gated = (int64_t(dist) * dist_gain) >> 10;
        int32_t apical_gain = sigmoid_gate_gpu(dist_gated, g_thresh);
        int32_t apical_gated = (int64_t(apical) * apical_gain) >> 10;

        // Neuromodulation / Lateral Gain
        int32_t mod_factor = (basal < 0) ? 800 : ((basal > 512) ? 1200 : 1024);

        int32_t current_pot = pot;
        current_pot += prox;
        current_pot += dist_gated;
        current_pot += apical_gated;
        current_pot += liquid;
        current_pot -= adaptation;
        current_pot = (int64_t(current_pot) * mod_factor) >> 10;

        // Leak Dynamics
        int32_t d_val = dynamic_decay_gpu(neurons.decay[i], prox, dist);
        int32_t final_pot = (int64_t(current_pot) * (1024 - d_val)) >> 10;

        // Exponential Refractory
        int32_t refr_mult = (refr > 0) ? (1 + (1 << refr)) : 1;
        int64_t eff_threshold = (int64_t)neurons.threshold[i] * refr_mult;

        bool fired = final_pot >= eff_threshold;
        new_spikes[i] = fired;

        if (fired) {
            neurons.potential[i] = 0;
            neurons.threshold[i] += ip_inc;
            neurons.adaptation_current[i] += 100;
            neurons.refractory_timer[i] = 4;
            neurons.last_spike_tick[i] = current_tick;
            neurons.backprop_signal[i] = 1024;
        } else {
            neurons.potential[i] = final_pot;
            if (neurons.threshold[i] > neurons.base_threshold[i]) {
                neurons.threshold[i] -= ip_dec;
            }
            neurons.adaptation_current[i] = (int64_t(neurons.adaptation_current[i]) * 95) / 100;
            if (refr > 0) neurons.refractory_timer[i]--;
            neurons.backprop_signal[i] = (int64_t(neurons.backprop_signal[i]) * 800) >> 10;
        }

        // Homeostasis (EMA)
        neurons.activity_ema[i] = (int64_t(neurons.activity_ema[i]) * 990 + (fired ? 1000 : 0)) / 1000;
        int32_t error = neurons.activity_ema[i] - 100;
        if (error > 0) {
            neurons.base_threshold[i] += (abs(error) > 100) ? 2 : 1;
        } else if (error < 0 && neurons.base_threshold[i] > 512) {
            neurons.base_threshold[i] -= 1;
        }

        // Reset inputs
        neurons.proximal_potential[i] = 0;
        neurons.distal_potential[i] = 0;
        neurons.apical_potential[i] = 0;
        neurons.basal_potential[i] = 0;
    }
}

__global__ void update_weights_gsop_kernel(
    SynapsesFFI synapses,
    const bool* previous_spikes,
    const bool* current_spikes,
    NeuronsFFI neurons,
    int32_t learning_rate,
    int32_t reward
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < synapses.len) {
        uint32_t src = synapses.source_index[idx];
        uint32_t tgt = synapses.target_index[idx];

        bool pre = previous_spikes[src];
        bool post = current_spikes[tgt];

        if (pre) {
            int32_t delta = (post) ? learning_rate : -learning_rate / 2;
            if (reward < 0) delta = -delta;

            // Scaled by backprop signal (SMBP)
            int32_t bprop = neurons.backprop_signal[tgt];
            delta = (int64_t(delta) * (1024 + bprop)) >> 10;

            synapses.weight[idx] += delta;
        }
    }
}

extern "C" {
    void cuda_propagate_spikes(
        SynapsesFFI synapses,
        const bool* previous_spikes,
        NeuronsFFI neurons,
        cudaStream_t stream
    ) {
        uint32_t threads = 256;
        uint32_t blocks = (neurons.len + threads - 1) / threads;
        propagate_spikes_csr_kernel<<<blocks, threads, 0, stream>>>(synapses, previous_spikes, neurons);
    }

    void cuda_update_neurons(
        NeuronsFFI neurons,
        uint32_t current_tick,
        bool* new_spikes,
        int32_t ip_inc,
        int32_t ip_dec,
        cudaStream_t stream
    ) {
        uint32_t threads = 256;
        uint32_t blocks = (neurons.len + threads - 1) / threads;
        update_neurons_full_kernel<<<blocks, threads, 0, stream>>>(neurons, current_tick, new_spikes, ip_inc, ip_dec);
    }

    void cuda_update_weights_gsop(
        SynapsesFFI synapses,
        const bool* previous_spikes,
        const bool* current_spikes,
        NeuronsFFI neurons,
        int32_t learning_rate,
        int32_t reward,
        cudaStream_t stream
    ) {
        uint32_t threads = 256;
        uint32_t blocks = (synapses.len + threads - 1) / threads;
        update_weights_gsop_kernel<<<blocks, threads, 0, stream>>>(synapses, previous_spikes, current_spikes, neurons, learning_rate, reward);
    }
}

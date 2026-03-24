#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// FFI structure matching NeuronsFFI in Rust
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

__global__ void propagate_spikes_kernel(
    const uint32_t* source_indices,
    const uint32_t* target_indices,
    const int32_t* weights,
    const uint8_t* compartments,
    uint32_t synapse_count,
    const bool* previous_spikes,
    NeuronsFFI neurons
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < synapse_count) {
        if (previous_spikes[source_indices[idx]]) {
            uint32_t target = target_indices[idx];
            int32_t weight = weights[idx];
            uint8_t comp = compartments[idx];

            // Use atomic addition for thread-safe potential updates
            switch (comp) {
                case 0: atomicAdd(&neurons.proximal_potential[target], weight); break;
                case 1: atomicAdd(&neurons.distal_potential[target], weight); break;
                case 2: atomicAdd(&neurons.apical_potential[target], weight); break;
                case 3: atomicAdd(&neurons.basal_potential[target], weight); break;
            }
        }
    }
}

// Grid-stride loop for spike propagation on CUDA
__global__ void cuda_update_neurons_kernel(
    NeuronsFFI neurons,
    uint32_t current_tick,
    bool* new_spikes,
    int32_t ip_inc,
    int32_t ip_dec
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < neurons.len) {
        // Leaky Integrate-and-Fire with Integer Physics on GPU
        int32_t pot = neurons.proximal_potential[i] + neurons.distal_potential[i];

        // Apply decay
        pot = (int64_t(pot) * (1024 - neurons.decay[i])) >> 10;

        if (pot >= neurons.threshold[i]) {
            new_spikes[i] = true;
            neurons.potential[i] = 0;
            neurons.threshold[i] += ip_inc;
        } else {
            new_spikes[i] = false;
            neurons.potential[i] = pot;
            if (neurons.threshold[i] > 1024) neurons.threshold[i] -= ip_dec;
        }

        // Reset buffers
        neurons.proximal_potential[i] = 0;
        neurons.distal_potential[i] = 0;
    }
}

extern "C" {
    void cuda_propagate_spikes(
        const uint32_t* source_indices,
        const uint32_t* target_indices,
        const int32_t* weights,
        const uint8_t* compartments,
        uint32_t synapse_count,
        const bool* previous_spikes,
        NeuronsFFI neurons,
        cudaStream_t stream
    ) {
        uint32_t threadsPerBlock = 256;
        uint32_t blocksPerGrid = (synapse_count + threadsPerBlock - 1) / threadsPerBlock;
        propagate_spikes_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            source_indices, target_indices, weights, compartments, synapse_count, previous_spikes, neurons
        );
    }

    void cuda_update_neurons(
        NeuronsFFI neurons,
        uint32_t current_tick,
        bool* new_spikes,
        int32_t ip_inc,
        int32_t ip_dec,
        cudaStream_t stream
    ) {
        uint32_t threadsPerBlock = 256;
        uint32_t blocksPerGrid = (neurons.len + threadsPerBlock - 1) / threadsPerBlock;
        cuda_update_neurons_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            neurons, current_tick, new_spikes, ip_inc, ip_dec
        );
    }
}

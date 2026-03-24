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

extern "C" {
    void cuda_propagate_spikes(
        const uint32_t* source_indices,
        const uint32_t* target_indices,
        const int32_t* weights,
        const uint8_t* compartments,
        uint32_t synapse_count,
        const bool* previous_spikes,
        NeuronsFFI neurons
    ) {
        // This is a skeleton. In a real implementation, we would manage VRAM here
        // or expect data to already be on the device.
    }
}

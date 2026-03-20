@group(0) @binding(0) var<storage, read> u_matrix: array<i32>;
@group(0) @binding(1) var<storage, read> spike_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> latent_state: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> rank: u32;
@group(1) @binding(1) var<uniform> spike_count: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let spike_idx = id.x;
    if (spike_idx >= spike_count) { return; }

    let neuron_idx = spike_indices[spike_idx];
    let offset = neuron_idx * rank;

    for (var r = 0u; r < rank; r = r + 1u) {
        atomicAdd(&latent_state[r], u_matrix[offset + r]);
    }
}

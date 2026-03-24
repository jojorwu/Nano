@group(0) @binding(0) var<storage, read> v_matrix: array<i32>;
@group(0) @binding(1) var<storage, read> latent_state: array<i32>;
@group(0) @binding(2) var<storage, read> dendritic_gates: array<i32>;
@group(0) @binding(3) var<storage, read_write> distal_potentials: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> rank: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let neuron_idx = id.x;
    if (neuron_idx >= arrayLength(&dendritic_gates)) { return; }

    var sum: f32 = 0.0;
    let n_count = arrayLength(&dendritic_gates);
    for (var r = 0u; r < rank; r = r + 1u) {
        sum = sum + f32(latent_state[r]) * f32(v_matrix[r * n_count + neuron_idx]);
    }

    let gate = f32(dendritic_gates[neuron_idx]);
    let contribution = i32((sum * gate) / 1048576.0); // SCALE^2 = 2^20

    atomicAdd(&distal_potentials[neuron_idx], contribution);
}

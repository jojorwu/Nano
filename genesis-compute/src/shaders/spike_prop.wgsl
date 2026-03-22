@group(0) @binding(0) var<storage, read> source_indices: array<u32>;
@group(0) @binding(1) var<storage, read> target_indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<i32>;
@group(0) @binding(3) var<storage, read> prev_spikes: array<u32>;
@group(0) @binding(4) var<storage, read> compartments: array<u32>;

@group(0) @binding(5) var<storage, read_write> proximal_potentials: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> distal_potentials: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> apical_potentials: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> basal_potentials: array<atomic<i32>>;

@group(0) @binding(9) var<storage, read> dendritic_gates: array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&source_indices)) { return; }

    let src = source_indices[idx];
    if (prev_spikes[src] != 0u) {
        let target = target_indices[idx];
        let gate = dendritic_gates[target];
        if (gate < 8) { return; }

        let gated_weight = i32((i64(weights[idx]) * i64(gate)) >> 10);
        let comp = compartments[idx];

        if (comp == 0u) {
            atomicAdd(&proximal_potentials[target], gated_weight);
        } else if (comp == 1u) {
            atomicAdd(&distal_potentials[target], gated_weight);
        } else if (comp == 2u) {
            atomicAdd(&apical_potentials[target], gated_weight);
        } else if (comp == 3u) {
            atomicAdd(&basal_potentials[target], gated_weight);
        }
    }
}

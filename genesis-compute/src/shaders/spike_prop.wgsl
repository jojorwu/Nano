@group(0) @binding(0) var<storage, read> source_indices: array<u32>;
@group(0) @binding(1) var<storage, read> target_indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<i32>;
@group(0) @binding(3) var<storage, read> delays: array<u32>;
@group(0) @binding(4) var<storage, read> compartments: array<u32>;

@group(0) @binding(5) var<storage, read_write> proximal_potentials: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> distal_potentials: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> apical_potentials: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> basal_potentials: array<atomic<i32>>;

@group(0) @binding(9) var<storage, read> dendritic_gates: array<i32>;
@group(0) @binding(10) var<storage, read> spike_history: array<u32>; // Circular buffer [16 * neurons]
@group(1) @binding(0) var<uniform> current_tick: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&source_indices)) { return; }

    let src = source_indices[idx];
    let delay = delays[idx];
    let n_count = arrayLength(&dendritic_gates);

    // Get spike state from history based on axonal delay (1-16 ticks)
    // Delay=1 is the previous tick spikes.
    // Spike history is stored as: [slot_0: n_count][slot_1: n_count]...[slot_15: n_count]

    // Calculate history slot: (current_tick - delay) % 16
    let slot = (current_tick + 16u - delay) % 16u;
    let fired = spike_history[slot * n_count + src];

    if (fired != 0u) {
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

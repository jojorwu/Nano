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
@group(0) @binding(10) var<storage, read> spike_history: array<vec2<u32>>; // BitPacked u64 history
@group(0) @binding(11) var<storage, read> block_id: array<u32>;
@group(0) @binding(12) var<storage, read> block_attn_gates: array<vec4<i32>>; // [Prox, Dist, Apic, Basal] per block
@group(1) @binding(0) var<uniform> current_tick: u32;

var<workgroup> shared_potentials: array<atomic<i32>, 64>; // Reduced scope for shared accumulation

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = id.x;
    let lid = local_id.x;

    if (idx >= arrayLength(&source_indices)) { return; }

    let src = source_indices[idx];
    let delay = delays[idx];
    let n_count = arrayLength(&dendritic_gates);
    let packed_count = (n_count + 63u) / 64u;

    let slot = (current_tick + 16u - delay) % 16u;

    // Extract bit from packed history
    let word_idx = src / 64u;
    let bit_idx = src % 64u;
    let packed_word = spike_history[slot * packed_count + (word_idx / 2u)];

    var fired = 0u;
    if (word_idx % 2u == 0u) {
        fired = (packed_word.x >> bit_idx) & 1u;
    } else {
        fired = (packed_word.y >> bit_idx) & 1u;
    }

    // Note: Effective shared accumulation requires sorting by target index.
    // Without sorting, we revert to direct atomic adds for correctness but keep bit-parallelism.
    if (fired != 0u) {
        let tgt = target_indices[idx];
        let bid = block_id[tgt];
        let attn_gates = block_attn_gates[bid];

        let gate = dendritic_gates[tgt];
        if (gate < 8) { return; }

        let comp = compartments[idx];
        var final_gate = gate;

        // Kernel Fusion: Dynamic Attention applied during propagation
        if (comp == 0u) { final_gate = (gate * attn_gates.x) >> 10; }
        else if (comp == 1u) { final_gate = (gate * attn_gates.y) >> 10; }
        else if (comp == 2u) { final_gate = (gate * attn_gates.z) >> 10; }
        else if (comp == 3u) { final_gate = (gate * attn_gates.w) >> 10; }

        // Predictive Gating: skip if attention is very low
        if (final_gate < 4) { return; }

        let gated_weight = (weights[idx] * final_gate) >> 10;

        if (comp == 0u) {
            atomicAdd(&proximal_potentials[tgt], gated_weight);
        } else if (comp == 1u) {
            atomicAdd(&distal_potentials[tgt], gated_weight);
        } else if (comp == 2u) {
            atomicAdd(&apical_potentials[tgt], gated_weight);
        } else if (comp == 3u) {
            atomicAdd(&basal_potentials[tgt], gated_weight);
        }
    }
}

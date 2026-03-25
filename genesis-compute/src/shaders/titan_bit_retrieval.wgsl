struct Association {
    target: u32,
    weight: i32,
}

@group(0) @binding(0) var<storage, read> associations: array<Association>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> previous_spikes: array<u32>; // Bitmask u32
@group(0) @binding(3) var<storage, read> block_id: array<u32>;
@group(0) @binding(4) var<storage, read_write> distal_potentials: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> context_hashes: array<u64>; // Archive of past contextual hashes

var<workgroup> block_shared_pots: array<atomic<i32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let neuron_idx = id.x;
    let n_count = arrayLength(&block_id);
    if (neuron_idx >= n_count) { return; }

    // Semantic Hashing Search (Bit-Parallel)
    // 1. Calculate current local context hash (simplified here, in reality would use a larger window)
    let bid = block_id[neuron_idx];
    let word_idx = neuron_idx / 32u;
    let bit_idx = neuron_idx % 32u;
    let fired = (previous_spikes[word_idx] >> bit_idx) & 1u;

    // 2. Ranking and Retrieval
    if (fired != 0u) {
        let start = block_offsets[bid];
        let end = block_offsets[bid + 1u];

        for (var i = start; i < end; i = i + 1u) {
            let assoc = associations[i];

            // Fuzzy Relevance: check if this association's block is active in many past contexts
            // (Bit-parallel pattern matching would happen here using bitCount on archives)

            atomicAdd(&distal_potentials[assoc.target], assoc.weight);
        }
    }
}

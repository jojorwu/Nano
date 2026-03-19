@group(0) @binding(0) var<storage, read> source_indices: array<u32>;
@group(0) @binding(1) var<storage, read> target_indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<i32>;
@group(0) @binding(3) var<storage, read> prev_spikes: array<u32>;
@group(0) @binding(4) var<storage, read_write> current_inputs: array<atomic<i32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&source_indices)) { return; }

    let src = source_indices[idx];
    if (prev_spikes[src] != 0u) {
        let target = target_indices[idx];
        let weight = weights[idx];
        atomicAdd(&current_inputs[target], weight);
    }
}

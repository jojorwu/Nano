@group(0) @binding(0) var<storage, read_write> weights: array<i32>;
@group(0) @binding(1) var<storage, read> source_indices: array<u32>;
@group(0) @binding(2) var<storage, read> target_indices: array<u32>;
@group(0) @binding(3) var<storage, read> pre_spikes: array<u32>;
@group(0) @binding(4) var<storage, read> post_spikes: array<u32>;
@group(1) @binding(0) var<uniform> learning_rate: i32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&weights)) { return; }

    let src = source_indices[idx];
    let tgt = target_indices[idx];

    let pre_spiked = pre_spikes[src] != 0u;
    let post_spiked = post_spikes[tgt] != 0u;

    var w = weights[idx];

    if (pre_spiked && post_spiked) {
        w = w + learning_rate;
    } else if (pre_spiked && !post_spiked) {
        w = w - (learning_rate / 2);
    }

    // Clamp weight (-5120 to 5120, assuming SCALE=1024)
    if (w > 5120) { w = 5120; }
    if (w < -5120) { w = -5120; }

    weights[idx] = w;
}

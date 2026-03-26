struct NeuronState {
    potential: i32,
    threshold: i32,
    decay: i32,
    refractory: i32,
    next_update: u32,
    last_spike_tick: u32,
    backprop_signal: i32,
    adaptation: i32,
    activity_ema: i32,
    base_threshold: i32,
    distal_gate: i32,
    apical_gate: i32,
    basal_gate: i32,
    block_id: u32,
    action: i32,
    packed: vec2<u32>, // u64 mirror
    plasticity_gate: i32,
    astro_calcium: i32,
    is_remote: u32,
    origin_node_id: u32,
    energy_level: i32,
    specialization_score: f32,
    liquid_current: i32,
    update_interval: u32,
}

@group(0) @binding(0) var<storage, read_write> weights: array<i32>;
@group(0) @binding(1) var<storage, read> source_indices: array<u32>;
@group(0) @binding(2) var<storage, read> target_indices: array<u32>;
@group(0) @binding(3) var<storage, read> pre_spikes: array<u32>;
@group(0) @binding(4) var<storage, read> post_spikes: array<u32>;
@group(0) @binding(5) var<storage, read> compartments: array<u32>;
@group(0) @binding(6) var<storage, read> neuron_states: array<NeuronState>;
@group(0) @binding(8) var<storage, read> is_excitatory: array<u32>;

struct Modulation {
    dopamine: i32,
    noradrenaline: i32,
    serotonin: i32,
}
@group(0) @binding(9) var<uniform> modulation: Modulation;
@group(1) @binding(0) var<uniform> base_learning_rate: i32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&weights)) { return; }

    let src = source_indices[idx];
    let tgt = target_indices[idx];
    let comp = compartments[idx];

    let pre_spiked = pre_spikes[src] != 0u;
    let post_spiked = post_spikes[tgt] != 0u;

    if (!pre_spiked && !post_spiked) { return; }

    var lr = i32(base_learning_rate);
    if (comp == 0u) { // Proximal
    } else if (comp == 1u) { // Distal
        lr = (lr * 8) / 10;
    } else {
        lr = lr / 2;
    }

    let neuromod_gain = 1024 + modulation.noradrenaline;
    let dopamine_gain = 1024 + abs(modulation.dopamine);

    var smbp_mod = 1024;
    if (comp != 0u) {
        smbp_mod = 1024 + neuron_states[tgt].backprop_signal;
    }

    // Manual 64-bit multiplication approximation using 32-bit components
    // (a * b * c * d) >> 30
    // Simplified since we expect values around 1024 (2^10)
    // lr_final = (lr * gain_a >> 10) * (gain_b * gain_c >> 10) >> 10

    // CPU-identical rounding: (lr * gain) >> 10
    var lr_acc: i32 = (lr * neuromod_gain) / 1024;
    lr_acc = (lr_acc * dopamine_gain) / 1024;
    lr_acc = (lr_acc * smbp_mod) / 1024;
    let lr_final = lr_acc;

    var w = weights[idx];
    let old_w = w;

    if (pre_spiked && post_spiked) {
        w = w + lr_final;
    } else if (pre_spiked && !post_spiked) {
        w = w - (lr_final / 2);
    }

    if (old_w > 0 && w < 0) { w = 1; }
    if (old_w < 0 && w > 0) { w = -1; }
    if (w > 5120) { w = 5120; }
    if (w < -5120) { w = -5120; }

    weights[idx] = w;
}

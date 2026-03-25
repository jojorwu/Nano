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
}

@group(0) @binding(0) var<storage, read_write> neuron_states: array<NeuronState>;
@group(0) @binding(1) var<storage, read> distal_potentials: array<i32>;
@group(0) @binding(2) var<storage, read> proximal_potentials: array<i32>;
@group(0) @binding(3) var<storage, read> apical_potentials: array<i32>;
@group(0) @binding(4) var<storage, read> basal_potentials: array<i32>;

struct Config {
    ip_increment: i32,
    ip_decay: i32,
    noise_amplitude: i32,
}
@group(0) @binding(5) var<storage, read> config: Config;
@group(0) @binding(6) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(7) var<storage, read_write> sparse_spikes: array<u32>;
@group(0) @binding(8) var<storage, read_write> spike_counter: atomic<u32>;
@group(0) @binding(9) var<storage, read> inputs: array<i32>;
@group(0) @binding(10) var<storage, read> intervals: array<u32>;
@group(0) @binding(11) var<storage, read> dendritic_gate: array<i32>;
@group(0) @binding(12) var<storage, read> gate_thresholds: array<i32>;
@group(0) @binding(13) var<storage, read> expert_mask: array<u32>;

struct Modulation {
    dopamine: i32,
    noradrenaline: i32,
    serotonin: i32,
}
@group(0) @binding(23) var<uniform> modulation: Modulation;
@group(1) @binding(0) var<uniform> current_tick: u32;

fn xorshift(seed: u32) -> u32 {
    var x = seed;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&neuron_states)) { return; }

    // Dynamic MoE: Expert Freezing
    // If all dendritic gates for this neuron are very low, skip potential calculation to save power/cycles
    let st = neuron_states[i];
    if (st.distal_gate < 16 && st.apical_gate < 16 && st.basal_gate < 16 && dendritic_gate[i] < 16) {
        spikes[i] = 0u;
        return;
    }

    if (expert_mask[i] == 0u) { return; }
    if (current_tick < neuron_states[i].next_update) { return; }

    if (neuron_states[i].refractory > 0) {
        neuron_states[i].refractory = neuron_states[i].refractory - 1;
        neuron_states[i].potential = 0;
        spikes[i] = 0u;
        neuron_states[i].next_update = current_tick + intervals[i];
        return;
    }

    let proximal = proximal_potentials[i];
    let distal = (distal_potentials[i] * neuron_states[i].distal_gate) >> 10;
    let apical = (apical_potentials[i] * neuron_states[i].apical_gate) >> 10;
    let basal = (basal_potentials[i] * neuron_states[i].basal_gate) >> 10;

    let gate_threshold = gate_thresholds[i];
    let dist_diff = proximal - gate_threshold;
    var dist_gain = 64;
    if (dist_diff > 512) { dist_gain = 1024; }
    else if (dist_diff > -512) { dist_gain = dist_diff + 512; }
    let dist_gated = (distal * dist_gain) >> 10;

    let apical_diff = dist_gated - gate_threshold;
    var apical_gain = 64;
    if (apical_diff > 512) { apical_gain = 1024; }
    else if (apical_diff > -512) { apical_gain = apical_diff + 512; }
    let apical_gated = (apical * apical_gain) >> 10;

    var mod_factor = 1024;
    if (basal < 0) { mod_factor = 800; }
    else if (basal > 512) { mod_factor = 1200; }
    mod_factor = (mod_factor * (1024 + modulation.noradrenaline)) >> 10;

    let gated_input = (inputs[i] * dendritic_gate[i]) >> 10;

    var noise = 0;
    if (config.noise_amplitude > 0) {
         let seed = i ^ current_tick;
         let amplitude = u32(config.noise_amplitude);
         noise = i32(xorshift(seed) % (amplitude * 2u)) - i32(amplitude);
    }

    var pot = neuron_states[i].potential + gated_input + proximal + dist_gated + apical_gated + noise - neuron_states[i].adaptation;
    pot = (pot * mod_factor) >> 10;

    let liquid_mod = ((abs(proximal) + abs(distal_potentials[i])) * 10) >> 10;
    let final_decay = max(1, neuron_states[i].decay - liquid_mod);
    pot = (pot * (1024 - final_decay)) >> 10;

    var refr_mult: i32 = 1;
    if (neuron_states[i].refractory > 0) {
        refr_mult = 1 + i32(1u << u32(neuron_states[i].refractory));
    }
    let effective_threshold = neuron_states[i].threshold * refr_mult;

    if (pot >= effective_threshold) {
        neuron_states[i].potential = 0;
        neuron_states[i].refractory = 4;
        spikes[i] = 1u;
        neuron_states[i].last_spike_tick = current_tick;
        neuron_states[i].adaptation = neuron_states[i].adaptation + 100;
        neuron_states[i].backprop_signal = 1024;

        let count = atomicAdd(&spike_counter, 1u);
        if (count < arrayLength(&sparse_spikes)) {
            sparse_spikes[count] = i;
        }
        neuron_states[i].threshold = neuron_states[i].threshold + config.ip_increment;
    } else {
        neuron_states[i].potential = pot;
        spikes[i] = 0u;
        if (neuron_states[i].threshold > neuron_states[i].base_threshold) {
            neuron_states[i].threshold = neuron_states[i].threshold - config.ip_decay;
        }
        neuron_states[i].backprop_signal = (neuron_states[i].backprop_signal * 800) >> 10;
        neuron_states[i].adaptation = (neuron_states[i].adaptation * 972) >> 10;
    }

    var cur_ema = neuron_states[i].activity_ema;
    if (pot >= effective_threshold) {
        cur_ema = (cur_ema * 990 + 1000) / 1000;
    } else {
        cur_ema = (cur_ema * 990) / 1000;
    }
    neuron_states[i].activity_ema = cur_ema;

    var homeo_rate = 1;
    if (cur_ema > 200) { homeo_rate = 2; }
    if (cur_ema > 100) {
        neuron_states[i].base_threshold = neuron_states[i].base_threshold + homeo_rate;
    } else if (cur_ema < 100 && neuron_states[i].base_threshold > 512) {
        neuron_states[i].base_threshold = neuron_states[i].base_threshold - 1;
    }

    neuron_states[i].next_update = current_tick + intervals[i];
}

struct Neuron {
    potential: i32,
    threshold: i32,
    decay: i32,
    refractory: i32,
}

@group(0) @binding(0) var<storage, read_write> potentials: array<i32>;
@group(0) @binding(9) var<storage, read> distal_potentials: array<i32>;
@group(0) @binding(10) var<storage, read> proximal_potentials: array<i32>;
@group(0) @binding(17) var<storage, read> apical_potentials: array<i32>;
@group(0) @binding(18) var<storage, read> basal_potentials: array<i32>;
@group(0) @binding(1) var<storage, read_write> thresholds: array<i32>;
@group(0) @binding(11) var<storage, read> base_thresholds: array<i32>;
@group(0) @binding(14) var<storage, read> layer_ids: array<u32>;
@group(0) @binding(15) var<storage, read_write> backprop_signals: array<i32>;
struct Config {
    ip_increment: i32,
    ip_decay: i32,
    noise_amplitude: i32,
}
@group(0) @binding(16) var<storage, read> config: Config;
@group(0) @binding(2) var<storage, read_write> decays: array<i32>;
@group(0) @binding(3) var<storage, read_write> refractory: array<i32>;
@group(0) @binding(4) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(12) var<storage, read_write> sparse_spikes: array<u32>;
@group(0) @binding(13) var<storage, read_write> spike_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read> inputs: array<i32>;
@group(0) @binding(6) var<storage, read_write> next_update: array<u32>;
@group(0) @binding(7) var<storage, read> intervals: array<u32>;
@group(0) @binding(8) var<storage, read> dendritic_gate: array<i32>;
@group(0) @binding(19) var<storage, read> gate_thresholds: array<i32>;
@group(0) @binding(20) var<storage, read> expert_mask: array<u32>;
@group(0) @binding(21) var<storage, read_write> last_spike_ticks: array<u32>;
@group(0) @binding(24) var<storage, read_write> adaptation: array<i32>;
@group(0) @binding(25) var<storage, read_write> activity_ema: array<i32>;
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
    if (i >= arrayLength(&potentials)) { return; }

    // Mixture-of-Experts: Skip updates if neuron is masked out
    if (expert_mask[i] == 0u) { return; }

    if (current_tick < next_update[i]) { return; }

    if (refractory[i] > 0) {
        refractory[i] = refractory[i] - 1;
        potentials[i] = 0;
        spikes[i] = 0;
        next_update[i] = current_tick + intervals[i];
        return;
    }

    let proximal = proximal_potentials[i];
    let distal = distal_potentials[i];
    let apical = apical_potentials[i];
    let basal = basal_potentials[i];

    // Hierarchical Sigmoidal Gating (Smooth NMDA-like response)
    let gate_threshold = gate_thresholds[i];

    // Distal gated by Proximal
    let dist_diff = proximal - gate_threshold;
    var dist_gain = 64; // Minimal leakage
    if (dist_diff > 512) { dist_gain = 1024; }
    else if (dist_diff > -512) { dist_gain = dist_diff + 512; }
    let dist_gated = (distal * dist_gain) >> 10;

    // Apical gated by Distal
    let apical_diff = dist_gated - gate_threshold;
    var apical_gain = 64;
    if (apical_diff > 512) { apical_gain = 1024; }
    else if (apical_diff > -512) { apical_gain = apical_diff + 512; }
    let apical_gated = (apical * apical_gain) >> 10;

    // Basal Modulation (Lateral inhibition/excitation)
    var mod_factor = 1024;
    if (basal < 0) { mod_factor = 800; }
    else if (basal > 512) { mod_factor = 1200; }

    // Neuromodulation: Noradrenaline increases gain/arousal
    mod_factor = (mod_factor * (1024 + modulation.noradrenaline)) >> 10;

    let gated_input = (inputs[i] * dendritic_gate[i]) >> 10;

    // Add Stochastic Noise (PRNG)
    var noise = 0;
    if (config.noise_amplitude > 0) {
         let seed = i ^ current_tick;
         let amplitude = u32(config.noise_amplitude);
         let rng = i32(xorshift(seed) % (amplitude * 2u)) - i32(amplitude);
         noise = rng;
    }

    var pot = potentials[i] + gated_input + proximal + dist_gated + apical_gated + noise - adaptation[i];
    pot = (pot * mod_factor) >> 10;

    // LLIF: Dynamic Decay
    let liquid_mod = ((abs(proximal) + abs(distal)) * 10) >> 10;
    let final_decay = max(1, decays[i] - liquid_mod);

    pot = (pot * (1024 - final_decay)) >> 10;

    // Relative Refractory: Exponentially decaying threshold multiplier
    var refr_mult = 1;
    if (refractory[i] > 0) { refr_mult = 1 + (1 << u32(refractory[i])); }
    let effective_threshold = thresholds[i] * refr_mult;

    if (pot >= effective_threshold) {
        potentials[i] = 0;
        refractory[i] = 4;
        spikes[i] = 1;
        last_spike_ticks[i] = current_tick;
        adaptation[i] = adaptation[i] + 100;

        // SMBP: Active backpropagation signal
        backprop_signals[i] = 1024;

        let count = atomicAdd(&spike_counter, 1u);
        if (count < arrayLength(&sparse_spikes)) {
            // Simple index storage for now, bit-packing can be done in a separate kernel if needed.
            sparse_spikes[count] = i;
        }

        // Intrinsic Plasticity
        thresholds[i] = thresholds[i] + config.ip_increment;

    } else {
        potentials[i] = pot;
        spikes[i] = 0;
        // Threshold decay back to base (Intrinsic Plasticity)
        if (thresholds[i] > base_thresholds[i]) {
            thresholds[i] = thresholds[i] - config.ip_decay;
        }
        // Decaying backprop signal
        backprop_signals[i] = (backprop_signals[i] * 800) >> 10;
        adaptation[i] = (adaptation[i] * 972) >> 10; // ~95% recovery
    }

    // Homeostatic Activity Control (Matches CPU logic)
    var cur_ema = activity_ema[i];
    if (pot >= effective_threshold) {
        cur_ema = (cur_ema * 990 + 1000) / 1000;
    } else {
        cur_ema = (cur_ema * 990) / 1000;
    }
    activity_ema[i] = cur_ema;

    var homeo_rate = 1;
    if (cur_ema > 200) { homeo_rate = 2; } // Accelerated adjustment

    if (cur_ema > 100) {
        base_thresholds[i] = base_thresholds[i] + homeo_rate;
    } else if (cur_ema < 100 && base_thresholds[i] > 512) {
        base_thresholds[i] = base_thresholds[i] - 1;
    }

    next_update[i] = current_tick + intervals[i];
}

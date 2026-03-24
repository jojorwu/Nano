use genesis_core::{IValue, SCALE};
use genesis_core::physics::{sigmoid_gate_approx, calculate_dynamic_decay};

pub fn calculate_membrane_potential(
    base_pot: IValue,
    proximal: IValue,
    distal: IValue,
    apical: IValue,
    basal: IValue,
    gate_threshold: IValue,
    liquid: IValue, // Reserved
    decay: IValue,
    noise_amp: IValue,
    adaptation: IValue
) -> IValue {
    // Non-linear Sigmoidal Dendritic Gating (Smooth NMDA-like response)
    let dist_gain = sigmoid_gate_approx(proximal, gate_threshold);
    let dist_gated = ((distal as i64 * dist_gain as i64) >> 10) as i32;

    let apical_gain = sigmoid_gate_approx(dist_gated, gate_threshold);
    let apical_gated = ((apical as i64 * apical_gain as i64) >> 10) as i32;

    // Basal Modulation (Lateral inhibition/excitation)
    let mod_factor = if basal < 0 { 800 } else if basal > 512 { 1200 } else { 1024 };

    // Fast Xorshift PRNG for CPU Neural Noise
    let noise = if noise_amp > 0 {
        thread_local! {
            static SEED: std::cell::Cell<u32> = std::cell::Cell::new(0xDEADBEEF);
        }
        SEED.with(|s| {
            let mut x = s.get();
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            s.set(x);
            (x % (noise_amp as u32 * 2)) as i32 - noise_amp as i32
        })
    } else { 0 };

    let mut pot = base_pot.saturating_add(proximal)
        .saturating_add(dist_gated)
        .saturating_add(apical_gated)
        .saturating_add(liquid)
        .saturating_add(noise)
        .saturating_sub(adaptation);

    // Apply neuromodulatory/basal scaling
    pot = ((pot as i64 * mod_factor as i64) >> 10) as i32;

    // LLIF: Leaky Integrate-and-Fire Dynamics
    let final_decay = calculate_dynamic_decay(decay, proximal, distal);

    // Pot = Pot * (1 - decay/SCALE)
    ((pot as i64 * (SCALE - final_decay) as i64) >> 10) as i32
}

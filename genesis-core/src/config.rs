use crate::{IValue, SCALE};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
    pub default_threshold: IValue,
    pub default_decay: IValue,
    pub learning_rate: IValue,

    // Advanced Structural Plasticity
    pub max_synapses: usize,
    pub max_neurons: usize,
    pub neurogenesis_reward_threshold: IValue,
    pub pruning_threshold: IValue,

    // Dendritic Gating
    pub dendritic_coincidence_threshold: IValue,

    // Intrinsic Plasticity (Core Memory / Adaptive Thresholds)
    pub ip_increment: IValue,
    pub ip_decay: IValue,

    // STDP Parameters
    pub stdp_tau: u64,
    pub stdp_a_plus: IValue,
    pub stdp_a_minus: IValue,

    // Metaplasticity
    pub metaplasticity_enabled: bool,

    // Stochastic Firing (Neural Noise)
    pub noise_amplitude: IValue, // SCALE = 1.0 (max noise)

    // Hardware Backend
    pub preferred_backend: String,

    // Titan Memory Configuration
    pub titan_byte_memory_size: usize,
    pub titan_max_associations: usize,
    pub titan_max_blocks: u32,
    pub titan_max_entries_per_block: usize,
    pub titan_surprise_threshold: IValue,

    // Global Physics & Plasticity
    pub default_refractory_ticks: i32,
    pub weight_clamp_limit: IValue,
    pub smbp_decay: i64,
    pub activity_ema_alpha: i64,
    pub target_activity_level: IValue,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            default_threshold: SCALE,
            default_decay: 50,
            learning_rate: 10,
            max_synapses: 1_000_000,
            max_neurons: 100_000,
            neurogenesis_reward_threshold: 200,
            pruning_threshold: 10,
            dendritic_coincidence_threshold: 512, // 0.5 * SCALE
            ip_increment: 50,
            ip_decay: 1,
            stdp_tau: 20,
            stdp_a_plus: 100,
            stdp_a_minus: 100,
            metaplasticity_enabled: true,
            noise_amplitude: 50, // 5% noise by default
            preferred_backend: "cpu".to_string(),
            titan_byte_memory_size: 1024 * 1024, // 1MB default
            titan_max_associations: 1_000_000,
            titan_max_blocks: 1_000_000,
            titan_max_entries_per_block: 256,
            titan_surprise_threshold: 100,
            default_refractory_ticks: 4,
            weight_clamp_limit: SCALE * 5,
            smbp_decay: 800,
            activity_ema_alpha: 990, // Alpha out of 1000 (0.99)
            target_activity_level: 100,
        }
    }
}

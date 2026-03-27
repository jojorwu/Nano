use crate::{IValue, SCALE};
use serde::{Serialize, Deserialize};

/// Centralized Configuration for the Neural Network and Simulation Engine.
/// This struct defines all hyperparameters, hardware settings, and architectural limits.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
    pub physics: PhysicsConfig,
    pub plasticity: PlasticityConfig,
    pub titan: TitanConfig,
    pub modules: ModuleConfig,
    pub hardware: HardwareConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PhysicsConfig {
    pub default_threshold: IValue,
    pub default_decay: IValue,
    pub default_refractory_ticks: i32,
    pub noise_amplitude: IValue,
    pub smbp_decay: i64,
    pub activity_ema_alpha: i64,
    pub target_activity_level: IValue,
    pub dendritic_coincidence_threshold: IValue,
    pub theta_rhythm: bool,
    pub theta_frequency: f32,
    pub specialization_decay: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PlasticityConfig {
    pub learning_rate: IValue,
    pub weight_clamp_limit: IValue,
    pub metaplasticity_enabled: bool,
    pub max_synapses: usize,
    pub max_neurons: usize,
    pub neurogenesis_reward_threshold: IValue,
    pub pruning_threshold: IValue,
    pub homeostatic_scaling_limit: i64,
    pub stdp_tau: u64,
    pub stdp_a_plus: IValue,
    pub stdp_a_minus: IValue,
    pub ip_increment: IValue,
    pub ip_decay: IValue,
    pub intrinsic_learning_rate: IValue,
    pub stp_resource_decay: i64,
    pub stp_calcium_decay: i64,
    pub stp_resource_recovery: i64,
    pub stp_calcium_recovery: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TitanConfig {
    pub byte_memory_size: usize,
    pub max_associations: usize,
    pub max_blocks: u32,
    pub max_entries_per_block: usize,
    pub surprise_threshold: IValue,
    pub deep_replay_threshold: IValue,
    pub elastic_window_max: usize,
    pub decay_rate: IValue,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ModuleConfig {
    pub think_max_ticks: usize,
    pub think_surprise_threshold_deep: IValue,
    pub think_surprise_threshold_low: IValue,
    pub workspace_broadcast_intensity: IValue,
    pub workspace_ignition_threshold: f32,
    pub hierarchical_control_enabled: bool,
    pub top_down_gain: IValue,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HardwareConfig {
    pub preferred_backend: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsConfig {
                default_threshold: SCALE,
                default_decay: 50,
                default_refractory_ticks: 4,
                noise_amplitude: 50,
                smbp_decay: 800,
                activity_ema_alpha: 990,
                target_activity_level: 100,
                dendritic_coincidence_threshold: 512,
                theta_rhythm: true,
                theta_frequency: 0.1,
                specialization_decay: 0.99,
            },
            plasticity: PlasticityConfig {
                learning_rate: 10,
                weight_clamp_limit: SCALE * 5,
                metaplasticity_enabled: true,
                max_synapses: 1_000_000,
                max_neurons: 100_000,
                neurogenesis_reward_threshold: 200,
                pruning_threshold: 10,
                homeostatic_scaling_limit: 1024 * 16,
                stdp_tau: 20,
                stdp_a_plus: 100,
                stdp_a_minus: 100,
                ip_increment: 50,
                ip_decay: 1,
                intrinsic_learning_rate: 5,
                stp_resource_decay: 800,
                stp_calcium_decay: 95,
                stp_resource_recovery: 99,
                stp_calcium_recovery: 200,
            },
            titan: TitanConfig {
                byte_memory_size: 1024 * 1024,
                max_associations: 1_000_000,
                max_blocks: 1_000_000,
                max_entries_per_block: 256,
                surprise_threshold: 100,
                deep_replay_threshold: 1500,
                elastic_window_max: 16,
                decay_rate: 1,
            },
            modules: ModuleConfig {
                think_max_ticks: 100,
                think_surprise_threshold_deep: 1500,
                think_surprise_threshold_low: 100,
                workspace_broadcast_intensity: SCALE / 2,
                workspace_ignition_threshold: 10.0,
                hierarchical_control_enabled: true,
                top_down_gain: SCALE / 4,
            },
            hardware: HardwareConfig {
                preferred_backend: "cpu".to_string(),
            },
        }
    }
}

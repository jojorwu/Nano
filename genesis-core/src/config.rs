use crate::{IValue, SCALE};
use serde::{Serialize, Deserialize};

/// Centralized Configuration for the Neural Network and Simulation Engine.
/// This struct defines all hyperparameters, hardware settings, and architectural limits.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
    /// Default firing threshold for new neurons.
    pub default_threshold: IValue,
    /// Default Leaky Integrate-and-Fire (LIF) decay rate.
    pub default_decay: IValue,
    /// Global learning rate for synaptic updates.
    pub learning_rate: IValue,

    // --- Advanced Structural Plasticity ---
    /// Maximum allowed synapses in the network.
    pub max_synapses: usize,
    /// Maximum allowed neurons (population cap).
    pub max_neurons: usize,
    /// Reward threshold to trigger dynamic neurogenesis.
    pub neurogenesis_reward_threshold: IValue,
    /// Absolute weight threshold for synaptic pruning.
    pub pruning_threshold: IValue,

    // --- Dendritic Gating ---
    /// Potential required in proximal compartment to open distal gates.
    pub dendritic_coincidence_threshold: IValue,

    // --- Intrinsic Plasticity (IP) ---
    /// Amount to increase threshold upon firing (Adaptive Threshold).
    pub ip_increment: IValue,
    /// Rate at which thresholds decay back to base level.
    pub ip_decay: IValue,

    // --- STDP Parameters ---
    /// Temporal window for LTP/LTD (ticks).
    pub stdp_tau: u64,
    /// Magnitude of Long-Term Potentiation (LTP).
    pub stdp_a_plus: IValue,
    /// Magnitude of Long-Term Depression (LTD).
    pub stdp_a_minus: IValue,

    // --- Metaplasticity ---
    /// Enable/Disable higher-order regulation of learning rates.
    pub metaplasticity_enabled: bool,

    // --- Stochastic Firing (Neural Noise) ---
    /// Amplitude of neural noise injected into membrane potentials.
    pub noise_amplitude: IValue,

    // --- Hardware Backend ---
    /// Preferred computing backend ("cpu", "wgpu", "cuda").
    pub preferred_backend: String,

    // --- Titan Memory Configuration ---
    /// Size of the byte-addressable memory buffer (RAM).
    pub titan_byte_memory_size: usize,
    /// Total capacity for associative memory links.
    pub titan_max_associations: usize,
    /// Maximum number of functional blocks (mini-columns).
    pub titan_max_blocks: u32,
    /// Max associations per block to prevent hub-neuron saturation.
    pub titan_max_entries_per_block: usize,
    /// Surprise level required to trigger new association learning.
    pub titan_surprise_threshold: IValue,
    /// Surprise level required to archive patterns for night replay.
    pub titan_deep_replay_threshold: IValue,
    /// Maximum depth of the temporal history search.
    pub titan_elastic_window_max: usize,
    /// Rate of weight decay for associative memories.
    pub titan_decay_rate: IValue,

    // --- Thinking (Reasoning) Module Configuration ---
    /// Maximum internal sub-ticks allowed per external tick.
    pub think_max_ticks: usize,
    /// Surprise trigger for deep "fantasy" reasoning.
    pub think_surprise_threshold_deep: IValue,
    /// Surprise floor to stop internal reasoning iterations.
    pub think_surprise_threshold_low: IValue,

    // --- Global Physics & Plasticity ---
    /// Default refractory period after firing.
    pub default_refractory_ticks: i32,
    /// Hard limit for synaptic weight magnitude.
    pub weight_clamp_limit: IValue,
    /// Decay rate for Somato-Dendritic Back-Propagation (SDBP) signals.
    pub smbp_decay: i64,
    /// Smoothing factor for the neuron activity monitor (0-1000).
    pub activity_ema_alpha: i64,
    /// Goal firing rate (Homeostatic Activity Control).
    pub target_activity_level: IValue,
    /// Maximum total synaptic weight a single neuron can receive.
    pub homeostatic_scaling_limit: i64,

    // --- STP (Short-Term Plasticity) Parameters ---
    /// Rate at which neurotransmitter resources are depleted.
    pub stp_resource_decay: i64,
    /// Rate at which local calcium levels decay.
    pub stp_calcium_decay: i64,
    /// Rate at which resources recover over time.
    pub stp_resource_recovery: i64,
    /// Calcium increment per spike.
    pub stp_calcium_recovery: i64,

    // --- GNW (Global Neuronal Workspace) Parameters ---
    /// Magnitude of global signal broadcasting during ignition.
    pub workspace_broadcast_intensity: IValue,
    /// Activity required for a block to enter the workspace.
    pub workspace_ignition_threshold: f32,

    // --- Astrocytic Parameters ---
    /// Calcium increment in astrocytes per local neural spike.
    pub astro_increment: IValue,
    /// Decay rate of astrocytic calcium (slow dynamics).
    pub astro_decay_rate: i64,

    // --- Hierarchical Control ---
    /// Enable/Disable top-down hierarchical predictions.
    pub hierarchical_control_enabled: bool,
    /// Strength of top-down predictions on lower layers.
    pub top_down_gain: IValue,
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
            titan_deep_replay_threshold: 1500,
            titan_elastic_window_max: 16,
            titan_decay_rate: 1,
            think_max_ticks: 100,
            think_surprise_threshold_deep: 1500,
            think_surprise_threshold_low: 100,
            default_refractory_ticks: 4,
            weight_clamp_limit: SCALE * 5,
            smbp_decay: 800,
            activity_ema_alpha: 990, // Alpha out of 1000 (0.99)
            target_activity_level: 100,
            homeostatic_scaling_limit: 1024 * 16,
            stp_resource_decay: 800,
            stp_calcium_decay: 95,
            stp_resource_recovery: 99,
            stp_calcium_recovery: 200,
            workspace_broadcast_intensity: SCALE / 2,
            workspace_ignition_threshold: 10.0,
            astro_increment: 50,
            astro_decay_rate: 990, // Alpha out of 1000
            hierarchical_control_enabled: true,
            top_down_gain: SCALE / 4, // 0.25 gain
        }
    }
}

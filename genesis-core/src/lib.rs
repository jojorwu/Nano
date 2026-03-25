use serde::{Serialize, Deserialize};

pub mod bus;
pub mod module;
pub mod model;
pub mod physics;
pub mod config;

#[cfg(feature = "titan")]
pub mod titan;
#[cfg(feature = "text")]
pub mod text;
#[cfg(feature = "vision")]
pub mod vision;
#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "rl")]
pub mod rl;
#[cfg(feature = "fusion")]
pub mod fusion;
pub mod graph;
pub mod robotics;
pub mod plasticity;
pub mod attention;
pub mod workspace;
pub mod hierarchical;
pub mod episodic;

// Re-exports for convenience and compatibility
pub use bus::{InputBus, Modality};
pub use module::{NanoModule, ModuleManager, ModuleRegistry, ModuleError, ModuleInput, ForeignModule, ForeignTickFn};
pub use model::{NeuronsSoA, SynapsesSoA, BakedModel, SpikeData, Compartment, LatentSynapseMatrix, NeuronsFFI, SynapsesFFI};
pub use attention::AttnResModule;
pub use workspace::WorkspaceModule;
pub use hierarchical::HierarchicalModule;
pub use episodic::EpisodicModule;
pub use config::NetworkConfig;

pub type IValue = i32;
pub const SCALE: IValue = 1024; // 2^10 for bit-shift optimizations

// --- Physics Constants (Legacy/Defaults - prefer config fields) ---
pub const DEFAULT_REFRACTORY_TICKS: i32 = 4;
pub const SMBP_DECAY: i64 = 800; // Multiplier out of SCALE
pub const ACTIVITY_EMA_ALPHA: i64 = 99; // Alpha out of 100
pub const TARGET_ACTIVITY_LEVEL: IValue = 100; // 10% target firing rate
pub const WEIGHT_CLAMP_LIMIT: IValue = SCALE * 5;

// --- Dendritic Gating Constants ---
pub const GATE_OPEN: IValue = SCALE;
pub const GATE_HALF: IValue = SCALE / 2;
pub const GATE_QUARTER: IValue = SCALE / 4;
pub const GATE_THREE_QUARTERS: IValue = SCALE * 3 / 4;

/// Represents the global chemical state of the network.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct NeuromodulationState {
    pub dopamine: IValue,       // Reward / Prediction Error
    pub noradrenaline: IValue,  // Surprise / Novelty / Arousal
    pub serotonin: IValue,      // Stability / Risk Mitigation
}

/// Thinking Mode Module: Enables "Chain of Thought" reasoning by performing
/// extra simulation sub-ticks for each external input tick.
/// This allows the network to iterate internally without new modality data.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ThinkModule {
    pub extra_ticks: usize,
    pub active: bool,
    pub max_ticks: usize,
    pub surprise_threshold_deep: IValue,
    pub surprise_threshold_low: IValue,
    /// Active Inference: Generative Replay (Fantasy) mode
    pub fantasy_mode: bool,
}

impl ThinkModule {
    pub fn new(ticks: usize) -> Self {
        Self {
            extra_ticks: ticks,
            active: true,
            max_ticks: 100,
            surprise_threshold_deep: 1500,
            surprise_threshold_low: 100,
            fantasy_mode: false,
        }
    }
}

impl NanoModule for ThinkModule {
    fn name(&self) -> &str { "think" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn inputs(&self) -> Vec<String> { vec!["proximal".to_string()] }
    fn handle_input(&mut self, input: &ModuleInput) {
        if let ModuleInput::Control(name, val) = input {
            if name == "active" { self.active = *val != 0; }
            if name == "ticks" { self.extra_ticks = *val as usize; }
            if name == "fantasy" { self.fantasy_mode = *val != 0; }
        }
    }
    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        // Active Inference: Generative Replay logic.
        // If in fantasy mode, drive proximal inputs using distal predictions from the last tick.
        if self.fantasy_mode {
             let prox = bus.proximal();
             let dist = bus.distal();
             for i in 0..prox.len() {
                  let prediction = dist[i].load(std::sync::atomic::Ordering::Relaxed);
                  if prediction > 100 {
                       InputBus::atomic_saturating_add(&prox[i], prediction / 2);
                  }
             }
        }
    }
    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, surprise: Option<IValue>) {
        if let Some(s) = surprise {
            // Adaptive Thought Depth: increase internal iteration depth during high uncertainty (surprise)
            if s > self.surprise_threshold_deep {
                self.extra_ticks = (self.extra_ticks + 5).min(self.max_ticks); // Deep reasoning
            } else if s > (self.surprise_threshold_deep / 3) {
                self.extra_ticks = (self.extra_ticks + 1).min(self.max_ticks / 2);
            } else if s < self.surprise_threshold_low {
                self.extra_ticks = self.extra_ticks.saturating_sub(1);
            }
        }
    }
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}
    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
    fn validate_state(&self, _neurons: &NeuronsSoA) -> Result<(), ModuleError> { Ok(()) }
    fn on_config_sync(&mut self, config: &NetworkConfig) {
        self.max_ticks = config.think_max_ticks;
        self.surprise_threshold_deep = config.think_surprise_threshold_deep;
        self.surprise_threshold_low = config.think_surprise_threshold_low;
    }
}

/// Context passed to plasticity rules to improve flexibility and reduce argument count.
pub struct PlasticityContext<'a> {
    pub pre_spiked: bool,
    pub post_spiked: bool,
    pub backprop_signal: IValue,
    pub prediction_error: IValue,
    pub compartment: Compartment,
    pub reward: Option<IValue>,
    pub neuromodulation: NeuromodulationState,
    pub pre_last_spike: u32,
    pub post_last_spike: u32,
    pub current_tick: u32,
    pub post_index: usize,
    pub neurons: &'a NeuronsSoA,
    pub config: &'a crate::config::NetworkConfig,
}

impl<'a> PlasticityContext<'a> {
    /// Calculates the combined modulation factor based on chemical state and SMBP.
    pub fn get_modulation_gain(&self) -> i64 {
        let neuromod = SCALE as i64 + self.neuromodulation.noradrenaline as i64;
        let dopamine = SCALE as i64 + self.neuromodulation.dopamine.abs() as i64;
        let smbp = if self.compartment != Compartment::Proximal {
            SCALE as i64 + self.backprop_signal as i64
        } else {
            SCALE as i64
        };
        (neuromod * dopamine * smbp) >> 20
    }

    pub fn is_rewarded(&self) -> bool {
        self.reward.is_some()
    }
}

/// Trait for weight update rules (e.g., GSOP, STDP)
pub trait PlasticityRule {
    fn apply(&self, weight: &mut IValue, ctx: &PlasticityContext);

    /// STC: Tagging phase. Instead of updating weight, we update the tag trace.
    fn tag(&self, tag: &mut IValue, timer: &mut u16, volatility: &mut u8, ctx: &PlasticityContext) {
        // Probabilistic Metaplasticity:
        // Probability of update depends on volatility.
        // If volatility is 0, the synapse is stable and very unlikely to change.
        if *volatility > 0 {
             use rand::Rng;
             let mut rng = rand::thread_rng();
             if rng.gen_range(0..255) <= *volatility {
                  let mut temp_weight = *tag;
                  self.apply(&mut temp_weight, ctx);

                  if temp_weight != *tag {
                       *tag = temp_weight;
                       *timer = 100; // Tag duration: 100 ticks
                       // Every successful update slightly reduces volatility (consolidation)
                       *volatility = volatility.saturating_sub(1);
                  }
             }
        }
    }

    fn update_contrastive(&self, weight: &mut IValue, layer_correlation: IValue) {
        if layer_correlation > 512 {
             *weight = (*weight as i64 * (1024 - (layer_correlation / 10)) as i64 >> 10) as i32;
        }
    }
}

pub struct GsopRule {
    pub learning_rate: IValue,
}

impl PlasticityRule for GsopRule {
    fn apply(&self, weight: &mut IValue, ctx: &PlasticityContext) {
        let mut lr = match ctx.compartment {
            Compartment::Proximal => self.learning_rate,
            Compartment::Distal => self.learning_rate * 8 / 10,
            _ => self.learning_rate / 2,
        };

        // Formalized Predictive Coding Learning
        // If Distal (Prediction), adjust learning based on prediction error
        if ctx.compartment == Compartment::Distal {
             // If prediction error is positive (Reality > Prediction), we need more Distal input (LTP)
             // If prediction error is negative (Prediction > Reality), we need less Distal input (LTD)
             let pc_factor = (ctx.prediction_error as i64 * SCALE as i64) >> 10;
             lr = (lr as i64 * (SCALE as i64 + pc_factor.abs()) >> 10) as i32;
        }

        let p_gate = ctx.neurons.plasticity_gate[ctx.post_index];
        let lr_final = (lr as i64 * ctx.get_modulation_gain() * p_gate as i64) >> 20;
        let lr_final = lr_final as i32;

        let reward_mod = if let Some(r) = ctx.reward { if r < 0 { -1 } else { 1 } } else { 1 };

        let lr_mod = (lr_final * reward_mod) as i32;

        let old_weight = *weight;
        if ctx.pre_spiked {
             if ctx.compartment == Compartment::Distal {
                  // Predictive Coding Update
                  if ctx.prediction_error > 100 { // Under-prediction
                       *weight = weight.saturating_add(lr_mod);
                  } else if ctx.prediction_error < -100 { // Over-prediction
                       *weight = weight.saturating_sub(lr_mod);
                  }
             } else {
                  // Standard Hebbian for Proximal
                  if ctx.post_spiked {
                       *weight = weight.saturating_add(lr_mod);
                  } else {
                       *weight = weight.saturating_sub(lr_mod / 2);
                  }
             }
        }

        crate::plasticity::clamp_and_preserve_sign_with_limit(weight, old_weight, ctx.config.weight_clamp_limit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_neurons_validation() {
        let mut neurons = NeuronsSoA::new(10);
        assert!(neurons.validate().is_ok());

        neurons.potential.push(0); // Break length consistency
        assert!(neurons.validate().is_err());
    }

    #[test]
    fn test_input_bus_parallel_injection() {
        use std::sync::Arc;
        let bus = Arc::new(InputBus::new(100));
        let threads: Vec<_> = (0..10).map(|_| {
            let b = Arc::clone(&bus);
            std::thread::spawn(move || {
                for i in 0..100 {
                    InputBus::atomic_saturating_add(&b.proximal()[i], 10);
                }
            })
        }).collect();

        for t in threads { t.join().unwrap(); }
        for i in 0..100 {
            assert_eq!(bus.proximal()[i].load(Ordering::Relaxed), 100);
        }

        bus.clear();
        assert_eq!(bus.proximal()[0].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_input_bus_clear_mut() {
        let mut bus = InputBus::new(100);
        bus.proximal()[50].store(500, Ordering::Relaxed);
        bus.clear_mut();
        assert_eq!(bus.proximal()[50].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_neurons_init_and_grow() {
        let mut neurons = NeuronsSoA::new(10);
        assert_eq!(neurons.len(), 10);
        assert_eq!(neurons.dendritic_gate[0], SCALE);
        assert_eq!(neurons.layer_id[0], 0);

        neurons.grow(5);
        assert_eq!(neurons.len(), 15);
        assert_eq!(neurons.dendritic_gate[14], SCALE);
        assert_eq!(neurons.potential[14], 0);

        neurons.shrink_to_fit();
        assert_eq!(neurons.len(), 15);
    }
}

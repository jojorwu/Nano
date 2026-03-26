use crate::{IValue, NanoModule, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}, module::{ModuleInput, ModuleError}, config::NetworkConfig};
use serde::{Serialize, Deserialize};

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
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
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
        self.max_ticks = config.modules.think_max_ticks;
        self.surprise_threshold_deep = config.modules.think_surprise_threshold_deep;
        self.surprise_threshold_low = config.modules.think_surprise_threshold_low;
    }
}

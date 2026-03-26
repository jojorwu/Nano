use crate::{IValue, SCALE, NanoModule, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}, config::NetworkConfig};
use serde::{Serialize, Deserialize};

/// Hierarchical Control Module (Deep Predictive Coding)
/// Implements top-down feedback where higher layers (abstraction) drive
/// distal predictions in lower layers (sensory), enabling state-dependent perception.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HierarchicalModule {
    pub enabled: bool,
    pub top_down_gain: IValue,
    /// Maps high layer ID to a list of lower layer IDs it predicts
    pub hierarchy_map: std::collections::HashMap<u16, Vec<u16>>,
    /// Cached average activity per layer
    pub layer_activity: std::collections::HashMap<u16, f32>,
    /// Cache: layer ID to list of neuron indices
    pub layer_to_neurons: std::collections::HashMap<u16, Vec<u32>>,
}

impl HierarchicalModule {
    pub fn new() -> Self {
        let mut hierarchy_map = std::collections::HashMap::new();
        // Default: Layer 2 predicts Layer 1, Layer 3 predicts Layer 2
        hierarchy_map.insert(2, vec![1]);
        hierarchy_map.insert(3, vec![2]);

        Self {
            enabled: true,
            top_down_gain: SCALE / 4,
            hierarchy_map,
            layer_activity: std::collections::HashMap::new(),
            layer_to_neurons: std::collections::HashMap::new(),
        }
    }
}

impl NanoModule for HierarchicalModule {
    fn name(&self) -> &str { "hierarchical" }
    fn tier(&self) -> u32 { 15 } // Runs after workspace but before final integration

    fn on_init(&mut self, neurons: &mut NeuronsSoA) -> Result<(), crate::module::ModuleError> {
        self.layer_to_neurons.clear();
        for (i, &lid) in neurons.layer_id.iter().enumerate() {
            self.layer_to_neurons.entry(lid).or_default().push(i as u32);
        }
        Ok(())
    }

    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        if !self.enabled { return; }

        // Top-Down Modulation: High-level layers drive distal predictions in lower layers.
        // This represents "State-Dependent Perception" or "Mental Imagery".
        for (&high_layer, low_layers) in &self.hierarchy_map {
             let high_act = self.layer_activity.get(&high_layer).cloned().unwrap_or(0.0);

             if high_act > 0.1 {
                  let boost = (high_act * self.top_down_gain as f32) as i32;
                  let dist = bus.distal();

                  for &low_layer in low_layers {
                       if let Some(neurons) = self.layer_to_neurons.get(&low_layer) {
                            for &idx in neurons {
                                 if (idx as usize) < dist.len() {
                                      InputBus::atomic_saturating_add(&dist[idx as usize], boost);
                                 }
                            }
                       }
                  }
             }
        }
    }

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _prev: &[bool], current: &[bool], _tick: u32, _reward: Option<IValue>) {
        // Monitor layer activity
        self.layer_activity.clear();
        let mut counts = std::collections::HashMap::new();
        let mut totals = std::collections::HashMap::new();

        for i in 0..neurons.len() {
            let lid = neurons.layer_id[i];
            *totals.entry(lid).or_insert(0) += 1;
            if current[i] {
                *counts.entry(lid).or_insert(0) += 1;
            }
        }

        for (lid, total) in totals {
            let count = counts.get(&lid).cloned().unwrap_or(0);
            self.layer_activity.insert(lid, count as f32 / total as f32);
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

    fn on_config_sync(&mut self, config: &NetworkConfig) {
        self.enabled = config.modules.hierarchical_control_enabled;
        self.top_down_gain = config.modules.top_down_gain;
    }

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_layer_monitoring() {
        let mut hierarchical = HierarchicalModule::new();
        let mut neurons = NeuronsSoA::new(10);
        for i in 0..10 { neurons.layer_id[i] = if i < 5 { 1 } else { 2 }; }

        let mut spikes = vec![false; 10];
        for i in 5..10 { spikes[i] = true; } // 100% activity in layer 2

        hierarchical.on_update_weights(&mut neurons, &[], &spikes, 0, None);

        let layer_2_act = hierarchical.layer_activity.get(&2).cloned().unwrap_or(0.0);
        assert!((layer_2_act - 1.0).abs() < 0.001);
    }
}

use crate::{IValue, SCALE, NanoModule, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}};
use serde::{Serialize, Deserialize};

/// Global Neuronal Workspace (GNW) Implementation
/// Based on Dehaene's GNW theory: "Competition, Ignition, and Broadcast".
/// This module selects the most active/salient neural block and broadcasts its
/// pattern across the entire network, enabling a form of "conscious" attention.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorkspaceModule {
    /// Currently 'ignited' block ID
    pub active_block: Option<u32>,
    /// Persistence of ignition (ticks remaining)
    pub ignition_timer: u32,
    /// Last activity per block for competition
    pub block_activity: std::collections::HashMap<u32, f32>,
    /// Broadcasting strength
    pub broadcast_intensity: IValue,
    /// Minimum activity to trigger ignition
    pub ignition_threshold: f32,
    /// Neuron to Block mapping
    pub neuron_to_block: Vec<u32>,
}

impl WorkspaceModule {
    pub fn new() -> Self {
        Self {
            active_block: None,
            ignition_timer: 0,
            block_activity: std::collections::HashMap::new(),
            broadcast_intensity: SCALE / 2, // 0.5 intensity
            ignition_threshold: 10.0,
            neuron_to_block: Vec::new(),
        }
    }
}

impl NanoModule for WorkspaceModule {
    fn name(&self) -> &str { "workspace" }
    fn tier(&self) -> u32 { 10 } // High tier: runs after sensory processing
    fn outputs(&self) -> Vec<String> { vec!["broadcast".to_string()] }

    fn on_init(&mut self, neurons: &mut NeuronsSoA) -> Result<(), crate::module::ModuleError> {
        self.neuron_to_block = neurons.block_id.clone();
        Ok(())
    }

    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {

        // 1. Competition: Aggregrate activity per block
        // In a real implementation, we would access block_id from neurons.
        // For efficiency in on_tick, we use a heuristic or assume fixed block size if needed.
        // However, Workspace needs proper block mapping.

        if self.ignition_timer > 0 {
            self.ignition_timer -= 1;

            // 2. Broadcast: Inject active block's pattern (from history/state)
            // For now, we simulate broadcast by boosting specific signals if they belong to workspace
            if let Some(bid) = self.active_block {
                let prox = bus.proximal();
                // Broadcast using the actual mapping stored during on_init
                for (i, &mapped_bid) in self.neuron_to_block.iter().enumerate() {
                    if mapped_bid == bid && i < prox.len() {
                        InputBus::atomic_saturating_add(&prox[i], self.broadcast_intensity);
                    }
                }
            }
        } else {
            self.active_block = None;

            // Find new winner
            let mut winner = None;
            let mut max_act = self.ignition_threshold;

            for (bid, &act) in &self.block_activity {
                if act > max_act {
                    max_act = act;
                    winner = Some(*bid);
                }
            }

            if let Some(bid) = winner {
                log::info!("GNW Ignition: Block {} won competition", bid);
                self.active_block = Some(bid);
                self.ignition_timer = 5; // Sustain for 5 ticks
            }
        }

        // Decay block activity
        for act in self.block_activity.values_mut() {
            *act *= 0.9;
        }
    }

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _previous_spikes: &[bool], current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {
        // Monitor current activity to update competition state
        for i in 0..neurons.len() {
            if current_spikes[i] {
                let bid = neurons.block_id[i];
                let entry = self.block_activity.entry(bid).or_insert(0.0);
                *entry += 1.0;
            }
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        self.block_activity.clear();
    }

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }

    fn on_config_sync(&mut self, config: &crate::config::NetworkConfig) {
        self.broadcast_intensity = config.modules.workspace_broadcast_intensity;
        self.ignition_threshold = config.modules.workspace_ignition_threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NeuronsSoA;
    use crate::bus::InputBus;

    #[test]
    fn test_workspace_ignition() {
        let mut workspace = WorkspaceModule::new();
        let mut neurons = NeuronsSoA::new(32);
        for i in 0..32 { neurons.block_id[i] = (i / 16) as u32; }

        workspace.on_init(&mut neurons).unwrap();
        let bus = InputBus::new(32);

        // Simulate activity in block 1
        let mut spikes = vec![false; 32];
        for i in 16..32 { spikes[i] = true; }

        workspace.on_update_weights(&mut neurons, &[], &spikes, 0, None);

        // Tick to trigger ignition
        workspace.on_tick(&bus, &spikes, 1);

        assert_eq!(workspace.active_block, Some(1));
        assert!(workspace.ignition_timer > 0);
    }
}

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
}

impl WorkspaceModule {
    pub fn new() -> Self {
        Self {
            active_block: None,
            ignition_timer: 0,
            block_activity: std::collections::HashMap::new(),
            broadcast_intensity: SCALE / 2, // 0.5 intensity
            ignition_threshold: 10.0,
        }
    }
}

impl NanoModule for WorkspaceModule {
    fn name(&self) -> &str { "workspace" }
    fn tier(&self) -> u32 { 10 } // High tier: runs after sensory processing
    fn outputs(&self) -> Vec<String> { vec!["broadcast".to_string()] }

    fn on_tick(&mut self, bus: &InputBus, previous_spikes: &[bool], _tick: u32) {
        let n_count = previous_spikes.len();

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
                // Simple broadcast: boost activity of all neurons in the ignited block
                // This creates a self-sustaining loop (re-entrant processing)
                for i in 0..n_count {
                    // Heuristic: map neuron index to block (ideally use block_id from state)
                    if (i / 16) as u32 == bid {
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
        self.broadcast_intensity = config.workspace_broadcast_intensity;
        self.ignition_threshold = config.workspace_ignition_threshold;
    }
}

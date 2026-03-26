use crate::{IValue, NanoModule, NeuronsSoA, SynapsesSoA};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpikingGraphModule {
    pub topology_type: TopologyType,
    pub rewiring_rate: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TopologyType {
    SmallWorld,
    ScaleFree,
    Random,
}

impl SpikingGraphModule {
    pub fn new(topology: TopologyType) -> Self {
        Self {
            topology_type: topology,
            rewiring_rate: 0.01,
        }
    }
}

impl NanoModule for SpikingGraphModule {
    fn name(&self) -> &str { "graph_engine" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn on_tick(&mut self, _bus: &crate::InputBus, _previous_spikes: &[bool], _tick: u32) {
        // Dynamic graph rewiring logic could be implemented here
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}

    fn on_night_phase(&mut self, neurons: &mut NeuronsSoA, synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        // Perform structural graph maintenance
        // Note: Backend must clear synapse_index if structural changes occur.
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let n_count = neurons.len();
        if n_count == 0 { return; }

        if rng.gen::<f32>() < self.rewiring_rate && synapses.len() > 10 {
            // Small-World Rewiring Heuristic: rewire a random synapse
            let idx = rng.gen_range(0..synapses.len());

            // Safety: Constraint new target within valid neuron indices
            let new_target = rng.gen_range(0..n_count) as u32;
            synapses.target_index[idx] = new_target;
        }
    }

    fn box_clone(&self) -> Box<dyn NanoModule> {
        Box::new(self.clone())
    }

    fn get_state(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) {
            *self = new_self;
        }
    }
}

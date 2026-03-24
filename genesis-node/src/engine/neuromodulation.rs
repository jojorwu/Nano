use crate::SimulationEngine;
use genesis_core::SCALE;

pub struct NeuromodulationEngine;

impl NeuromodulationEngine {
    pub fn update(engine: &mut SimulationEngine, surprise: i32, normalized_reward: Option<i32>) {
        engine.state.global_modulators.noradrenaline = surprise;
        if let Some(r) = normalized_reward {
            engine.state.global_modulators.dopamine = r;
        } else {
            engine.state.global_modulators.dopamine = (engine.state.global_modulators.dopamine * 9) / 10;
        }
        // Serotonin tracks long-term stability
        engine.state.global_modulators.serotonin = (engine.state.global_modulators.serotonin * 99 + (SCALE - surprise).max(0)) / 100;
    }
}

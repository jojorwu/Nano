use serde::{Serialize, Deserialize};
use crate::events::{SimulationObserver, SimulationEvent};
use crate::engine::SimulationEngine;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Telemetry {
    pub spike_counts: Vec<usize>,
    pub episode_rewards: Vec<i32>,
}

impl SimulationObserver for Telemetry {
    fn on_event(&mut self, event: &SimulationEvent, _engine: &mut SimulationEngine) {
        match event {
            SimulationEvent::TickComplete { tick: _, spike_count, data: _, execution_time: _ } => {
                self.spike_counts.push(*spike_count);
            }
            SimulationEvent::RewardReceived(reward) => {
                self.episode_rewards.push(*reward);
            }
            _ => {}
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

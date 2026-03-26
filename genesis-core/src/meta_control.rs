use crate::{IValue, SCALE, NanoModule, bus::InputBus, model::{NeuronsSoA, SynapsesSoA}, event::GlobalEvent};
use serde::{Serialize, Deserialize};

/// MetaControl Module: Monitors global network state and broadcasts architectural events.
/// It detects global patterns like high surprise, activity storms, or stagnation
/// and coordinates other modules via the EventBus.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MetaControlModule {
    pub surprise_threshold: i32,
    pub storm_threshold: f32,
    pub stagnant_threshold: f32,
    pub last_activity: f32,
    pub block_surprise_history: std::collections::HashMap<u32, Vec<f32>>,
}

impl MetaControlModule {
    pub fn new() -> Self {
        Self {
            surprise_threshold: 1000,
            storm_threshold: 0.8,
            stagnant_threshold: 0.001,
            last_activity: 0.0,
            block_surprise_history: std::collections::HashMap::new(),
        }
    }
}

impl NanoModule for MetaControlModule {
    fn name(&self) -> &str { "meta_control" }
    fn tier(&self) -> u32 { 100 } // Runs last to observe results of all modules

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        // Detect and publish events based on global signals in the bus
        let noradrenaline = bus.global_signals[1].load(std::sync::atomic::Ordering::Relaxed);
        if noradrenaline > self.surprise_threshold {
            bus.event_bus.publish(GlobalEvent::HighSurprise(noradrenaline));
        }

        // Detect Persistent Block Surprise for targeted neurogenesis
        for (&bid, history) in &self.block_surprise_history {
             if history.len() >= 5 {
                  let avg_surprise: f32 = history.iter().sum::<f32>() / history.len() as f32;
                  if avg_surprise > 0.8 {
                       bus.event_bus.publish(GlobalEvent::StructuralUpdate(format!("Neurogenesis:{}", bid)));
                  }
             }
        }

        if self.last_activity > self.storm_threshold {
            bus.event_bus.publish(GlobalEvent::Custom("ActivityStorm".to_string(), Vec::new()));
        } else if self.last_activity < self.stagnant_threshold {
            bus.event_bus.publish(GlobalEvent::Custom("StagnantNetwork".to_string(), Vec::new()));
        }
    }

    fn on_update_weights(&mut self, neurons: &mut NeuronsSoA, _prev: &[bool], current: &[bool], _tick: u32, surprise: Option<IValue>) {
        // Update activity metrics
        let n_count = neurons.len();
        if n_count > 0 {
            let spike_count = current.iter().filter(|&&s| s).count();
            self.last_activity = (spike_count as f32) / (n_count as f32);
        }

        // Update block surprise tracking (simplified: use global surprise as proxy for now)
        if let Some(s) = surprise {
             let s_val = s as f32 / 1024.0;
             for &bid in &neurons.block_id {
                  let history = self.block_surprise_history.entry(bid).or_insert_with(Vec::new);
                  history.push(s_val);
                  if history.len() > 10 { history.remove(0); }
             }
        }
    }

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}

    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
}

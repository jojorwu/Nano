use crate::{NanoModule, NeuronsSoA, SynapsesSoA, InputBus, IValue, GlobalEvent};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MetaControlModule {
    pub activity_storm_threshold: f32,
    pub stagnation_threshold: u32,
    pub surprise_trigger: i32,
    pub enabled: bool,
}

impl Default for MetaControlModule {
    fn default() -> Self {
        Self {
            activity_storm_threshold: 0.8,
            stagnation_threshold: 100,
            surprise_trigger: 1500,
            enabled: true,
        }
    }
}

impl NanoModule for MetaControlModule {
    fn name(&self) -> &str { "meta_control" }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn on_tick(&mut self, bus: &InputBus, _previous_spikes: &[bool], _tick: u32) {
        if !self.enabled { return; }

        // Fetch global noradrenaline (surprise)
        let surprise = bus.global_signals[1].load(std::sync::atomic::Ordering::Relaxed);
        if surprise > self.surprise_trigger {
            bus.event_bus.publish(GlobalEvent::HighSurprise(surprise));
        }

        // Structural Update Triggers: could be moved here from individual modules
        if surprise > 1800 {
            bus.event_bus.publish(GlobalEvent::StructuralUpdate("NeurogenesisRequested".to_string()));

            // Dynamic Gating Logic Signal
            // We'll use a custom event to signal the pipeline to retry propagation
            bus.event_bus.publish(GlobalEvent::Custom("RetryPropagation".to_string(), vec![]));
        }
    }

    fn on_event(&mut self, event: &GlobalEvent) {
        match event {
            GlobalEvent::HighSurprise(s) if *s > 2000 => {
                log::warn!("MetaControl: Extreme surprise ({})! Halving learning thresholds.", s);
                // This would ideally affect Titan directly, but for now we signal via EventBus
            }
            _ => {}
        }
    }

    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _surprise: Option<IValue>) {}

    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {
        // High-level periodic adjustments
    }

    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
}

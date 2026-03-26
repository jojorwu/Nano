use crate::SimulationEngine;
use genesis_core::SCALE;

pub struct NeuromodulationEngine;

impl NeuromodulationEngine {
    pub fn update(engine: &mut SimulationEngine, surprise: i32, normalized_reward: Option<i32>) {
        // 1. Process Event-Driven Modulation
        let events = engine.input_bus.event_bus.poll_all();
        for event in events {
             match event {
                 genesis_core::event::GlobalEvent::ModulationShift(sig_id, delta) => {
                     match sig_id {
                         0 => engine.state.global_modulators.dopamine = engine.state.global_modulators.dopamine.saturating_add(delta),
                         1 => engine.state.global_modulators.noradrenaline = engine.state.global_modulators.noradrenaline.saturating_add(delta),
                         2 => engine.state.global_modulators.serotonin = engine.state.global_modulators.serotonin.saturating_add(delta),
                         _ => {}
                     }
                 }
                 genesis_core::event::GlobalEvent::RewardSignal(r) => {
                      engine.state.global_modulators.dopamine = engine.state.global_modulators.dopamine.saturating_add(r);
                 }
                 _ => {}
             }
             // Re-publish events for other modules to consume (they poll from InputBus too)
             engine.input_bus.event_bus.publish(event);
        }

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

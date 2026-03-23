use genesis_core::BakedModel;

use crate::events::{SimulationObserver, SimulationEvent};
use crate::engine::SimulationEngine;

pub struct Observer {
    pub max_spikes_per_tick: usize,
    pub energy_budget_per_tick: u64,
    pub current_energy_usage: u64,
    pub total_energy_consumed: u64,
}

impl Observer {
    pub fn new(n_count: usize) -> Self {
        Self {
            max_spikes_per_tick: n_count / 2,
            energy_budget_per_tick: (n_count / 10) as u64,
            current_energy_usage: 0,
            total_energy_consumed: 0
        }
    }

    pub fn process_spikes(&mut self, spikes: &mut [bool], model: &mut BakedModel) {
        let spike_count = spikes.iter().filter(|&&s| s).count();
        self.current_energy_usage = spike_count as u64;

        // 1. Critical Path: Activity Capping (Synchronous Safety)
        if spike_count > self.max_spikes_per_tick {
            for i in 0..spikes.len() { spikes[i] = false; }
            log::error!("Activity cap triggered! {} spikes inhibited.", spike_count);
        }

        // 2. Control Path: Graceful Degradation (Sync threshold update)
        if self.current_energy_usage > self.energy_budget_per_tick {
            let overload = (self.current_energy_usage - self.energy_budget_per_tick) as i32;
            let increment = (overload / 10).max(1);
            for t in model.neurons.threshold.iter_mut() {
                *t = t.saturating_add(increment);
            }
        }

        // 3. Telemetry Path: (Could be async, keeping count for now)
        self.total_energy_consumed += spike_count as u64;
    }
}

impl SimulationObserver for Observer {
    fn on_event(&mut self, event: &SimulationEvent, _engine: &mut SimulationEngine) {
        // Observer mostly works in process_spikes for sync safety,
        // but can use events for async logging or cleanup.
        if let SimulationEvent::NightPhaseStarted(tick) = event {
            log::info!("Night phase started at tick {}", tick);
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

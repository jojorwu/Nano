use genesis_core::BakedModel;

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
        let spike_count = spikes.iter().filter(|&&s| s).count() as u64;
        self.current_energy_usage = spike_count; // Simplified: 1 spike = 1 energy unit

        if self.current_energy_usage > self.energy_budget_per_tick {
            // Graceful Degradation: Increase thresholds globally if budget is exceeded
            let overload = (self.current_energy_usage - self.energy_budget_per_tick) as i32;
            let increment = (overload / 10).max(1);
            for t in model.neurons.threshold.iter_mut() {
                *t = t.saturating_add(increment);
            }
            log::warn!("Energy budget exceeded. Applying global threshold increment: {}", increment);
        }

        if spike_count as usize > self.max_spikes_per_tick {
            // Activity capping: Spike Storm Protection (still useful as safety)
            for i in 0..spikes.len() { spikes[i] = false; }
        }
        self.total_energy_consumed += spike_count;
    }
}

use genesis_core::{BakedModel, ModuleManager, SpikeData};
use genesis_core::model::SimulationState;
use genesis_compute::ComputeBackend;
use crate::SimulationSettings;
use std::sync::{Arc, Mutex};

pub mod pipeline;
pub mod neuromodulation;

pub struct SimulationEngine {
    pub model: BakedModel,
    pub state: SimulationState,
    pub modules: ModuleManager,
    pub backend: Box<dyn ComputeBackend + Send + Sync>,
    pub input_bus: genesis_core::InputBus,
    pub remote_spike_queue: Arc<Mutex<Vec<usize>>>,
}

impl SimulationEngine {
    pub fn new(model: BakedModel, modules: ModuleManager, backend: Box<dyn ComputeBackend + Send + Sync>, settings: &SimulationSettings) -> Self {
        let n_count = model.neurons.len();
        Self {
            model,
            state: SimulationState::new(n_count, settings.night_phase_interval as usize),
            modules,
            backend,
            input_bus: genesis_core::InputBus::new(n_count),
            remote_spike_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn reset_potential_buffers(&mut self) {
        self.model.neurons.proximal_potential.fill(0);
        self.model.neurons.distal_potential.fill(0);
        self.model.neurons.apical_potential.fill(0);
        self.model.neurons.basal_potential.fill(0);
    }

    pub fn reconstruct_history(&self, window: usize) -> Vec<Vec<bool>> {
        let n_count = self.model.neurons.len();
        let hist_len = self.state.spikes_history.len();

        (0..window.min(hist_len)).map(|i| {
            let idx = (self.state.history_ptr + hist_len - 1 - i) % hist_len;
            let data = &self.state.spikes_history[idx];
            let mut vec = vec![false; n_count];
            match data {
                SpikeData::Sparse(indices) => { for &idx in indices { if idx < n_count { vec[idx] = true; } } }
                SpikeData::Dense(mask) => {
                    for i in 0..n_count { if (mask[i / 8] >> (i % 8)) & 1 == 1 { vec[i] = true; } }
                }
                _ => {}
            }
            vec
        }).collect()
    }

    pub fn update_previous_spikes(&mut self, indices: &[usize]) {
        let n_count = self.model.neurons.len();
        let vec = &mut self.state.previous_spikes;
        vec.fill(false);
        for &idx in indices { if idx < n_count { vec[idx] = true; } }
    }

    pub fn finalize_potentials_from_bus(&mut self) {
        use rayon::prelude::*;
        use std::sync::atomic::Ordering;
        self.model.neurons.proximal_potential.par_iter_mut()
            .zip(self.input_bus.proximal())
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
        self.model.neurons.distal_potential.par_iter_mut()
            .zip(self.input_bus.distal())
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
        self.model.neurons.apical_potential.par_iter_mut()
            .zip(self.input_bus.apical())
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
        self.model.neurons.basal_potential.par_iter_mut()
            .zip(self.input_bus.basal())
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
    }

}

use genesis_core::{BakedModel, ModuleManager, SpikeData};
use genesis_core::model::SimulationState;
use genesis_compute::ComputeBackend;
use crate::SimulationSettings;
use std::sync::{Arc, Mutex};

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
            .zip(&self.input_bus.proximal)
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
        self.model.neurons.distal_potential.par_iter_mut()
            .zip(&self.input_bus.distal)
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
        self.model.neurons.apical_potential.par_iter_mut()
            .zip(&self.input_bus.apical)
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
        self.model.neurons.basal_potential.par_iter_mut()
            .zip(&self.input_bus.basal)
            .for_each(|(p, b)| *p = p.saturating_add(b.load(Ordering::Relaxed)));
    }

    pub fn execute_thinking_cycles(&mut self, initial_spike_data: SpikeData) -> SpikeData {
        let n_count = self.model.neurons.len();
        let mut think_ticks = 0;
        for m in &self.modules.modules {
            if m.name() == "think" {
                if let Some(tm) = m.as_any().downcast_ref::<genesis_core::ThinkModule>() {
                    if tm.active { think_ticks = tm.extra_ticks; }
                } else if let Ok(state) = bincode::deserialize::<genesis_core::ThinkModule>(&m.get_state()) {
                     if state.active { think_ticks = state.extra_ticks; }
                }
                break;
            }
        }

        if think_ticks > 0 {
            let mut current_dense = vec![false; n_count];
            match &initial_spike_data {
                SpikeData::Sparse(indices) => { for &i in indices { if i < n_count { current_dense[i] = true; } } }
                SpikeData::Dense(mask) => { for i in 0..n_count { if (mask[i/8] >> (i%8)) & 1 == 1 { current_dense[i] = true; } } }
                _ => {}
            }
            self.backend.think_cycles(&mut self.model, &current_dense, think_ticks, self.state.tick_counter, self.state.global_modulators)
        } else {
            initial_spike_data
        }
    }

    pub fn calculate_surprise(&mut self, current_spike_count: usize) -> i32 {
        let current = current_spike_count as f32;
        let surprise = (current - self.state.rolling_spike_count).abs();
        self.state.rolling_spike_count = self.state.rolling_spike_count * 0.9 + current * 0.1;
        let scaled_surprise = (surprise * 1024.0 / (self.state.rolling_spike_count + 1.0)) as i32;
        scaled_surprise.min(2048)
    }

    pub fn prepare_merged_inputs(&mut self, external_inputs: &[i32]) {
        let n_count = self.model.neurons.len();
        self.state.merged_inputs_buffer.fill(0);
        let merge_len = external_inputs.len().min(n_count);
        self.state.merged_inputs_buffer[..merge_len].copy_from_slice(&external_inputs[..merge_len]);

        let mut remote_spikes = self.remote_spike_queue.lock().unwrap();
        for &idx in remote_spikes.iter() {
            if idx < n_count {
                self.state.merged_inputs_buffer[idx] = self.state.merged_inputs_buffer[idx].saturating_add(genesis_core::SCALE);
            }
        }
        remote_spikes.clear();
    }
}

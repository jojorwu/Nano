use genesis_core::{BakedModel, IValue, SpikeData, NeuromodulationState};
use thiserror::Error;

pub mod cpu;
pub mod kernels;
#[cfg(feature = "wgpu")]
pub mod wgpu;
#[cfg(feature = "cpp")]
pub mod cpp_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("WGPU Adapter not found")]
    AdapterNotFound,
    #[error("Failed to create WGPU Device: {0}")]
    DeviceCreationFailed(String),
}

/// Defines the interface for simulation execution and learning logic.
/// Backends can be optimized for different hardware (CPU, WGPU, etc.) while
/// maintaining numerical parity through standardized bit-shift physics.
/// Represents a composable simulation step.
pub enum SimulationKernel {
    PropagateSynapses,
    UpdateMembranePotentials,
    GenerateSpikes,
    ApplyModulation(NeuromodulationState),
}

pub trait ComputeBackend {
    /// Executes a specific simulation kernel. This allows the runtime to orchestrate
    /// the simulation steps more flexibly.
    fn execute_kernel(&mut self, kernel: SimulationKernel, model: &mut BakedModel, context: &KernelContext) -> Option<SpikeData>;

    /// Executes the main simulation kernels for a single tick:
    /// spike propagation, multi-compartment potential integration, and spike generation.
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState) -> SpikeData;

    /// Executes multiple internal "Thinking" cycles (Chain-of-Thought) without external inputs.
    /// Returns the final spike state after all cycles.
    fn think_cycles(&mut self, model: &mut BakedModel, initial_spikes: &[bool], cycles: usize, current_tick: u32, modulation: NeuromodulationState) -> SpikeData {
        let mut current_spikes = initial_spikes.to_vec();
        let n_count = model.neurons.len();
        let empty_inputs = vec![0; n_count];

        for _ in 0..cycles {
            // Reset compartment potentials between sub-ticks
            model.neurons.proximal_potential.fill(0);
            model.neurons.distal_potential.fill(0);
            model.neurons.apical_potential.fill(0);
            model.neurons.basal_potential.fill(0);

            let spike_data = self.day_phase(model, &empty_inputs, &current_spikes, &[], current_tick, modulation);
            let mut next_spikes = vec![false; n_count];
            match spike_data {
                SpikeData::Sparse(indices) => { for i in indices { if i < n_count { next_spikes[i] = true; } } }
                SpikeData::Dense(mask) => { for i in 0..n_count { if (mask[i/8] >> (i%8)) & 1 == 1 { next_spikes[i] = true; } } }
                _ => {}
            }
            if next_spikes == current_spikes { break; }
            current_spikes = next_spikes;
        }

        let active_indices: Vec<usize> = current_spikes.iter().enumerate()
            .filter(|&(_, &s)| s).map(|(i, _)| i).collect();
        SpikeData::Sparse(active_indices)
    }

    /// Standard weight update rule (typically GSOP or STDP).
    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]);

    /// Weight update with neuromodulatory context (Dopamine/Noradrenaline signals).
    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: NeuromodulationState, history: &[Vec<bool>]);

    /// Performs large-scale structural changes (pruning, neurogenesis, and SNNaS optimization).
    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]);

    /// Synchronizes device-resident state (GPU) back to the CPU model buffers.
    fn sync_state(&mut self, _model: &mut BakedModel) {}

    /// Returns the display name of the backend.
    fn name(&self) -> &'static str;
}

pub struct KernelContext<'a> {
    pub external_inputs: &'a [i32],
    pub previous_spikes: &'a [bool],
    pub history: &'a [Vec<bool>],
    pub current_tick: u32,
    pub modulation: NeuromodulationState,
}

pub use cpu::CpuBackend;
#[cfg(feature = "wgpu")]
pub use wgpu::WgpuBackend;
#[cfg(feature = "cpp")]
pub use cpp_backend::CppBackend;
#[cfg(feature = "cuda")]
pub use cuda_backend::CudaBackend;

pub struct BackendRegistry {
    pub backends: std::collections::HashMap<String, Box<dyn Fn() -> Option<Box<dyn ComputeBackend + Send + Sync>>>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        let mut registry = Self { backends: std::collections::HashMap::new() };
        registry.register("cpu", || Some(Box::new(CpuBackend::default())));
        #[cfg(feature = "wgpu")]
        registry.register("wgpu", || WgpuBackend::new().ok().map(|b| Box::new(b) as Box<dyn ComputeBackend + Send + Sync>));
        #[cfg(feature = "cpp")]
        registry.register("cpp", || Some(Box::new(CppBackend::default())));
        #[cfg(feature = "cuda")]
        registry.register("cuda", || Some(Box::new(CudaBackend::default())));
        registry
    }

    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Option<Box<dyn ComputeBackend + Send + Sync>> + 'static,
    {
        self.backends.insert(name.to_string(), Box::new(factory));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn ComputeBackend + Send + Sync>> {
        self.backends.get(name).and_then(|f| f())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA};

    #[test]
    fn test_membrane_potential_calc() {
        use crate::kernels::calculate_membrane_potential;

        // Base case: minimal decay (1) applied
        let pot = calculate_membrane_potential(500, 0, 0, 0, 0, 512, 0, 0, 0, 0);
        assert!(pot <= 500 && pot >= 499);

        // Gated Distal Input: proximal (600) > threshold (512)
        let pot = calculate_membrane_potential(0, 600, 100, 0, 0, 512, 0, 0, 0, 0);
        // Calculation: 600 + (100 * 600 >> 10) = 658. Using 650 for tolerance.
        assert!(pot >= 650);

        // Blocked Distal Input: proximal (100) < threshold (512)
        let pot = calculate_membrane_potential(0, 100, 100, 0, 0, 512, 0, 0, 0, 0);
        // Calculation: 100 + (100 * 100 >> 10) = 109.
        assert!(pot >= 100 && pot <= 115);
    }

    #[test]
    fn test_sigmoidal_gating_curve() {
        use crate::kernels::calculate_membrane_potential;
        let thresh = 512;

        // Far below threshold
        let pot_low = calculate_membrane_potential(0, 100, 1000, 0, 0, thresh, 0, 0, 0, 0);
        // total = proximal (100) + distal_gated (97) = 197.
        assert!(pot_low < 200);

        // Near threshold
        let pot_near = calculate_membrane_potential(0, 512, 1000, 0, 0, thresh, 0, 0, 0, 0);
        assert!(pot_near >= 500 && pot_near < 1024);

        // Far above threshold
        let pot_high = calculate_membrane_potential(0, 1500, 1000, 0, 0, thresh, 0, 0, 0, 0);
        assert!(pot_high >= 1000);
    }

    #[test]
    fn test_backend_parity() {
        let model = BakedModel {
            version: "4.2".to_string(),
            config: genesis_core::NetworkConfig { learning_rate: 100, ..Default::default() },
            node_id: 0,
            local_range: (0, 2),
            neurons: {
                let mut n = NeuronsSoA::new(2);
                n.last_spike_tick[0] = 5;
                n.last_spike_tick[1] = 8;
                n
            },
            synapses: {
                let mut s = SynapsesSoA::with_capacity(1);
                s.push(0, 1, 1000);
                s
            },
            module_states: std::collections::HashMap::new(),
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            has_vision: false,
            has_audio: false,
            has_robotics: false,
            has_fusion: false,
            vocabulary: std::collections::HashMap::new(),
        };

        let mut cpu = CpuBackend::default();
        let mut model_cpu = model.clone();
        cpu.update_weights_modulated(&mut model_cpu, &[true, false], &[false, true], 10, None, Default::default(), &[]);

        #[cfg(feature = "wgpu")]
        {
            match crate::wgpu::WgpuBackend::new() {
                Ok(mut wgpu) => {
                    let mut model_gpu = model.clone();
                    wgpu.update_weights_modulated(&mut model_gpu, &[true, false], &[false, true], 10, None, Default::default(), &[]);
                    wgpu.sync_state(&mut model_gpu);
                    assert_eq!(model_cpu.synapses.weight[0], model_gpu.synapses.weight[0], "CPU/GPU Weight parity failed");
                }
                Err(e) => {
                    log::warn!("WGPU Backend not available for parity test: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_cpu_dendritic_gating() {
        let mut model = BakedModel {
            version: "4.0".to_string(),
            config: genesis_core::NetworkConfig::default(),
            node_id: 0,
            local_range: (0, 2),
            neurons: NeuronsSoA::new(2),
            synapses: {
                let mut s = SynapsesSoA::with_capacity(1);
                s.push(0, 1, 2000); // High weight
                s
            },
            module_states: std::collections::HashMap::new(),
            #[cfg(feature = "titan")]
            titan_memory: None,
            has_text: false,
            has_vision: false,
            has_audio: false,
            has_robotics: false,
            has_fusion: false,
            vocabulary: std::collections::HashMap::new(),
        };

        let mut backend = CpuBackend::default();
        let prev_spikes = vec![true, false];

        // 1. Open gate (1.0)
        model.neurons.dendritic_gate[1] = 1024;
        let spike_data = backend.day_phase(&mut model, &[0, 0], &prev_spikes, &[], 1, Default::default());
        let mut spikes = vec![false; 2];
        if let SpikeData::Sparse(indices) = spike_data {
            for i in indices { if i < 2 { spikes[i] = true; } }
        }
        assert!(spikes[1]); // Should spike

        // 2. Closed gate (0.0)
        model.neurons.potential[1] = 0;
        model.neurons.dendritic_gate[1] = 0;
        let spike_data = backend.day_phase(&mut model, &[0, 0], &prev_spikes, &[], 2, Default::default());
        let mut spikes = vec![false; 2];
        if let SpikeData::Sparse(indices) = spike_data {
            for i in indices { if i < 2 { spikes[i] = true; } }
        }
        assert!(!spikes[1]); // Should not spike
    }
}

use genesis_core::{BakedModel, IValue, SpikeData, NeuromodulationState, NeuronsFFI};
use crate::{ComputeBackend, SimulationKernel, KernelContext};

// Opaque type for CUDA stream
#[repr(C)]
pub struct CudaStream(std::ffi::c_void);

extern "C" {
    fn cuda_propagate_spikes(
        source_indices: *const u32,
        target_indices: *const u32,
        weights: *const IValue,
        compartments: *const u8,
        synapse_count: u32,
        previous_spikes: *const bool,
        neurons: NeuronsFFI,
        stream: *mut CudaStream
    );

    fn cuda_update_neurons(
        neurons: NeuronsFFI,
        current_tick: u32,
        new_spikes: *mut bool,
        ip_inc: i32,
        ip_dec: i32,
        stream: *mut CudaStream
    );
}

pub struct CudaBackend {
    pub name: &'static str,
    stream: *mut CudaStream,
}

// SAFETY: CudaStream is an opaque pointer that can be moved between threads in typical CUDA usage
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl Default for CudaBackend {
    fn default() -> Self {
        Self {
            name: "CudaBackend",
            stream: std::ptr::null_mut(), // In real implementation, create a stream here
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &'static str { self.name }

    fn execute_kernel(&mut self, kernel: SimulationKernel, model: &mut BakedModel, ctx: &KernelContext) -> Option<SpikeData> {
        match kernel {
            SimulationKernel::PropagateSynapses => {
                unsafe {
                    cuda_propagate_spikes(
                        model.synapses.source_index.as_ptr(),
                        model.synapses.target_index.as_ptr(),
                        model.synapses.weight.as_ptr(),
                        model.synapses.compartment.as_ptr() as *const u8,
                        model.synapses.len() as u32,
                        ctx.previous_spikes.as_ptr(),
                        model.neurons.as_ffi(),
                        self.stream
                    );
                }
                None
            }
            SimulationKernel::GenerateSpikes => {
                let n_count = model.neurons.len();
                let mut new_spikes = vec![false; n_count];
                unsafe {
                    cuda_update_neurons(
                        model.neurons.as_ffi(),
                        ctx.current_tick,
                        new_spikes.as_mut_ptr(),
                        model.config.ip_increment,
                        model.config.ip_decay,
                        self.stream
                    );
                }
                let active_indices: Vec<usize> = new_spikes.iter().enumerate()
                    .filter(|&(_, &s)| s).map(|(i, _)| i).collect();
                Some(SpikeData::Sparse(active_indices))
            }
            _ => None,
        }
    }

    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState) -> SpikeData {
        let ctx = KernelContext { external_inputs, previous_spikes, history, current_tick, modulation };
        self.execute_kernel(SimulationKernel::PropagateSynapses, model, &ctx);
        self.execute_kernel(SimulationKernel::GenerateSpikes, model, &ctx).unwrap_or_default()
    }

    fn update_weights(&mut self, _model: &mut BakedModel, _previous_spikes: &[bool], _current_spikes: &[bool], _current_tick: u32, _reward: Option<IValue>, _history: &[Vec<bool>]) {
        // GPU learning logic would go here
    }

    fn update_weights_modulated(&mut self, _model: &mut BakedModel, _previous_spikes: &[bool], _current_spikes: &[bool], _current_tick: u32, _reward: Option<IValue>, _modulation: NeuromodulationState, _history: &[Vec<bool>]) {
        // GPU learning logic would go here
    }

    fn structural_plasticity(&mut self, _model: &mut BakedModel, _reward: Option<IValue>, _history: &[Vec<bool>]) {
        // GPU pruning/growth would go here
    }
}

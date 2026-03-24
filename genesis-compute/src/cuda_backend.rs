use genesis_core::{BakedModel, IValue, SpikeData, NeuromodulationState, NeuronsFFI, SynapsesFFI};
use crate::{ComputeBackend, SimulationKernel, KernelContext};

// Opaque type for CUDA stream
#[repr(C)]
pub struct CudaStream(std::ffi::c_void);

extern "C" {
    fn cuda_propagate_spikes(
        synapses: SynapsesFFI,
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

    fn cuda_update_weights_gsop(
        synapses: SynapsesFFI,
        previous_spikes: *const bool,
        current_spikes: *const bool,
        neurons: NeuronsFFI,
        learning_rate: i32,
        reward: i32,
        stream: *mut CudaStream
    );
}

pub struct CudaBackend {
    pub name: &'static str,
    stream: *mut CudaStream,
    pub synapse_offsets: Vec<u32>,
    pub synapse_indices_flat: Vec<usize>,

    // GPU side mirrors for persistent data
    d_potential: *mut IValue,
    d_proximal: *mut IValue,
    d_distal: *mut IValue,
    d_apical: *mut IValue,
    d_basal: *mut IValue,
    d_threshold: *mut IValue,
    d_decay: *mut IValue,
    d_dendritic_gate: *mut IValue,
    d_gate_threshold: *mut IValue,
    d_adaptation: *mut IValue,
    d_refr: *mut i32,
    d_last_spike: *mut u32,
    d_backprop: *mut IValue,
    d_activity_ema: *mut IValue,
    d_base_threshold: *mut IValue,
    d_liquid: *mut IValue,
    d_excitatory: *mut u8,

    d_syn_src: *mut u32,
    d_syn_tgt: *mut u32,
    d_syn_weight: *mut IValue,
    d_syn_comp: *mut u8,
    d_syn_offsets: *mut u32,
    d_syn_indices: *mut usize,

    d_prev_spikes: *mut bool,
    d_new_spikes: *mut bool,

    n_count: usize,
    s_count: usize,
}

// SAFETY: CudaStream is an opaque pointer that can be moved between threads in typical CUDA usage
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl Default for CudaBackend {
    fn default() -> Self {
        Self {
            name: "CudaBackend",
            stream: std::ptr::null_mut(),
            synapse_offsets: Vec::new(),
            synapse_indices_flat: Vec::new(),
            d_potential: std::ptr::null_mut(),
            d_proximal: std::ptr::null_mut(),
            d_distal: std::ptr::null_mut(),
            d_apical: std::ptr::null_mut(),
            d_basal: std::ptr::null_mut(),
            d_threshold: std::ptr::null_mut(),
            d_decay: std::ptr::null_mut(),
            d_dendritic_gate: std::ptr::null_mut(),
            d_gate_threshold: std::ptr::null_mut(),
            d_adaptation: std::ptr::null_mut(),
            d_refr: std::ptr::null_mut(),
            d_last_spike: std::ptr::null_mut(),
            d_backprop: std::ptr::null_mut(),
            d_activity_ema: std::ptr::null_mut(),
            d_base_threshold: std::ptr::null_mut(),
            d_liquid: std::ptr::null_mut(),
            d_excitatory: std::ptr::null_mut(),
            d_syn_src: std::ptr::null_mut(),
            d_syn_tgt: std::ptr::null_mut(),
            d_syn_weight: std::ptr::null_mut(),
            d_syn_comp: std::ptr::null_mut(),
            d_syn_offsets: std::ptr::null_mut(),
            d_syn_indices: std::ptr::null_mut(),
            d_prev_spikes: std::ptr::null_mut(),
            d_new_spikes: std::ptr::null_mut(),
            n_count: 0,
            s_count: 0,
        }
    }
}

impl CudaBackend {
    fn allocate_gpu_memory(&mut self, model: &BakedModel) {
        // Implementation would use cudaMalloc via FFI
        self.n_count = model.neurons.len();
        self.s_count = model.synapses.len();
    }

    fn upload_to_gpu(&mut self, model: &BakedModel) {
        // Implementation would use cudaMemcpy via FFI
    }

    fn download_from_gpu(&mut self, model: &mut BakedModel) {
        // Implementation would use cudaMemcpy via FFI
    }

    fn rebuild_index_internal(&mut self, model: &BakedModel) {
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();

        let mut forward_counts = vec![0u32; n_count * 16];
        for (&src, &delay) in model.synapses.source_index.iter().zip(&model.synapses.delay) {
            let src = src as usize;
            if src < n_count {
                let d_idx = (delay.clamp(1, 16) - 1) as usize;
                forward_counts[src * 16 + d_idx] += 1;
            }
        }

        self.synapse_offsets = vec![0u32; n_count * 16 + 1];
        for i in 0..(n_count * 16) {
            self.synapse_offsets[i + 1] = self.synapse_offsets[i] + forward_counts[i];
        }

        self.synapse_indices_flat = vec![0; s_count];
        let mut current_forward_offsets = self.synapse_offsets.clone();
        for (i, (&src, &delay)) in model.synapses.source_index.iter().zip(&model.synapses.delay).enumerate() {
            let src = src as usize;
            if src < n_count {
                let d_idx = (delay.clamp(1, 16) - 1) as usize;
                let pos = &mut current_forward_offsets[src * 16 + d_idx];
                self.synapse_indices_flat[*pos as usize] = i;
                *pos += 1;
            }
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &'static str { self.name }

    fn rebuild_index(&mut self, model: &BakedModel) {
        self.rebuild_index_internal(model);
    }

    fn execute_kernel(&mut self, kernel: SimulationKernel, model: &mut BakedModel, ctx: &KernelContext) -> Option<SpikeData> {
        match kernel {
            SimulationKernel::PropagateSynapses => {
                if self.synapse_offsets.is_empty() {
                    self.rebuild_index_internal(model);
                }
                unsafe {
                    cuda_propagate_spikes(
                        model.synapses.as_ffi(self.synapse_offsets.as_ptr(), self.synapse_indices_flat.as_ptr()),
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

    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], _current_tick: u32, reward: Option<IValue>, _history: &[Vec<bool>]) {
        let r = reward.unwrap_or(0);
        unsafe {
            cuda_update_weights_gsop(
                model.synapses.as_ffi(self.synapse_offsets.as_ptr(), self.synapse_indices_flat.as_ptr()),
                previous_spikes.as_ptr(),
                current_spikes.as_ptr(),
                model.neurons.as_ffi(),
                model.config.learning_rate,
                r,
                self.stream
            );
        }
    }

    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], tick: u32, reward: Option<IValue>, _modulation: NeuromodulationState, history: &[Vec<bool>]) {
        self.update_weights(model, previous_spikes, current_spikes, tick, reward, history);
    }

    fn structural_plasticity(&mut self, _model: &mut BakedModel, _reward: Option<IValue>, _history: &[Vec<bool>]) {
        // GPU pruning/growth would go here
    }
}

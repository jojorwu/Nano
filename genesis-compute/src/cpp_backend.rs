use genesis_core::{BakedModel, IValue, SpikeData, NeuromodulationState, NeuronsFFI, SynapsesFFI};
use crate::{ComputeBackend, SimulationKernel, KernelContext};

extern "C" {
    fn cpp_propagate_spikes(
        synapses: SynapsesFFI,
        previous_spikes: *const bool,
        neurons: NeuronsFFI
    );

    fn cpp_recover_stp(synapses: SynapsesFFI);

    fn cpp_update_neurons(
        neurons: NeuronsFFI,
        current_tick: u32,
        new_spikes: *mut u8,
        ip_inc: i32,
        ip_dec: i32,
        noise_amp: i32
    );
}

pub struct CppBackend {
    pub name: &'static str,
    pub synapse_offsets: Vec<u32>,
    pub synapse_indices_flat: Vec<usize>,
}

impl Default for CppBackend {
    fn default() -> Self {
        Self {
            name: "CppBackend",
            synapse_offsets: Vec::new(),
            synapse_indices_flat: Vec::new(),
        }
    }
}

impl CppBackend {
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

impl ComputeBackend for CppBackend {
    fn name(&self) -> &'static str { self.name }

    fn rebuild_index(&mut self, model: &BakedModel) {
        self.rebuild_index_internal(model);
    }

    fn execute_kernel(&mut self, kernel: SimulationKernel, model: &mut BakedModel, ctx: &KernelContext) -> Option<SpikeData> {
        match kernel {
            SimulationKernel::PropagateSynapses => {
                let n_count = model.neurons.len();
                if self.synapse_offsets.is_empty() {
                    self.rebuild_index_internal(model);
                }

                // Apply external inputs directly to proximal potential with dendritic gating
                for (i, &val) in ctx.external_inputs.iter().enumerate() {
                    if i < n_count {
                        let gate = model.neurons.dendritic_gate[i];
                        let gated_val = ((val as i64 * gate as i64) >> 10) as i32;
                        model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(gated_val);
                    }
                }
                if let Some(td) = ctx.top_down_modulation {
                    for (i, &boost) in td.iter().enumerate() {
                        if i < n_count {
                            model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(boost);
                        }
                    }
                }

                unsafe {
                    let syn_ffi = model.synapses.as_ffi(self.synapse_offsets.as_ptr(), self.synapse_indices_flat.as_ptr());
                    cpp_propagate_spikes(
                        syn_ffi,
                        ctx.previous_spikes.as_ptr(),
                        model.neurons.as_ffi()
                    );
                    cpp_recover_stp(syn_ffi);
                }
                None
            }
            SimulationKernel::UpdateMembranePotentials => {
                // Potential updates are handled within GenerateSpikes for this backend
                None
            }
            SimulationKernel::GenerateSpikes => {
                let n_count = model.neurons.len();
                let mut new_spikes = vec![0u8; n_count];

                unsafe {
                    cpp_update_neurons(
                        model.neurons.as_ffi(),
                        ctx.current_tick,
                        new_spikes.as_mut_ptr(),
                        model.config.plasticity.ip_increment,
                        model.config.plasticity.ip_decay,
                        model.config.physics.noise_amplitude
                    );
                }

                let active_indices: Vec<usize> = new_spikes.iter().enumerate()
                    .filter(|&(_, &s)| s != 0).map(|(i, _)| i).collect();
                Some(SpikeData::Sparse(active_indices))
            }
            SimulationKernel::ApplyModulation(_) => None,
        }
    }

    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState) -> SpikeData {
        let ctx = KernelContext { external_inputs, previous_spikes, history, current_tick, modulation, top_down_modulation: None };
        self.execute_kernel(SimulationKernel::PropagateSynapses, model, &ctx);
        self.execute_kernel(SimulationKernel::GenerateSpikes, model, &ctx).unwrap_or_default()
    }

    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
        let mut cpu = crate::cpu::CpuBackend::default();
        cpu.update_weights(model, previous_spikes, current_spikes, current_tick, reward, history);
    }

    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: NeuromodulationState, history: &[Vec<bool>]) {
        let mut cpu = crate::cpu::CpuBackend::default();
        cpu.update_weights_modulated(model, previous_spikes, current_spikes, current_tick, reward, modulation, history);
    }

    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        let mut cpu = crate::cpu::CpuBackend::default();
        cpu.structural_plasticity(model, reward, history);
    }

    fn structural_plasticity_with_surprise(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>], block_surprise: &[f32]) {
        let mut cpu = crate::cpu::CpuBackend::default();
        cpu.structural_plasticity_with_surprise(model, reward, history, block_surprise);
    }
}

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
}

impl Default for CppBackend {
    fn default() -> Self {
        Self { name: "CppBackend" }
    }
}

impl ComputeBackend for CppBackend {
    fn name(&self) -> &'static str { self.name }

    fn execute_kernel(&mut self, kernel: SimulationKernel, model: &mut BakedModel, ctx: &KernelContext) -> Option<SpikeData> {
        match kernel {
            SimulationKernel::PropagateSynapses => {
                let n_count = model.neurons.len();
                // Apply external inputs directly to proximal potential with dendritic gating
                for (i, &val) in ctx.external_inputs.iter().enumerate() {
                    if i < n_count {
                        let gate = model.neurons.dendritic_gate[i];
                        let gated_val = ((val as i64 * gate as i64) >> 10) as i32;
                        model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(gated_val);
                    }
                }

                unsafe {
                    cpp_propagate_spikes(
                        model.synapses.as_ffi(),
                        ctx.previous_spikes.as_ptr(),
                        model.neurons.as_ffi()
                    );
                    cpp_recover_stp(model.synapses.as_ffi());
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
                        model.config.ip_increment,
                        model.config.ip_decay,
                        model.config.noise_amplitude
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
        let ctx = KernelContext { external_inputs, previous_spikes, history, current_tick, modulation };
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
}

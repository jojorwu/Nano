use genesis_core::{BakedModel, IValue, SpikeData, NeuromodulationState, NeuronsFFI};
use crate::{ComputeBackend, SimulationKernel, KernelContext};

extern "C" {
    fn cpp_propagate_spikes(
        source_indices: *const u32,
        target_indices: *const u32,
        weights: *const IValue,
        compartments: *const u8,
        synapse_count: u32,
        previous_spikes: *const bool,
        neurons: NeuronsFFI
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
                // Apply external inputs directly to proximal potential
                for (i, &val) in ctx.external_inputs.iter().enumerate() {
                    if i < n_count {
                        model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(val);
                    }
                }

                unsafe {
                    cpp_propagate_spikes(
                        model.synapses.source_index.as_ptr(),
                        model.synapses.target_index.as_ptr(),
                        model.synapses.weight.as_ptr(),
                        model.synapses.compartment.as_ptr() as *const u8,
                        model.synapses.len() as u32,
                        ctx.previous_spikes.as_ptr(),
                        model.neurons.as_ffi()
                    );
                }
                None
            }
            SimulationKernel::UpdateMembranePotentials => {
                // For now, delegate back to CPU backend or implement in C++
                None
            }
            SimulationKernel::GenerateSpikes => {
                // Delegate to CPU logic for now to ensure parity while we migrate kernels
                let mut cpu = crate::cpu::CpuBackend::default();
                cpu.execute_kernel(SimulationKernel::GenerateSpikes, model, ctx)
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

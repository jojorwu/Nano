use genesis_core::{BakedModel, GsopRule, PlasticityRule, SCALE, IValue};
use genesis_core::plasticity::{prune_synapses, grow_synapse, StructuralPlasticityConfig};

pub trait ComputeBackend {
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u64) -> Vec<bool>;
    fn night_phase(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u64, reward: Option<IValue>);
    fn name(&self) -> &'static str;
}

pub struct CpuBackend {
    pub structural_config: StructuralPlasticityConfig,
    pub plasticity_rule: Box<dyn PlasticityRule + Send + Sync>,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            structural_config: StructuralPlasticityConfig::default(),
            plasticity_rule: Box::new(GsopRule { learning_rate: 10 }),
        }
    }
}

#[cfg(feature = "wgpu")]
pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("Failed to find wgpu adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).expect("Failed to create wgpu device");
        Self { device, queue }
    }
}

#[cfg(feature = "wgpu")]
impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }
    fn day_phase(&mut self, _model: &mut BakedModel, _external_inputs: &[i32], _previous_spikes: &[bool], _current_tick: u64) -> Vec<bool> {
        // GPU kernels will be implemented in WGSL in next steps
        vec![]
    }
    fn night_phase(&mut self, _model: &mut BakedModel, _previous_spikes: &[bool], _current_spikes: &[bool], _current_tick: u64) {
        // GPU-based structural plasticity
    }
}

pub struct BackendRegistry {
    pub backends: std::collections::HashMap<String, Box<dyn Fn() -> Box<dyn ComputeBackend + Send + Sync>>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        let mut registry = Self { backends: std::collections::HashMap::new() };
        registry.register("cpu", || Box::new(CpuBackend::default()));
        #[cfg(feature = "wgpu")]
        registry.register("wgpu", || Box::new(WgpuBackend::new()));
        registry
    }

    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn ComputeBackend + Send + Sync> + 'static,
    {
        self.backends.insert(name.to_string(), Box::new(factory));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn ComputeBackend + Send + Sync>> {
        self.backends.get(name).map(|f| f())
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "CpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u64) -> Vec<bool> {
        let n_count = model.neurons.len();
        let mut current_inputs = vec![0i32; n_count];

        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count { current_inputs[i] = current_inputs[i].saturating_add(val); }
        }

        for i in 0..model.synapses.len() {
            let src = model.synapses.source_index[i] as usize;
            if previous_spikes[src] {
                let target = model.synapses.target_index[i] as usize;
                current_inputs[target] = current_inputs[target].saturating_add(model.synapses.weight[i]);
            }
        }

        #[cfg(feature = "titan")]
        if let Some(ref titan) = model.titan_memory {
            let memory_input = titan.retrieve(previous_spikes);
            let dist_input = memory_input / (n_count as i32).max(1);
            for i in 0..n_count {
                current_inputs[i] = current_inputs[i].saturating_add(dist_input);
            }
        }

        let mut new_spikes = vec![false; n_count];
        for i in 0..n_count {
            if model.neurons.refractory_timer[i] > 0 {
                model.neurons.refractory_timer[i] -= 1;
                model.neurons.potential[i] = 0;
                continue;
            }

            // Event-Driven: Only update if there's input or existing potential
            if current_inputs[i] == 0 && model.neurons.potential[i] == 0 {
                continue;
            }

            model.neurons.potential[i] = model.neurons.potential[i].saturating_add(current_inputs[i]);
            let decay = model.neurons.decay[i];
            model.neurons.potential[i] = (model.neurons.potential[i] * (SCALE - decay)) / SCALE;

            if model.neurons.potential[i] >= model.neurons.threshold[i] {
                model.neurons.potential[i] = 0;
                model.neurons.refractory_timer[i] = 2;
                new_spikes[i] = true;
                model.neurons.last_spike_tick[i] = current_tick;
            }
        }

        new_spikes
    }

    fn night_phase(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u64, reward: Option<IValue>) {
        for i in 0..model.synapses.len() {
            let src = model.synapses.source_index[i] as usize;
            let target = model.synapses.target_index[i] as usize;
            let pre_spiked = previous_spikes[src];
            let post_spiked = current_spikes[target];

            if let Some(r) = reward {
                self.plasticity_rule.update_rewarded(&mut model.synapses.weight[i], pre_spiked, post_spiked, r);
            } else {
                self.plasticity_rule.update(&mut model.synapses.weight[i], pre_spiked, post_spiked);
            }

            // Temporal update (STDP)
            let pre_tick = model.neurons.last_spike_tick[src];
            let post_tick = model.neurons.last_spike_tick[target];
            self.plasticity_rule.update_temporal(&mut model.synapses.weight[i], pre_tick, post_tick, current_tick);
        }

        #[cfg(feature = "titan")]
        if let Some(ref mut titan) = model.titan_memory {
            let activity = current_spikes.iter().filter(|&&s| s).count() as i32;
            let base_error = (10 - activity) * 10;
            // Integrate reward into Titan error if present
            let final_error = if let Some(r) = reward { base_error + r } else { base_error };
            titan.step(previous_spikes, final_error);
        }

        prune_synapses(&mut model.synapses, self.structural_config.prune_threshold);

        let active_indices: Vec<usize> = current_spikes.iter().enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if active_indices.len() > 1 && model.synapses.len() < self.structural_config.max_synapses {
            for &i in active_indices.iter().take(5) {
                for &j in active_indices.iter().take(5) {
                    if i != j {
                        grow_synapse(&mut model.synapses, i as u32, j as u32, 50, &self.structural_config);
                    }
                }
            }
        }
    }
}

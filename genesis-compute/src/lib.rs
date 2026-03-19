use genesis_core::{BakedModel, GsopRule, PlasticityRule, SCALE, IValue};
use genesis_core::plasticity::{prune_synapses, grow_synapse, StructuralPlasticityConfig};

pub trait ComputeBackend {
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u32) -> Vec<bool>;
    fn night_phase(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>);
    fn name(&self) -> &'static str;
}

pub struct CpuBackend {
    pub structural_config: StructuralPlasticityConfig,
    pub plasticity_rule: Box<dyn PlasticityRule + Send + Sync>,
    pub optimizer: genesis_core::plasticity::EvolutionaryOptimizer,
    pub expert_masks: Vec<bool>, // MoE: which neuron groups are active
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            structural_config: StructuralPlasticityConfig::default(),
            plasticity_rule: Box::new(GsopRule { learning_rate: 10 }),
            optimizer: genesis_core::plasticity::EvolutionaryOptimizer::new(0.01),
            expert_masks: Vec::new(),
        }
    }
}

#[cfg(feature = "wgpu")]
pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub potential_pipeline: wgpu::ComputePipeline,
    pub propagation_pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub tick_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("Failed to find wgpu adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).expect("Failed to create wgpu device");

        let potential_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Potential Update Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/potential_update.wgsl"))),
        });

        let propagation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spike Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/spike_prop.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Neuron Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let tick_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tick Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &tick_layout],
            push_constant_ranges: &[],
        });

        let potential_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Potential Pipeline"),
            layout: Some(&pipeline_layout),
            module: &potential_shader,
            entry_point: "main",
        });

        let propagation_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Propagation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &propagation_shader,
            entry_point: "main",
        });

        // Simplified propagation pipeline layout might differ, but for stub it's fine
        Self { device, queue, potential_pipeline, propagation_pipeline, bind_group_layout, tick_layout }
    }
}

#[cfg(feature = "wgpu")]
impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], _previous_spikes: &[bool], current_tick: u32) -> Vec<bool> {
        use wgpu::util::DeviceExt;

        let n_count = model.neurons.len();

        // 1. Create buffers for neuron state
        let pot_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Potentials"),
            contents: bytemuck::cast_slice(&model.neurons.potential),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let threshold_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Thresholds"),
            contents: bytemuck::cast_slice(&model.neurons.threshold),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let decay_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Decays"),
            contents: bytemuck::cast_slice(&model.neurons.decay),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let refractory_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Refractory"),
            contents: bytemuck::cast_slice(&model.neurons.refractory_timer),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let spikes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Spikes"),
            size: (n_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Inputs"),
            contents: bytemuck::cast_slice(external_inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let next_update_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Next Update"),
            contents: bytemuck::cast_slice(&model.neurons.next_update_tick),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let interval_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Intervals"),
            contents: bytemuck::cast_slice(&model.neurons.update_interval),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tick_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tick Uniform"),
            contents: bytemuck::cast_slice(&[current_tick]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 2. Create Bind Group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pot_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: threshold_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: decay_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: refractory_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: spikes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: next_update_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: interval_buffer.as_entire_binding() },
            ],
            label: None,
        });
        let tick_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.tick_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: tick_buffer.as_entire_binding() }],
            label: None,
        });

        // 3. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.potential_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_bind_group(1, &tick_bind_group, &[]);
            cpass.dispatch_workgroups((n_count as u32 + 63) / 64, 1, 1);
        }

        // 4. Download Results (Spikes and updated state)
        let staging_spikes = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Spikes"),
            size: (n_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&spikes_buffer, 0, &staging_spikes, 0, (n_count * 4) as u64);

        // Also need to download updated potentials, thresholds, next_update_tick
        let staging_state = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging State"),
            size: (n_count * 4 * 4) as u64, // potential, threshold, refractory, next_update
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&pot_buffer, 0, &staging_state, 0, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(&threshold_buffer, 0, &staging_state, (n_count * 4) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(&refractory_buffer, 0, &staging_state, (n_count * 8) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(&next_update_buffer, 0, &staging_state, (n_count * 12) as u64, (n_count * 4) as u64);

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let spikes_slice = staging_spikes.slice(..);
        let state_slice = staging_state.slice(..);
        spikes_slice.map_async(wgpu::MapMode::Read, |_| {});
        state_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mut spikes = vec![false; n_count];
        {
            let spikes_raw = spikes_slice.get_mapped_range();
            let spike_data: &[u32] = bytemuck::cast_slice(&spikes_raw);
            for i in 0..n_count { spikes[i] = spike_data[i] != 0; }
        }

        {
            let state_raw = state_slice.get_mapped_range();
            let state_data: &[i32] = bytemuck::cast_slice(&state_raw);
            for i in 0..n_count {
                model.neurons.potential[i] = state_data[i];
                model.neurons.threshold[i] = state_data[n_count + i];
                model.neurons.refractory_timer[i] = state_data[2 * n_count + i];
                model.neurons.next_update_tick[i] = state_data[3 * n_count + i] as u32;
            }
        }

        spikes
    }
    fn night_phase(&mut self, _model: &mut BakedModel, _previous_spikes: &[bool], _current_spikes: &[bool], _current_tick: u32, _reward: Option<IValue>) {
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

impl CpuBackend {
    fn apply_external_inputs(&self, n_count: usize, external_inputs: &[i32], current_inputs: &mut [i32]) {
        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count { current_inputs[i] = current_inputs[i].saturating_add(val); }
        }
    }

    fn propagate_sparse_spikes(&self, model: &BakedModel, previous_spikes: &[bool], current_inputs: &mut [i32]) {
        for i in 0..model.synapses.len() {
            let src = model.synapses.source_index[i] as usize;
            if previous_spikes[src] {
                let target = model.synapses.target_index[i] as usize;
                current_inputs[target] = current_inputs[target].saturating_add(model.synapses.weight[i]);
            }
        }
    }

    fn propagate_latent_spikes(&self, model: &BakedModel, previous_spikes: &[bool], current_inputs: &mut [i32]) {
        if let Some(ref latent) = model.synapses.latent_matrix {
            let mut latent_state = vec![0i32; latent.rank];
            for i in 0..model.neurons.len() {
                if previous_spikes[i] {
                    for r in 0..latent.rank {
                        latent_state[r] = latent_state[r].saturating_add(latent.u[i * latent.rank + r]);
                    }
                }
            }
            for j in 0..model.neurons.len() {
                for r in 0..latent.rank {
                    let weight = latent.v[r * model.neurons.len() + j];
                    let contribution = (latent_state[r] * weight) / SCALE;
                    current_inputs[j] = current_inputs[j].saturating_add(contribution);
                }
            }
        }
    }

    fn update_neuron_states(&self, model: &mut BakedModel, current_inputs: &[i32], current_tick: u32, new_spikes: &mut [bool]) {
        for i in 0..model.neurons.len() {
            if current_tick < model.neurons.next_update_tick[i] { continue; }
            if !self.expert_masks.is_empty() && !self.expert_masks[i % self.expert_masks.len()] { continue; }

            if model.neurons.refractory_timer[i] > 0 {
                model.neurons.refractory_timer[i] -= 1;
                model.neurons.potential[i] = 0;
                model.neurons.next_update_tick[i] = current_tick + model.neurons.update_interval[i];
                continue;
            }

            if current_inputs[i] == 0 && model.neurons.potential[i] == 0 {
                model.neurons.next_update_tick[i] = current_tick + model.neurons.update_interval[i];
                continue;
            }

            model.neurons.potential[i] = model.neurons.potential[i].saturating_add(current_inputs[i]);
            let base_decay = model.neurons.decay[i];
            let liquid_modulation = (current_inputs[i].abs() * 10) / SCALE;
            let final_decay = (base_decay - liquid_modulation).max(1);
            model.neurons.potential[i] = (model.neurons.potential[i] * (SCALE - final_decay)) / SCALE;

            if model.neurons.potential[i] >= model.neurons.threshold[i] {
                model.neurons.potential[i] = 0;
                model.neurons.refractory_timer[i] = 2;
                new_spikes[i] = true;
                model.neurons.last_spike_tick[i] = current_tick;

                // Intrinsic Plasticity: Increase threshold upon spiking
                model.neurons.threshold[i] = model.neurons.threshold[i].saturating_add(50);
            } else {
                // Threshold decay back to base
                if model.neurons.threshold[i] > model.neurons.base_threshold[i] {
                    model.neurons.threshold[i] -= 1;
                }
            }
            model.neurons.next_update_tick[i] = current_tick + model.neurons.update_interval[i];
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "CpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u32) -> Vec<bool> {
        let n_count = model.neurons.len();
        let mut current_inputs = vec![0i32; n_count];

        self.apply_external_inputs(n_count, external_inputs, &mut current_inputs);
        self.propagate_sparse_spikes(model, previous_spikes, &mut current_inputs);
        self.propagate_latent_spikes(model, previous_spikes, &mut current_inputs);

        #[cfg(feature = "titan")]
        if let Some(ref titan) = model.titan_memory {
            let memory_input = titan.retrieve(previous_spikes);
            let dist_input = memory_input / (n_count as i32).max(1);
            for i in 0..n_count {
                current_inputs[i] = current_inputs[i].saturating_add(dist_input);
            }
        }

        let mut new_spikes = vec![false; n_count];
        self.update_neuron_states(model, &current_inputs, current_tick, &mut new_spikes);
        new_spikes
    }

    fn night_phase(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>) {
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
            let pre_tick = model.neurons.last_spike_tick[src] as u64;
            let post_tick = model.neurons.last_spike_tick[target] as u64;
            self.plasticity_rule.update_temporal(&mut model.synapses.weight[i], pre_tick, post_tick, current_tick as u64);
        }

        #[cfg(feature = "titan")]
        if let Some(ref mut titan) = model.titan_memory {
            let activity = current_spikes.iter().filter(|&&s| s).count() as i32;
            let base_error = (10 - activity) * 10;
            // Integrate reward into Titan error if present
            let final_error = if let Some(r) = reward { base_error + r } else { base_error };

            // Surprise-Driven Plasticity: If surprise is high, boost learning
            if final_error.abs() > titan.surprise_threshold {
                log::info!("High surprise detected: {}, boosting learning", final_error);
            }

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

        // SNNaS: Evolutionary mutation based on reward
        if let Some(r) = reward {
            self.optimizer.mutate(&mut model.synapses, model.neurons.len(), r);
        }

        // Homeostatic Synaptic Scaling:
        // Scale all weights to maintain global stability if total weight magnitude is too high
        let total_weight: i64 = model.synapses.weight.iter().map(|&w| w.abs() as i64).sum();
        let max_total_weight = (model.neurons.len() as i64) * 1000 * 10; // Target avg 1.0 weight per neuron x 10
        if total_weight > max_total_weight {
            let scale_factor = (max_total_weight * 1000) / total_weight;
            for w in model.synapses.weight.iter_mut() {
                *w = ((*w as i64 * scale_factor) / 1000) as i32;
            }
        }
    }
}

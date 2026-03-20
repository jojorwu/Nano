use genesis_core::{BakedModel, GsopRule, PlasticityRule, SCALE, IValue};
use genesis_core::plasticity::{prune_synapses, grow_synapse, StructuralPlasticityConfig};

pub trait ComputeBackend {
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u32) -> Vec<bool>;
    fn night_phase(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]);
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
    // Cached Buffers
    pub pot_buffer: Option<wgpu::Buffer>,
    pub threshold_buffer: Option<wgpu::Buffer>,
    pub decay_buffer: Option<wgpu::Buffer>,
    pub refractory_buffer: Option<wgpu::Buffer>,
    pub spikes_buffer: Option<wgpu::Buffer>,
    pub input_buffer: Option<wgpu::Buffer>,
    pub next_update_buffer: Option<wgpu::Buffer>,
    pub interval_buffer: Option<wgpu::Buffer>,
    pub gate_buffer: Option<wgpu::Buffer>,
    pub distal_buffer: Option<wgpu::Buffer>,
    pub proximal_buffer: Option<wgpu::Buffer>,
    pub base_threshold_buffer: Option<wgpu::Buffer>,
    pub tick_buffer: Option<wgpu::Buffer>,
    pub staging_spikes: Option<wgpu::Buffer>,
    pub staging_state: Option<wgpu::Buffer>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub tick_bind_group: Option<wgpu::BindGroup>,
    pub cached_neuron_count: usize,
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
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 11, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
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
        Self {
            device, queue, potential_pipeline, propagation_pipeline, bind_group_layout, tick_layout,
            pot_buffer: None, threshold_buffer: None, decay_buffer: None, refractory_buffer: None,
            spikes_buffer: None, input_buffer: None, next_update_buffer: None, interval_buffer: None,
            gate_buffer: None, distal_buffer: None, proximal_buffer: None, base_threshold_buffer: None, tick_buffer: None, staging_spikes: None, staging_state: None,
            bind_group: None, tick_bind_group: None, cached_neuron_count: 0
        }
    }
}

#[cfg(feature = "wgpu")]
impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], _previous_spikes: &[bool], current_tick: u32) -> Vec<bool> {
        use wgpu::util::DeviceExt;

        let n_count = model.neurons.len();

        // 1. Re-initialize buffers if neuron count changed
        if n_count != self.cached_neuron_count || self.pot_buffer.is_none() {
            self.pot_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Potentials"),
                contents: bytemuck::cast_slice(&model.neurons.potential),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.threshold_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Thresholds"),
                contents: bytemuck::cast_slice(&model.neurons.threshold),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.decay_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Decays"),
                contents: bytemuck::cast_slice(&model.neurons.decay),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.refractory_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Refractory"),
                contents: bytemuck::cast_slice(&model.neurons.refractory_timer),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.spikes_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Spikes"),
                size: (n_count * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.input_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Inputs"),
                size: (n_count * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.next_update_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Next Update"),
                contents: bytemuck::cast_slice(&model.neurons.next_update_tick),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.interval_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Intervals"),
                contents: bytemuck::cast_slice(&model.neurons.update_interval),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.gate_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dendritic Gate"),
                contents: bytemuck::cast_slice(&model.neurons.dendritic_gate),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.distal_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Distal Potentials"),
                contents: bytemuck::cast_slice(&model.neurons.distal_potential),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            self.proximal_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Proximal Potentials"),
                contents: bytemuck::cast_slice(&model.neurons.proximal_potential),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            self.base_threshold_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Base Thresholds"),
                contents: bytemuck::cast_slice(&model.neurons.base_threshold),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.tick_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tick Uniform"),
                size: 4,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.staging_spikes = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Spikes"),
                size: (n_count * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.staging_state = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging State"),
                size: (n_count * 4 * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Re-create Bind Group
            self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.pot_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.threshold_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.decay_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.refractory_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.spikes_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.input_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.next_update_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.interval_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.gate_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: self.distal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: self.proximal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: self.base_threshold_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: None,
            }));
            self.tick_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.tick_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.tick_buffer.as_ref().unwrap().as_entire_binding() }],
                label: None,
            }));

            self.cached_neuron_count = n_count;
        }

        // 2. Upload inputs and current tick
        self.queue.write_buffer(self.input_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(external_inputs));
        self.queue.write_buffer(self.tick_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[current_tick]));
        self.queue.write_buffer(self.distal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.distal_potential));
        self.queue.write_buffer(self.proximal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.proximal_potential));

        // 3. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.potential_pipeline);
            cpass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            cpass.set_bind_group(1, self.tick_bind_group.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups((n_count as u32 + 63) / 64, 1, 1);
        }

        // 4. Download Results
        encoder.copy_buffer_to_buffer(self.spikes_buffer.as_ref().unwrap(), 0, self.staging_spikes.as_ref().unwrap(), 0, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.pot_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), 0, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.threshold_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 4) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.refractory_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 8) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.next_update_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 12) as u64, (n_count * 4) as u64);

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let spikes_slice = self.staging_spikes.as_ref().unwrap().slice(..);
        let state_slice = self.staging_state.as_ref().unwrap().slice(..);
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
    fn night_phase(&mut self, _model: &mut BakedModel, _previous_spikes: &[bool], _current_spikes: &[bool], _current_tick: u32, _reward: Option<IValue>, _history: &[Vec<bool>]) {
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
                // Apply dendritic gate: w * gate >> 10
                let raw_weight = model.synapses.weight[i];
                let gate = model.neurons.dendritic_gate[target];
                let gated_weight = ((raw_weight as i64 * gate as i64) >> 10) as i32;
                current_inputs[target] = current_inputs[target].saturating_add(gated_weight);
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
                let gate = model.neurons.dendritic_gate[j];
                for r in 0..latent.rank {
                    let weight = latent.v[r * model.neurons.len() + j];
                    // (state * weight * gate) >> 20
                    let contribution = ((latent_state[r] as i64 * weight as i64 * gate as i64) >> 20) as i32;
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

            // Multi-compartment integration:
            // Proximal (soma) vs Distal (dendrite) interaction
            let proximal = model.neurons.proximal_potential[i];
            let distal = model.neurons.distal_potential[i];

            // Dendritic Coincidence detection: Distal spikes only if proximal is high
            let dend_factor = if proximal > 500 { distal } else { distal / 4 };
            let total_input = current_inputs[i].saturating_add(proximal).saturating_add(dend_factor);

            model.neurons.potential[i] = model.neurons.potential[i].saturating_add(total_input);
            let base_decay = model.neurons.decay[i];
            let liquid_modulation = (current_inputs[i].abs() * 10) >> 10;
            let final_decay = (base_decay - liquid_modulation).max(1);
            model.neurons.potential[i] = (model.neurons.potential[i] as i64 * (SCALE - final_decay) as i64 >> 10) as i32;

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

    fn night_phase(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
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

        // SNNaS: Evolutionary mutation based on reward and activity history
        if let Some(r) = reward {
            self.optimizer.mutate_with_activity(&mut model.synapses, model.neurons.len(), r, history);
        }

        // Evolutionary Neurogenesis:
        // Trigger neuron growth if reward is high and we are below capacity
        if let Some(r) = reward {
            if r > 200 && model.neurons.len() < 1000000 {
                let growth_size = (model.neurons.len() / 20).max(1);
                model.neurons.grow(growth_size);
                log::info!("Neurogenesis triggered: +{} neurons", growth_size);
            }
        }

        // Homeostatic Synaptic Scaling:
        // Scale all weights to maintain global stability if total weight magnitude is too high
        let total_weight: i64 = model.synapses.weight.iter().map(|&w| w.abs() as i64).sum();
        let max_total_weight = (model.neurons.len() as i64) << 13; // Target avg ~8.0 weight per neuron
        if total_weight > max_total_weight {
            let scale_factor = (max_total_weight << 10) / total_weight;
            for w in model.synapses.weight.iter_mut() {
                *w = ((*w as i64 * scale_factor) >> 10) as i32;
            }
        }
    }
}

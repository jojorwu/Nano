use genesis_core::{BakedModel, GsopRule, PlasticityRule, SCALE, IValue, Compartment};
use genesis_core::plasticity::{prune_synapses, StructuralPlasticityConfig};

pub trait ComputeBackend {
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u32) -> Vec<bool>;
    /// Weight updates (fast phase of learning)
    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]);
    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]);
    fn sync_state(&mut self, _model: &mut BakedModel) {} // Optional: download state from device
    fn name(&self) -> &'static str;
}

pub struct CpuBackend {
    pub structural_config: StructuralPlasticityConfig,
    pub plasticity_rule: Box<dyn PlasticityRule + Send + Sync>,
    pub optimizer: genesis_core::plasticity::EvolutionaryOptimizer,
    pub expert_masks: Vec<bool>, // MoE: which neuron groups are active
    pub synapse_index: Vec<Vec<usize>>, // source_neuron -> list of synapse indices
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            structural_config: StructuralPlasticityConfig::default(),
            plasticity_rule: Box::new(GsopRule { learning_rate: 10 }),
            optimizer: genesis_core::plasticity::EvolutionaryOptimizer::new(0.01),
            expert_masks: Vec::new(),
            synapse_index: Vec::new(),
        }
    }
}

#[cfg(feature = "wgpu")]
pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub potential_pipeline: wgpu::ComputePipeline,
    pub propagation_pipeline: wgpu::ComputePipeline,
    pub gsop_pipeline: wgpu::ComputePipeline,
    pub latent_accum_pipeline: wgpu::ComputePipeline,
    pub latent_distrib_pipeline: wgpu::ComputePipeline,
    pub potential_layout: wgpu::BindGroupLayout,
    pub gsop_layout: wgpu::BindGroupLayout,
    pub latent_accum_layout: wgpu::BindGroupLayout,
    pub latent_distrib_layout: wgpu::BindGroupLayout,
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
    pub weight_buffer: Option<wgpu::Buffer>,
    pub source_buffer: Option<wgpu::Buffer>,
    pub target_buffer: Option<wgpu::Buffer>,
    pub gate_buffer: Option<wgpu::Buffer>,
    pub distal_buffer: Option<wgpu::Buffer>,
    pub proximal_buffer: Option<wgpu::Buffer>,
    pub base_threshold_buffer: Option<wgpu::Buffer>,
    pub layer_id_buffer: Option<wgpu::Buffer>,
    pub backprop_buffer: Option<wgpu::Buffer>,
    pub u_matrix_buffer: Option<wgpu::Buffer>,
    pub v_matrix_buffer: Option<wgpu::Buffer>,
    pub latent_state_buffer: Option<wgpu::Buffer>,
    pub config_uniform_buffer: Option<wgpu::Buffer>,
    pub tick_buffer: Option<wgpu::Buffer>,
    pub sparse_spike_buffer: Option<wgpu::Buffer>,
    pub spike_counter_buffer: Option<wgpu::Buffer>,
    pub staging_spikes: Option<wgpu::Buffer>,
    pub staging_state: Option<wgpu::Buffer>,
    pub pre_spike_buffer: Option<wgpu::Buffer>,
    pub post_spike_buffer: Option<wgpu::Buffer>,
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

        let gsop_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GSOP Update Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/gsop_update.wgsl"))),
        });

        let latent_accum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Latent Accum Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/latent_accum.wgsl"))),
        });

        let latent_distrib_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Latent Distrib Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/latent_distrib.wgsl"))),
        });

        let potential_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Potential Layout"),
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
                wgpu::BindGroupLayoutEntry { binding: 14, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 15, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 12, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 13, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 16, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let gsop_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GSOP Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let latent_accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Latent Accum Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let latent_distrib_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Latent Distrib Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let tick_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tick Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let potential_p_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Potential Pipeline Layout"),
            bind_group_layouts: &[&potential_layout, &tick_layout],
            push_constant_ranges: &[],
        });

        let propagation_p_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Propagation Pipeline Layout"),
            bind_group_layouts: &[&potential_layout, &tick_layout],
            push_constant_ranges: &[],
        });

        let gsop_p_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GSOP Pipeline Layout"),
            bind_group_layouts: &[&gsop_layout, &tick_layout],
            push_constant_ranges: &[],
        });

        let latent_accum_p_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Latent Accum Pipeline Layout"),
            bind_group_layouts: &[&latent_accum_layout, &tick_layout],
            push_constant_ranges: &[],
        });

        let latent_distrib_p_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Latent Distrib Pipeline Layout"),
            bind_group_layouts: &[&latent_distrib_layout, &tick_layout],
            push_constant_ranges: &[],
        });

        let potential_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Potential Pipeline"),
            layout: Some(&potential_p_layout),
            module: &potential_shader,
            entry_point: "main",
        });

        let propagation_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Propagation Pipeline"),
            layout: Some(&propagation_p_layout),
            module: &propagation_shader,
            entry_point: "main",
        });

        let gsop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GSOP Pipeline"),
            layout: Some(&gsop_p_layout),
            module: &gsop_shader,
            entry_point: "main",
        });

        let latent_accum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Latent Accum Pipeline"),
            layout: Some(&latent_accum_p_layout),
            module: &latent_accum_shader,
            entry_point: "main",
        });

        let latent_distrib_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Latent Distrib Pipeline"),
            layout: Some(&latent_distrib_p_layout),
            module: &latent_distrib_shader,
            entry_point: "main",
        });

        // Simplified propagation pipeline layout might differ, but for stub it's fine
        Self {
            device, queue, potential_pipeline, propagation_pipeline, gsop_pipeline, latent_accum_pipeline, latent_distrib_pipeline,
            potential_layout, gsop_layout, latent_accum_layout, latent_distrib_layout, tick_layout,
            pot_buffer: None, threshold_buffer: None, decay_buffer: None, refractory_buffer: None,
            spikes_buffer: None, input_buffer: None, next_update_buffer: None, interval_buffer: None,
            weight_buffer: None, source_buffer: None, target_buffer: None,
            gate_buffer: None, distal_buffer: None, proximal_buffer: None, base_threshold_buffer: None, layer_id_buffer: None, backprop_buffer: None, config_uniform_buffer: None, tick_buffer: None,
            u_matrix_buffer: None, v_matrix_buffer: None, latent_state_buffer: None,
            sparse_spike_buffer: None, spike_counter_buffer: None, staging_spikes: None, staging_state: None,
            pre_spike_buffer: None, post_spike_buffer: None,
            bind_group: None, tick_bind_group: None, cached_neuron_count: 0
        }
    }
}

#[cfg(feature = "wgpu")]
impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u32) -> Vec<bool> {
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
            self.layer_id_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Layer IDs"),
                contents: bytemuck::cast_slice(&model.neurons.layer_id),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.backprop_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Backprop Signals"),
                contents: bytemuck::cast_slice(&model.neurons.backprop_signal),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.tick_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tick Uniform"),
                size: 4,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.config_uniform_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Config Buffer"),
                size: 8, // ip_increment, ip_decay
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.sparse_spike_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sparse Spikes"),
                size: (n_count * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.spike_counter_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Spike Counter"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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
                layout: &self.potential_layout,
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
                    wgpu::BindGroupEntry { binding: 14, resource: self.layer_id_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 15, resource: self.backprop_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: self.sparse_spike_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: self.spike_counter_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 16, resource: self.config_uniform_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: None,
            }));
            self.tick_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.tick_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.tick_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: None,
            }));

            if let Some(ref latent) = model.synapses.latent_matrix {
                self.u_matrix_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("U Matrix"),
                    contents: bytemuck::cast_slice(&latent.u),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
                self.v_matrix_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("V Matrix"),
                    contents: bytemuck::cast_slice(&latent.v),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
                self.latent_state_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Latent State"),
                    size: (latent.rank * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            self.cached_neuron_count = n_count;
        }

        // 2. Upload inputs and current tick
        self.queue.write_buffer(self.input_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(external_inputs));
        self.queue.write_buffer(self.tick_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[current_tick]));
        self.queue.write_buffer(self.config_uniform_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[model.config.intrinsic_plasticity_increment]));
        self.queue.write_buffer(self.config_uniform_buffer.as_ref().unwrap(), 4, bytemuck::cast_slice(&[model.config.intrinsic_plasticity_decay]));
        self.queue.write_buffer(self.distal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.distal_potential));
        self.queue.write_buffer(self.proximal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.proximal_potential));

        // 3. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Sparse MLA: Accumulate Latent State
        if let Some(ref latent) = model.synapses.latent_matrix {
            let active_indices: Vec<u32> = previous_spikes.iter().enumerate()
                .filter(|&(_, &s)| s).map(|(i, _)| i as u32).collect();

            if !active_indices.is_empty() {
                let accum_spike_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None, contents: bytemuck::cast_slice(&active_indices), usage: wgpu::BufferUsages::STORAGE,
                });
                let rank_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None, contents: bytemuck::cast_slice(&[latent.rank as u32, active_indices.len() as u32]), usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.latent_accum_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.u_matrix_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: accum_spike_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.latent_state_buffer.as_ref().unwrap().as_entire_binding() },
                    ],
                    label: None,
                });
                let rank_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.tick_layout,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: rank_buffer.as_entire_binding() }],
                    label: None,
                });

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cpass.set_pipeline(&self.latent_accum_pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.set_bind_group(1, &rank_bind_group, &[]);
                    cpass.dispatch_workgroups((active_indices.len() as u32 + 63) / 64, 1, 1);
                }
            }
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.potential_pipeline);
            cpass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            cpass.set_bind_group(1, self.tick_bind_group.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups((n_count as u32 + 63) / 64, 1, 1);
        }

        // Latent Distrib Pass
        if let Some(ref latent) = model.synapses.latent_matrix {
             let rank_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&[latent.rank as u32]), usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.latent_distrib_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.v_matrix_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.latent_state_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.gate_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.input_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: None,
            });
            let rank_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.tick_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: rank_buffer.as_entire_binding() }],
                label: None,
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.latent_distrib_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.set_bind_group(1, &rank_bind_group, &[]);
                cpass.dispatch_workgroups((n_count as u32 + 63) / 64, 1, 1);
            }
        }

        // 4. Download ONLY Sparse Spikes (Efficient)
        self.queue.write_buffer(self.spike_counter_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[0u32]));

        encoder.copy_buffer_to_buffer(self.sparse_spike_buffer.as_ref().unwrap(), 0, self.staging_spikes.as_ref().unwrap(), 0, (n_count * 4) as u64);
        // Also download counter
        let counter_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
             label: None, size: 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(self.spike_counter_buffer.as_ref().unwrap(), 0, &counter_staging, 0, 4);

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let spikes_slice = self.staging_spikes.as_ref().unwrap().slice(..);
        counter_staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        spikes_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mut spikes = vec![false; n_count];
        {
            let counter_raw = counter_staging.slice(..).get_mapped_range();
            let counter_data: &[u32] = bytemuck::cast_slice(&counter_raw);
            let count = counter_data[0] as usize;
            let spikes_raw = spikes_slice.get_mapped_range();
            let spike_indices: &[u32] = bytemuck::cast_slice(&spikes_raw);
            for i in 0..count.min(n_count) {
                let idx = spike_indices[i] as usize;
                if idx < n_count { spikes[idx] = true; }
            }
        }

        spikes
    }
    fn sync_state(&mut self, model: &mut BakedModel) {
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(self.pot_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), 0, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.threshold_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 4) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.refractory_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 8) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.next_update_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 12) as u64, (n_count * 4) as u64);

        // Download weights
        let weight_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weight Sync Staging"),
            size: (s_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(ref wb) = self.weight_buffer {
            encoder.copy_buffer_to_buffer(wb, 0, &weight_staging, 0, (s_count * 4) as u64);
        }

        self.queue.submit(Some(encoder.finish()));

        let state_slice = self.staging_state.as_ref().unwrap().slice(..);
        state_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

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

        // Download weights if available
        if self.weight_buffer.is_some() {
            let weight_slice = weight_staging.slice(..);
            weight_slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::Maintain::Wait);
            {
                let weight_raw = weight_slice.get_mapped_range();
                let weight_data: &[i32] = bytemuck::cast_slice(&weight_raw);
                for i in 0..s_count { model.synapses.weight[i] = weight_data[i]; }
            }
        }
    }

    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        // 1. Sync weights and thresholds back to CPU for structural processing
        self.sync_state(model);

        // 2. Perform structural plasticity using CPU implementation (re-using logic)
        let mut cpu_backend = CpuBackend::default();
        cpu_backend.structural_plasticity(model, reward, history);

        // 3. Reset GPU buffers in next tick (day_phase handles this by checking neuron count)
        // Resetting to None forces re-allocation and re-upload in the next day_phase
        self.cached_neuron_count = 0;
        self.weight_buffer = None;
        self.pot_buffer = None;
        log::info!("GPU Structural Plasticity: Weights re-synced and buffers cleared for re-initialization.");
    }
    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], _current_tick: u32, _reward: Option<IValue>, _history: &[Vec<bool>]) {
        use wgpu::util::DeviceExt;
        let s_count = model.synapses.len();
        let n_count = model.neurons.len();
        if s_count == 0 { return; }

        // 1. Re-initialize buffers if needed
        if self.weight_buffer.is_none() || s_count != model.synapses.len() {
             self.weight_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Synapse Weights"),
                contents: bytemuck::cast_slice(&model.synapses.weight),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.source_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Synapse Sources"),
                contents: bytemuck::cast_slice(&model.synapses.source_index),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.target_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Synapse Targets"),
                contents: bytemuck::cast_slice(&model.synapses.target_index),
                usage: wgpu::BufferUsages::STORAGE,
            }));
        }
        if self.pre_spike_buffer.is_none() || n_count != self.cached_neuron_count {
            self.pre_spike_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pre Spikes"),
                size: (n_count * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.post_spike_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Post Spikes"),
                size: (n_count * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // 2. Upload spikes
        let pre_data: Vec<u32> = previous_spikes.iter().map(|&s| if s { 1u32 } else { 0u32 }).collect();
        let post_data: Vec<u32> = current_spikes.iter().map(|&s| if s { 1u32 } else { 0u32 }).collect();
        self.queue.write_buffer(self.pre_spike_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&pre_data));
        self.queue.write_buffer(self.post_spike_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&post_data));

        // 3. Create GSOP Bind Group
        let gsop_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.gsop_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.weight_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.source_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.target_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.pre_spike_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.post_spike_buffer.as_ref().unwrap().as_entire_binding() },
            ],
            label: None,
        });

        // 4. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.gsop_pipeline);
            cpass.set_bind_group(0, &gsop_bind_group, &[]);
            // Re-using tick_bind_group for learning_rate if bound to group 1
            cpass.set_bind_group(1, self.tick_bind_group.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups((s_count as u32 + 63) / 64, 1, 1);
        }

        // 5. Download weights (Slow, but necessary for now until we move save logic to GPU)
        let staging_weights = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Weights"),
            size: (s_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(self.weight_buffer.as_ref().unwrap(), 0, &staging_weights, 0, (s_count * 4) as u64);

        self.queue.submit(Some(encoder.finish()));

        let weights_slice = staging_weights.slice(..);
        weights_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        // Weight download moved to sync_state to keep weights on GPU during simulation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::{BakedModel, NeuronsSoA, SynapsesSoA};

    #[test]
    fn test_cpu_dendritic_gating() {
        use crate::ComputeBackend;
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
        let spikes = backend.day_phase(&mut model, &[0, 0], &prev_spikes, 1);
        assert!(spikes[1]); // Should spike

        // 2. Closed gate (0.0)
        model.neurons.potential[1] = 0;
        model.neurons.dendritic_gate[1] = 0;
        let spikes = backend.day_phase(&mut model, &[0, 0], &prev_spikes, 2);
        assert!(!spikes[1]); // Should not spike
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

    pub fn rebuild_index(&mut self, model: &BakedModel) {
        let n_count = model.neurons.len();
        self.synapse_index = vec![Vec::new(); n_count];
        for i in 0..model.synapses.len() {
            let src = model.synapses.source_index[i] as usize;
            if src < n_count {
                self.synapse_index[src].push(i);
            }
        }
    }

    fn propagate_sparse_spikes(&mut self, model: &mut BakedModel, previous_spikes: &[bool]) {
        let active_indices: Vec<usize> = previous_spikes.iter().enumerate()
            .filter(|&(_, &s)| s).map(|(i, _)| i).collect();

        if active_indices.is_empty() { return; }

        // Ensure index is ready (Lazy initialization or rebuild after structural changes)
        if self.synapse_index.len() != model.neurons.len() {
            self.rebuild_index(model);
        }

        // Optimized spike propagation: O(active_spikes * average_fanout)
        for &src in &active_indices {
            if src >= self.synapse_index.len() { continue; }
            for &syn_idx in &self.synapse_index[src] {
                let target = model.synapses.target_index[syn_idx] as usize;
                let gate = model.neurons.dendritic_gate[target];

                if gate < 8 { continue; }

                let gated_weight = ((model.synapses.weight[syn_idx] as i64 * gate as i64) >> 10) as i32;

                match model.synapses.compartment[syn_idx] {
                    Compartment::Proximal => {
                        model.neurons.proximal_potential[target] = model.neurons.proximal_potential[target].saturating_add(gated_weight);
                    }
                    Compartment::Distal => {
                        model.neurons.distal_potential[target] = model.neurons.distal_potential[target].saturating_add(gated_weight);
                    }
                }
            }
        }
    }

    fn propagate_latent_spikes(&self, model: &mut BakedModel, previous_spikes: &[bool]) {
        if let Some(ref latent) = model.synapses.latent_matrix {
            let mut latent_state = vec![0i32; latent.rank];
            // Sparse MLA: iterate only over spiked neurons
            let active_indices: Vec<usize> = previous_spikes.iter().enumerate()
                .filter(|&(_, &s)| s).map(|(i, _)| i).collect();

            if active_indices.is_empty() { return; }

            for i in active_indices {
                let offset = i * latent.rank;
                for r in 0..latent.rank {
                    latent_state[r] = latent_state[r].saturating_add(latent.u[offset + r]);
                }
            }

            // SIMD-friendly second pass: propagate latent state to all neurons
            for j in 0..model.neurons.len() {
                let gate = model.neurons.dendritic_gate[j];
                let mut sum = 0i64;
                for r in 0..latent.rank {
                    let weight = latent.v[r * model.neurons.len() + j];
                    sum += latent_state[r] as i64 * weight as i64;
                }
                let contribution = ((sum * gate as i64) >> 20) as i32;

                // Latent MLA connections are predominantly distal in this architecture
                model.neurons.distal_potential[j] = model.neurons.distal_potential[j].saturating_add(contribution);
            }
        }
    }

    fn update_neuron_states(&self, model: &mut BakedModel, current_tick: u32, new_spikes: &mut [bool]) {
        let n_count = model.neurons.len();
        let coincidence_threshold = model.config.dendritic_coincidence_threshold;
        let ip_inc = model.config.ip_increment;
        let ip_dec = model.config.ip_decay;

        // Vectorized-friendly loop for neuron updates
        for i in 0..n_count {
            if current_tick < model.neurons.next_update_tick[i] { continue; }
            if !self.expert_masks.is_empty() && !self.expert_masks[i % self.expert_masks.len()] { continue; }

            let refr = model.neurons.refractory_timer[i];
            if refr > 0 {
                model.neurons.refractory_timer[i] = refr - 1;
                model.neurons.potential[i] = 0;
                model.neurons.next_update_tick[i] = current_tick + model.neurons.update_interval[i];
                continue;
            }

            let proximal = model.neurons.proximal_potential[i];
            let distal = model.neurons.distal_potential[i];

            let dend_factor = if proximal >= coincidence_threshold { distal } else { distal >> 2 };
            let mut pot = model.neurons.potential[i].saturating_add(proximal).saturating_add(dend_factor);

            // LLIF: Dynamic Decay
            let liquid_mod = ((proximal.abs() + distal.abs()) * 10) >> 10;
            let final_decay = (model.neurons.decay[i] - liquid_mod).max(1);
            pot = ((pot as i64 * (SCALE - final_decay) as i64) >> 10) as i32;

            if pot >= model.neurons.threshold[i] {
                model.neurons.potential[i] = 0;
                model.neurons.refractory_timer[i] = 2;
                new_spikes[i] = true;
                model.neurons.last_spike_tick[i] = current_tick;
                model.neurons.backprop_signal[i] = SCALE;
                model.neurons.threshold[i] = model.neurons.threshold[i].saturating_add(ip_inc);
            } else {
                model.neurons.potential[i] = pot;
                if model.neurons.threshold[i] > model.neurons.base_threshold[i] {
                    model.neurons.threshold[i] = model.neurons.threshold[i].saturating_sub(ip_dec);
                }
                model.neurons.backprop_signal[i] = ((model.neurons.backprop_signal[i] as i64 * 800) >> 10) as i32;
            }
            model.neurons.next_update_tick[i] = current_tick + model.neurons.update_interval[i];
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "CpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], current_tick: u32) -> Vec<bool> {
        let n_count = model.neurons.len();

        // Apply external inputs directly to proximal potential
        for (i, &val) in external_inputs.iter().enumerate() {
            if i < n_count { model.neurons.proximal_potential[i] = model.neurons.proximal_potential[i].saturating_add(val); }
        }

        self.propagate_sparse_spikes(model, previous_spikes);
        self.propagate_latent_spikes(model, previous_spikes);

        let mut new_spikes = vec![false; n_count];
        self.update_neuron_states(model, current_tick, &mut new_spikes);
        new_spikes
    }

    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, _history: &[Vec<bool>]) {
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

            let pre_tick = model.neurons.last_spike_tick[src] as u64;
            let post_tick = model.neurons.last_spike_tick[target] as u64;
            self.plasticity_rule.update_temporal(&mut model.synapses.weight[i], pre_tick, post_tick, current_tick as u64);
            self.plasticity_rule.update_contrastive(&mut model.synapses.weight[i], 0);
        }
    }

    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        prune_synapses(&mut model.synapses, self.structural_config.prune_threshold);

        // Structure changed -> Index must be rebuilt next tick
        self.synapse_index.clear();

        // SNNaS: Evolutionary mutation
        if let Some(r) = reward {
            self.optimizer.mutate_with_activity(&mut model.synapses, model.neurons.len(), r, history);

            if r > model.config.neurogenesis_reward_threshold && model.neurons.len() < model.config.max_synapses {
                model.neurons.grow((model.neurons.len() / 20).max(1));
            }

            if model.config.metaplasticity_enabled {
                if r.abs() < 10 {
                    model.config.learning_rate = (model.config.learning_rate as i64 * 900 >> 10) as i32;
                } else if r.abs() > 500 {
                    model.config.learning_rate = (model.config.learning_rate as i64 * 1100 >> 10) as i32;
                }
            }
        }

        // Local Homeostatic Scaling
        let n_count = model.neurons.len();
        let mut sum_weights = vec![0i64; n_count];
        for i in 0..model.synapses.len() {
            let target = model.synapses.target_index[i] as usize;
            sum_weights[target] += model.synapses.weight[i].abs() as i64;
        }
        let max_neuron_sum = 1024 * 16;
        for i in 0..model.synapses.len() {
            let target = model.synapses.target_index[i] as usize;
            if sum_weights[target] > max_neuron_sum {
                let scale_factor = (max_neuron_sum << 10) / sum_weights[target];
                model.synapses.weight[i] = ((model.synapses.weight[i] as i64 * scale_factor) >> 10) as i32;
            }
        }
    }
}

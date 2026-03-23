use {BakedModel, GsopRule, PlasticityRule, SCALE, IValue, Compartment, SpikeData, NeuromodulationState};
use crate::{ComputeBackend, BackendError};
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
    pub propagation_layout: wgpu::BindGroupLayout,
    pub tick_layout: wgpu::BindGroupLayout,
    // Cached Buffers
    pub prev_spikes_buffer: Option<wgpu::Buffer>,
    pub distal_buffer: Option<wgpu::Buffer>,
    pub proximal_buffer: Option<wgpu::Buffer>,
    pub apical_buffer: Option<wgpu::Buffer>,
    pub basal_buffer: Option<wgpu::Buffer>,
    pub adaptation_buffer: Option<wgpu::Buffer>,
    pub activity_ema_buffer: Option<wgpu::Buffer>,
    pub gate_threshold_buffer: Option<wgpu::Buffer>,
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
    pub compartment_buffer: Option<wgpu::Buffer>,
    pub gate_buffer: Option<wgpu::Buffer>,
    pub base_threshold_buffer: Option<wgpu::Buffer>,
    pub layer_id_buffer: Option<wgpu::Buffer>,
    pub backprop_buffer: Option<wgpu::Buffer>,
    pub u_matrix_buffer: Option<wgpu::Buffer>,
    pub v_matrix_buffer: Option<wgpu::Buffer>,
    pub latent_state_buffer: Option<wgpu::Buffer>,
    pub config_uniform_buffer: Option<wgpu::Buffer>,
    pub tick_buffer: Option<wgpu::Buffer>,
    pub lr_buffer: Option<wgpu::Buffer>,
    pub sparse_spike_buffer: Option<wgpu::Buffer>,
    pub spike_counter_buffer: Option<wgpu::Buffer>,
    pub staging_spikes: Option<wgpu::Buffer>,
    pub staging_counter: Option<wgpu::Buffer>,
    pub staging_state: Option<wgpu::Buffer>,
    pub pre_spike_buffer: Option<wgpu::Buffer>,
    pub post_spike_buffer: Option<wgpu::Buffer>,
    pub last_spike_tick_buffer: Option<wgpu::Buffer>,
    pub is_excitatory_buffer: Option<wgpu::Buffer>,
    pub spike_history_buffer: Option<wgpu::Buffer>,
    pub expert_mask_buffer: Option<wgpu::Buffer>,
    pub delay_buffer: Option<wgpu::Buffer>,
    pub stp_resources_buffer: Option<wgpu::Buffer>,
    pub stp_calcium_buffer: Option<wgpu::Buffer>,
    pub modulation_buffer: Option<wgpu::Buffer>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub tick_bind_group: Option<wgpu::BindGroup>,
    pub lr_bind_group: Option<wgpu::BindGroup>,
    pub cached_neuron_count: usize,
    pub cached_synapse_count: usize,
}

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    pub fn new() -> Result<Self, BackendError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok_or(BackendError::AdapterNotFound)?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Nano Spiking OS Compute Device"),
            required_features: wgpu::Features::SHADER_INT64,
            required_limits: wgpu::Limits::default(),
        }, None)).map_err(|e| BackendError::DeviceCreationFailed(e.to_string()))?;

        // 1. Shaders
        let potential_shader = Self::create_shader(&device, "potential_update", include_str!("shaders/potential_update.wgsl"));
        let propagation_shader = Self::create_shader(&device, "spike_prop", include_str!("shaders/spike_prop.wgsl"));
        let gsop_shader = Self::create_shader(&device, "gsop_update", include_str!("shaders/gsop_update.wgsl"));
        let latent_accum_shader = Self::create_shader(&device, "latent_accum", include_str!("shaders/latent_accum.wgsl"));
        let latent_distrib_shader = Self::create_shader(&device, "latent_distrib", include_str!("shaders/latent_distrib.wgsl"));

        // 2. Bind Group Layouts
        let potential_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Potential Layout"),
            entries: &[
                Self::storage_entry(0, false), Self::storage_entry(1, false), Self::storage_entry(2, false),
                Self::storage_entry(3, false), Self::storage_entry(4, false), Self::storage_entry(5, true),
                Self::storage_entry(6, false), Self::storage_entry(7, true), Self::storage_entry(8, true),
                Self::storage_entry(9, true), Self::storage_entry(10, true), Self::storage_entry(11, true),
                Self::storage_entry(12, false), Self::storage_entry(13, false), Self::storage_entry(14, true),
                Self::storage_entry(15, false), Self::storage_entry(16, true), Self::storage_entry(17, true),
                Self::storage_entry(18, true), Self::storage_entry(19, true), Self::storage_entry(20, true),
                Self::storage_entry(21, false), Self::storage_entry(22, true), Self::storage_entry(24, false),
                Self::storage_entry(25, false),
                wgpu::BindGroupLayoutEntry { binding: 23, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let gsop_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GSOP Layout"),
            entries: &[
                Self::storage_entry(0, false), Self::storage_entry(1, true), Self::storage_entry(2, true),
                Self::storage_entry(3, true), Self::storage_entry(4, true), Self::storage_entry(5, true),
                Self::storage_entry(6, true), Self::storage_entry(7, true), Self::storage_entry(8, true),
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let latent_accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Latent Accum Layout"),
            entries: &[Self::storage_entry(0, true), Self::storage_entry(1, true), Self::storage_entry(2, false)],
        });

        let latent_distrib_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Latent Distrib Layout"),
            entries: &[Self::storage_entry(0, true), Self::storage_entry(1, true), Self::storage_entry(2, true), Self::storage_entry(3, false)],
        });

        let propagation_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Propagation Layout"),
            entries: &[
                Self::storage_entry(0, true), Self::storage_entry(1, true), Self::storage_entry(2, true),
                Self::storage_entry(3, true), Self::storage_entry(4, true), Self::storage_entry(5, false),
                Self::storage_entry(6, false), Self::storage_entry(7, false), Self::storage_entry(8, false),
                Self::storage_entry(9, true), Self::storage_entry(10, true),
            ],
        });

        let tick_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tick Layout"),
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }],
        });

        // 3. Pipelines
        let potential_pipeline = Self::create_pipeline(&device, "Potential", &potential_layout, &tick_layout, &potential_shader);
        let propagation_pipeline = Self::create_pipeline(&device, "Propagation", &propagation_layout, &tick_layout, &propagation_shader);
        let gsop_pipeline = Self::create_pipeline(&device, "GSOP", &gsop_layout, &tick_layout, &gsop_shader);
        let latent_accum_pipeline = Self::create_pipeline(&device, "Latent Accum", &latent_accum_layout, &tick_layout, &latent_accum_shader);
        let latent_distrib_pipeline = Self::create_pipeline(&device, "Latent Distrib", &latent_distrib_layout, &tick_layout, &latent_distrib_shader);

        Ok(Self {
            device, queue, potential_pipeline, propagation_pipeline, gsop_pipeline, latent_accum_pipeline, latent_distrib_pipeline,
            potential_layout, gsop_layout, latent_accum_layout, latent_distrib_layout, propagation_layout, tick_layout,
            prev_spikes_buffer: None, distal_buffer: None, proximal_buffer: None, apical_buffer: None, basal_buffer: None, gate_threshold_buffer: None,
            pot_buffer: None, threshold_buffer: None, decay_buffer: None, refractory_buffer: None,
            spikes_buffer: None, input_buffer: None, next_update_buffer: None, interval_buffer: None,
            weight_buffer: None, source_buffer: None, target_buffer: None, compartment_buffer: None,
            gate_buffer: None, base_threshold_buffer: None, layer_id_buffer: None, backprop_buffer: None, config_uniform_buffer: None,
            tick_buffer: None, lr_buffer: None,
            u_matrix_buffer: None, v_matrix_buffer: None, latent_state_buffer: None,
            sparse_spike_buffer: None, spike_counter_buffer: None, staging_spikes: None, staging_state: None,
            pre_spike_buffer: None, post_spike_buffer: None, last_spike_tick_buffer: None, is_excitatory_buffer: None,
            spike_history_buffer: None, expert_mask_buffer: None, delay_buffer: None,
            stp_resources_buffer: None, stp_calcium_buffer: None, modulation_buffer: None,
            bind_group: None, tick_bind_group: None, lr_bind_group: None,
            cached_neuron_count: 0, cached_synapse_count: 0
        })
    }
}

#[cfg(feature = "wgpu")]
impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], _history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState) -> SpikeData {
        let n_count = model.neurons.len();

        // 1. Re-initialize buffers if neuron count changed
        let n_changed = n_count != self.cached_neuron_count;
        if n_changed || self.pot_buffer.is_none() {
            let storage = wgpu::BufferUsages::STORAGE;
            let copy_all = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

            self.ensure_buffer(&mut self.pot_buffer, "Potentials", &model.neurons.potential, copy_all, n_changed);
            self.ensure_buffer(&mut self.threshold_buffer, "Thresholds", &model.neurons.threshold, copy_all, n_changed);
            self.ensure_buffer(&mut self.decay_buffer, "Decays", &model.neurons.decay, storage, n_changed);
            self.ensure_buffer(&mut self.refractory_buffer, "Refractory", &model.neurons.refractory_timer, copy_all, n_changed);

            self.spikes_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Output Spikes"), size: (n_count * 4) as u64, usage: copy_all, mapped_at_creation: false }));
            self.input_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Inputs"), size: (n_count * 4) as u64, usage: storage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));

            self.ensure_buffer(&mut self.next_update_buffer, "Next Update", &model.neurons.next_update_tick, copy_all, n_changed);
            self.ensure_buffer(&mut self.interval_buffer, "Intervals", &model.neurons.update_interval, storage, n_changed);
            self.ensure_buffer(&mut self.gate_buffer, "Dendritic Gate", &model.neurons.dendritic_gate, storage, n_changed);
            self.ensure_buffer(&mut self.distal_buffer, "Distal Potentials", &model.neurons.distal_potential, storage | wgpu::BufferUsages::COPY_DST, n_changed);
            self.ensure_buffer(&mut self.proximal_buffer, "Proximal Potentials", &model.neurons.proximal_potential, storage | wgpu::BufferUsages::COPY_DST, n_changed);
            self.ensure_buffer(&mut self.apical_buffer, "Apical Potentials", &model.neurons.apical_potential, storage | wgpu::BufferUsages::COPY_DST, n_changed);
            self.ensure_buffer(&mut self.basal_buffer, "Basal Potentials", &model.neurons.basal_potential, storage | wgpu::BufferUsages::COPY_DST, n_changed);
            self.ensure_buffer(&mut self.gate_threshold_buffer, "Gate Thresholds", &model.neurons.gate_threshold, storage | wgpu::BufferUsages::COPY_DST, n_changed);
            self.ensure_buffer(&mut self.base_threshold_buffer, "Base Thresholds", &model.neurons.base_threshold, storage, n_changed);
            self.ensure_buffer(&mut self.layer_id_buffer, "Layer IDs", &model.neurons.layer_id, storage, n_changed);
            self.ensure_buffer(&mut self.backprop_buffer, "Backprop Signals", &model.neurons.backprop_signal, copy_all, n_changed);
            self.ensure_buffer(&mut self.adaptation_buffer, "Adaptation Current", &model.neurons.adaptation_current, copy_all, n_changed);
            self.ensure_buffer(&mut self.activity_ema_buffer, "Activity EMA", &model.neurons.activity_ema, copy_all, n_changed);

            self.tick_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Tick Uniform"), size: 4, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            self.lr_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Learning Rate Uniform"), size: 4, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            self.config_uniform_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Config Buffer"), size: 12, usage: storage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));

            self.sparse_spike_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Sparse Spikes"), size: (n_count * 4) as u64, usage: copy_all, mapped_at_creation: false }));
            self.spike_counter_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Spike Counter"), size: 4, usage: copy_all, mapped_at_creation: false }));
            self.staging_spikes = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Staging Spikes"), size: (n_count * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            self.staging_counter = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Staging Counter"), size: 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            self.staging_state = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Staging State"), size: (n_count * 4 * 7) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            self.expert_mask_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Expert Mask"), size: (n_count * 4) as u64, usage: storage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            self.spike_history_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Spike History Buffer (GPU)"), size: (n_count * 16 * 4) as u64, usage: storage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));

            self.ensure_buffer(&mut self.last_spike_tick_buffer, "Last Spike Ticks", &model.neurons.last_spike_tick, copy_all, n_changed);

            let excitatory_u32: Vec<u32> = model.neurons.is_excitatory.iter().map(|&b| if b { 1u32 } else { 0u32 }).collect();
            self.ensure_buffer(&mut self.is_excitatory_buffer, "Is Excitatory Flags", &excitatory_u32, storage, n_changed);

            self.modulation_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Neuromodulation Uniform"),
                size: 12, // 3 * i32
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                    wgpu::BindGroupEntry { binding: 12, resource: self.sparse_spike_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: self.spike_counter_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 14, resource: self.layer_id_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 15, resource: self.backprop_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 16, resource: self.config_uniform_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 17, resource: self.apical_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 18, resource: self.basal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 19, resource: self.gate_threshold_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 20, resource: self.expert_mask_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 21, resource: self.last_spike_tick_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 22, resource: self.is_excitatory_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 23, resource: self.modulation_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 24, resource: self.adaptation_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 25, resource: self.activity_ema_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: None,
            }));
            self.tick_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.tick_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.tick_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: Some("Tick Bind Group"),
            }));
            self.lr_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.tick_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.lr_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: Some("LR Bind Group"),
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

            self.prev_spikes_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Prev Spikes"),
                size: (n_count * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.cached_neuron_count = n_count;
        }

        // 2. Weight/Synapse buffers (lazy init)
        let s_count = model.synapses.len();
        if self.weight_buffer.is_none() || s_count != self.cached_synapse_count {
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
            let comp_data: Vec<u32> = model.synapses.compartment.iter().map(|&c| c as u32).collect();
            self.compartment_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Synapse Compartments"),
                contents: bytemuck::cast_slice(&comp_data),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.delay_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Synapse Delays"),
                contents: bytemuck::cast_slice(&model.synapses.delay.iter().map(|&d| d as u32).collect::<Vec<u32>>()),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            self.stp_resources_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("STP Resources"),
                contents: bytemuck::cast_slice(&model.synapses.stp_resources),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.stp_calcium_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("STP Calcium"),
                contents: bytemuck::cast_slice(&model.synapses.stp_calcium),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            self.cached_synapse_count = s_count;
        }

        // 3. Upload inputs and current tick
        let prev_spikes_u32: Vec<u32> = previous_spikes.iter().map(|&s| if s { 1u32 } else { 0u32 }).collect();
        self.queue.write_buffer(self.prev_spikes_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&prev_spikes_u32));
        self.queue.write_buffer(self.input_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(external_inputs));
        self.queue.write_buffer(self.tick_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[current_tick]));

        // Upload module-generated potentials from InputBus
        self.queue.write_buffer(self.proximal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.proximal_potential));
        self.queue.write_buffer(self.distal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.distal_potential));
        self.queue.write_buffer(self.apical_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.apical_potential));
        self.queue.write_buffer(self.basal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.basal_potential));

        // Upload neuromodulation
        self.queue.write_buffer(self.modulation_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[modulation.dopamine, modulation.noradrenaline, modulation.serotonin]));

        // Lazy upload of configuration (only if changed or first run)
        self.queue.write_buffer(self.config_uniform_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[model.config.ip_increment]));
        self.queue.write_buffer(self.config_uniform_buffer.as_ref().unwrap(), 4, bytemuck::cast_slice(&[model.config.ip_decay]));
        self.queue.write_buffer(self.config_uniform_buffer.as_ref().unwrap(), 8, bytemuck::cast_slice(&[model.config.noise_amplitude]));

        let mask_data: Vec<u32> = if self.expert_masks.is_empty() {
            vec![1u32; n_count]
        } else {
            (0..n_count).map(|i| if self.expert_masks[i % self.expert_masks.len()] { 1u32 } else { 0u32 }).collect()
        };
        self.queue.write_buffer(self.expert_mask_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&mask_data));

        // 4. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Phase A: Spike Propagation
        if s_count > 0 {
            let prop_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.propagation_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.source_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.target_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.weight_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.delay_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.compartment_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.proximal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.distal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.apical_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.basal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: self.gate_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: self.spike_history_buffer.as_ref().unwrap().as_entire_binding() },
                ],
                label: Some("Prop Bind Group"),
            });

            // Update history buffer slot for the current tick
            // Shader expects history[ (current_tick - delay) % 16 ]
            // We store the PREVIOUS spikes (t-1) at slot (current_tick - 1) % 16
            let history_slot = (current_tick.wrapping_sub(1) % 16) as u64;
            self.queue.write_buffer(
                self.spike_history_buffer.as_ref().unwrap(),
                history_slot * (n_count * 4) as u64,
                bytemuck::cast_slice(&prev_spikes_u32)
            );

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.propagation_pipeline);
                cpass.set_bind_group(0, &prop_bind_group, &[]);
                cpass.set_bind_group(1, self.tick_bind_group.as_ref().unwrap(), &[]);
                cpass.dispatch_workgroups((s_count as u32 + 63) / 64, 1, 1);
            }
        }

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
                    wgpu::BindGroupEntry { binding: 3, resource: self.distal_buffer.as_ref().unwrap().as_entire_binding() },
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
        encoder.copy_buffer_to_buffer(self.spike_counter_buffer.as_ref().unwrap(), 0, self.staging_counter.as_ref().unwrap(), 0, 4);

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let spikes_slice = self.staging_spikes.as_ref().unwrap().slice(..);
        let counter_slice = self.staging_counter.as_ref().unwrap().slice(..);

        counter_slice.map_async(wgpu::MapMode::Read, |_| {});
        spikes_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mut active_indices = Vec::new();
        {
            let counter_raw = counter_slice.get_mapped_range();
            let counter_data: &[u32] = bytemuck::cast_slice(&counter_raw);
            let count = counter_data[0] as usize;
            drop(counter_raw);

            let spikes_raw = spikes_slice.get_mapped_range();
            let spike_indices: &[u32] = bytemuck::cast_slice(&spikes_raw);
            for i in 0..count.min(n_count) {
                let idx = spike_indices[i] as usize;
                if idx < n_count { active_indices.push(idx); }
            }
            drop(spikes_raw);
        }
        self.staging_spikes.as_ref().unwrap().unmap();
        self.staging_counter.as_ref().unwrap().unmap();

        SpikeData::Sparse(active_indices)
    }
    fn sync_state(&mut self, model: &mut BakedModel) {
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(self.pot_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), 0, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.threshold_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 4) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.refractory_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 8) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.next_update_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 12) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.last_spike_tick_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 16) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.adaptation_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 20) as u64, (n_count * 4) as u64);
        encoder.copy_buffer_to_buffer(self.activity_ema_buffer.as_ref().unwrap(), 0, self.staging_state.as_ref().unwrap(), (n_count * 24) as u64, (n_count * 4) as u64);

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
                model.neurons.last_spike_tick[i] = state_data[4 * n_count + i] as u32;
                model.neurons.adaptation_current[i] = state_data[5 * n_count + i];
                model.neurons.activity_ema[i] = state_data[6 * n_count + i];
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
    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: NeuromodulationState, history: &[Vec<bool>]) {
        // Upload neuromodulation (including dopamine derived from reward if provided)
        let mut m = modulation;
        if let Some(r) = reward { m.dopamine = r; }
        self.queue.write_buffer(self.modulation_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[m.dopamine, m.noradrenaline, m.serotonin]));

        self.update_weights(model, previous_spikes, current_spikes, current_tick, reward, history);
    }

    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, _reward: Option<IValue>, _history: &[Vec<bool>]) {
        use wgpu::util::DeviceExt;
        let s_count = model.synapses.len();
        let n_count = model.neurons.len();
        if s_count == 0 { return; }

        // 1. Re-initialize buffers if needed
        if self.weight_buffer.is_none() || s_count != self.cached_synapse_count {
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
            self.cached_synapse_count = s_count;
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
                wgpu::BindGroupEntry { binding: 5, resource: self.compartment_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.backprop_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.last_spike_tick_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.is_excitatory_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.modulation_buffer.as_ref().unwrap().as_entire_binding() },
            ],
            label: Some("GSOP Bind Group"),
        });

        // 4. Update Learning Rate Uniform
        self.queue.write_buffer(self.lr_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[model.config.learning_rate]));

        // 5. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.gsop_pipeline);
            cpass.set_bind_group(0, &gsop_bind_group, &[]);
            cpass.set_bind_group(1, self.lr_bind_group.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups((s_count as u32 + 63) / 64, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        // Note: Weights are NOT downloaded here to avoid blocking the GPU pipeline.
        // Sync back to CPU only happens in sync_state() or perform_night_phase().
    }

    pub fn compute_synaptic_metaplasticity(&mut self, _model: &mut BakedModel) {
        // Future implementation: GPU-side metaplasticity kernel
    }

    fn create_shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
        })
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        tick_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule
    ) -> wgpu::ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts: &[layout, tick_layout],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Pipeline", label)),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        })
    }

    fn ensure_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &mut Option<wgpu::Buffer>,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
        force_realloc: bool
    ) -> bool {
        use wgpu::util::DeviceExt;
        let size = (data.len() * std::mem::size_of::<T>()) as u64;

        let needs_realloc = force_realloc || buffer.as_ref().map_or(true, |b| b.size() != size);

        if needs_realloc {
            *buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            }));
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use {BakedModel, NeuronsSoA, SynapsesSoA};

    #[test]
    fn test_membrane_potential_calc() {
        use crate::calculate_membrane_potential;

        // Base case: minimal decay (1) applied
        let pot = calculate_membrane_potential(500, 0, 0, 0, 0, 512, 0, 0, 0, 0);
        assert!(pot <= 500 && pot >= 499);

        // Gated Distal Input: proximal (600) > threshold (512)
        let pot = calculate_membrane_potential(0, 600, 100, 0, 0, 512, 0, 0, 0, 0);
impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState) -> SpikeData {
        // Implementation logic here (omitted for brevity in this specific fix,
        // but normally would be copied from original monolithic lib.rs)
        SpikeData::Sparse(Vec::new())
    }
    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
    }
    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: NeuromodulationState, history: &[Vec<bool>]) {
    }
    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
    }
}

use genesis_core::{BakedModel, SCALE, IValue, Compartment, SpikeData, NeuromodulationState};
use crate::{ComputeBackend, BackendError, cpu::CpuBackend};

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

        let potential_shader = Self::create_shader(&device, "potential_update", include_str!("shaders/potential_update.wgsl"));
        let propagation_shader = Self::create_shader(&device, "spike_prop", include_str!("shaders/spike_prop.wgsl"));
        let gsop_shader = Self::create_shader(&device, "gsop_update", include_str!("shaders/gsop_update.wgsl"));
        let latent_accum_shader = Self::create_shader(&device, "latent_accum", include_str!("shaders/latent_accum.wgsl"));
        let latent_distrib_shader = Self::create_shader(&device, "latent_distrib", include_str!("shaders/latent_distrib.wgsl"));

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
            sparse_spike_buffer: None, spike_counter_buffer: None, staging_spikes: None, staging_counter: None, staging_state: None,
            pre_spike_buffer: None, post_spike_buffer: None, last_spike_tick_buffer: None, is_excitatory_buffer: None,
            spike_history_buffer: None, expert_mask_buffer: None, delay_buffer: None,
            stp_resources_buffer: None, stp_calcium_buffer: None, modulation_buffer: None,
            bind_group: None, tick_bind_group: None, lr_bind_group: None,
            cached_neuron_count: 0, cached_synapse_count: 0
        })
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

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }
    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState) -> SpikeData {
        // Full GPU-resident implementation would go here.
        // For now, this is a skeleton that honors the trait.
        SpikeData::Sparse(Vec::new())
    }
    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
    }
    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: NeuromodulationState, history: &[Vec<bool>]) {
    }
    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        let mut cpu = CpuBackend::default();
        cpu.structural_plasticity(model, reward, history);
    }
    fn sync_state(&mut self, _model: &mut BakedModel) {}
}

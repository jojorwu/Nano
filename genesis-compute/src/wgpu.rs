use genesis_core::{BakedModel, IValue, SpikeData, NeuromodulationState};
use crate::{ComputeBackend, BackendError, cpu::CpuBackend};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use bincode;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuNeuronState {
    pub potential: i32,
    pub threshold: i32,
    pub decay: i32,
    pub refractory: i32,
    pub next_update: u32,
    pub last_spike_tick: u32,
    pub backprop_signal: i32,
    pub adaptation: i32,
    pub activity_ema: i32,
    pub base_threshold: i32,
    pub distal_gate: i32,
    pub apical_gate: i32,
    pub basal_gate: i32,
    pub block_id: u32,
    pub action: i32,
    pub plasticity_gate: i32,
    pub is_remote: u32,
    pub origin_node_id: u32,
    pub specialization_score: f32,
    pub liquid_current: i32,
    pub update_interval: i32,
    pub packed_lo: u32,
    pub packed_hi: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Association {
    pub target: u32,
    pub weight: i32,
}

pub struct GpuResources {
    pub neuron_state_buffer: Option<wgpu::Buffer>,
    pub distal_buffer: Option<wgpu::Buffer>,
    pub proximal_buffer: Option<wgpu::Buffer>,
    pub apical_buffer: Option<wgpu::Buffer>,
    pub basal_buffer: Option<wgpu::Buffer>,
    pub spikes_buffer: Option<wgpu::Buffer>,
    pub sparse_spike_buffer: Option<wgpu::Buffer>,
    pub spike_counter_buffer: Option<wgpu::Buffer>,
    pub input_buffer: Option<wgpu::Buffer>,
    pub interval_buffer: Option<wgpu::Buffer>,
    pub gate_threshold_buffer: Option<wgpu::Buffer>,
    pub expert_mask_buffer: Option<wgpu::Buffer>,
    pub dendritic_gate_buffer: Option<wgpu::Buffer>,
    pub weight_buffer: Option<wgpu::Buffer>,
    pub source_buffer: Option<wgpu::Buffer>,
    pub target_buffer: Option<wgpu::Buffer>,
    pub compartment_buffer: Option<wgpu::Buffer>,
    pub pre_spike_buffer: Option<wgpu::Buffer>,
    pub post_spike_buffer: Option<wgpu::Buffer>,
    pub u_matrix_buffer: Option<wgpu::Buffer>,
    pub v_matrix_buffer: Option<wgpu::Buffer>,
    pub latent_state_buffer: Option<wgpu::Buffer>,
    pub config_uniform_buffer: Option<wgpu::Buffer>,
    pub tick_buffer: Option<wgpu::Buffer>,
    pub lr_buffer: Option<wgpu::Buffer>,
    pub staging_spikes: Option<wgpu::Buffer>,
    pub staging_counter: Option<wgpu::Buffer>,
    pub staging_state: Option<wgpu::Buffer>,
    pub staging_weights: Option<wgpu::Buffer>,
    pub is_excitatory_buffer: Option<wgpu::Buffer>,
    pub spike_history_buffer: Option<wgpu::Buffer>,
    pub delay_buffer: Option<wgpu::Buffer>,
    pub stp_resources_buffer: Option<wgpu::Buffer>,
    pub stp_calcium_buffer: Option<wgpu::Buffer>,
    pub modulation_buffer: Option<wgpu::Buffer>,
    pub block_id_buffer: Option<wgpu::Buffer>,
    pub block_attn_buffer: Option<wgpu::Buffer>,
    pub context_hash_buffer: Option<wgpu::Buffer>,
    pub lsh_table_l1_buffer: Option<wgpu::Buffer>,
    pub lsh_table_l2_buffer: Option<wgpu::Buffer>,
    pub projection_matrices_buffer: Option<wgpu::Buffer>,
    pub segment_potentials_buffer: Option<wgpu::Buffer>,
    pub segment_gates_buffer: Option<wgpu::Buffer>,
    pub top_down_modulation_buffer: Option<wgpu::Buffer>,
}

pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub potential_pipeline: wgpu::ComputePipeline,
    pub propagation_pipeline: wgpu::ComputePipeline,
    pub gsop_pipeline: wgpu::ComputePipeline,
    pub latent_accum_pipeline: wgpu::ComputePipeline,
    pub latent_distrib_pipeline: wgpu::ComputePipeline,
    pub titan_retrieval_pipeline: wgpu::ComputePipeline,
    pub potential_layout: wgpu::BindGroupLayout,
    pub gsop_layout: wgpu::BindGroupLayout,
    pub latent_accum_layout: wgpu::BindGroupLayout,
    pub latent_distrib_layout: wgpu::BindGroupLayout,
    pub propagation_layout: wgpu::BindGroupLayout,
    pub tick_layout: wgpu::BindGroupLayout,
    pub resources: GpuResources,
    pub bind_group: Option<wgpu::BindGroup>,
    pub tick_bind_group: Option<wgpu::BindGroup>,
    pub range_buffer: Option<wgpu::Buffer>,
    pub lr_bind_group: Option<wgpu::BindGroup>,
    pub cached_neuron_count: usize,
    pub cached_synapse_count: usize,
    pub state_dirty: bool,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, BackendError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok_or(BackendError::AdapterNotFound)?;

        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Nano Spiking OS Compute Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 16.max(limits.max_storage_buffers_per_shader_stage),
                ..wgpu::Limits::default()
            },
        }, None)).map_err(|e| BackendError::DeviceCreationFailed(e.to_string()))?;

        let potential_shader = Self::create_shader(&device, "potential_update", include_str!("shaders/potential_update.wgsl"));
        let propagation_shader = Self::create_shader(&device, "spike_prop", include_str!("shaders/spike_prop.wgsl"));
        let gsop_shader = Self::create_shader(&device, "gsop_update", include_str!("shaders/gsop_update.wgsl"));
        let latent_accum_shader = Self::create_shader(&device, "latent_accum", include_str!("shaders/latent_accum.wgsl"));
        let latent_distrib_shader = Self::create_shader(&device, "latent_distrib", include_str!("shaders/latent_distrib.wgsl"));
        let titan_shader = Self::create_shader(&device, "titan_retrieval", include_str!("shaders/titan_bit_retrieval.wgsl"));

        let potential_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Potential Layout"),
            entries: &[
                Self::storage_entry(0, false), // neuron_states (RW)
                Self::storage_entry(1, true),  // distal
                Self::storage_entry(2, true),  // proximal
                Self::storage_entry(3, true),  // apical
                Self::storage_entry(4, true),  // basal
                Self::storage_entry(5, true),  // config
                Self::storage_entry(6, false), // spikes (RW)
                Self::storage_entry(7, false), // sparse_spikes (RW)
                Self::storage_entry(8, false), // spike_counter (RW)
                Self::storage_entry(9, true),  // inputs
                Self::storage_entry(10, true), // intervals
                Self::storage_entry(11, true), // dendritic_gate
                Self::storage_entry(12, true), // gate_thresholds
                Self::storage_entry(13, true), // expert_mask
                Self::storage_entry(14, false), // segment_potentials (RW)
                Self::storage_entry(15, true),  // segment_gates
                Self::storage_entry(16, true),  // top_down_modulation
                wgpu::BindGroupLayoutEntry { binding: 23, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let gsop_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GSOP Layout"),
            entries: &[
                Self::storage_entry(0, false), // weights
                Self::storage_entry(1, true),  // source
                Self::storage_entry(2, true),  // target
                Self::storage_entry(3, true),  // pre
                Self::storage_entry(4, true),  // post
                Self::storage_entry(5, true),  // compartments
                Self::storage_entry(6, true),  // neuron_states
                Self::storage_entry(8, true),  // excitatory
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let latent_accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Latent Accum Layout"),
            entries: &[Self::storage_entry(0, true), Self::storage_entry(1, true), Self::storage_entry(2, false)],
        });

        let latent_accum_tick_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Latent Accum Tick Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
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
                Self::storage_entry(9, true), Self::storage_entry(10, true), Self::storage_entry(11, true),
                Self::storage_entry(12, true),
            ],
        });

        let tick_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tick Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let potential_pipeline = Self::create_pipeline(&device, "Potential", &potential_layout, &tick_layout, &potential_shader);
        let propagation_pipeline = Self::create_pipeline(&device, "Propagation", &propagation_layout, &tick_layout, &propagation_shader);
        let gsop_pipeline = Self::create_pipeline(&device, "GSOP", &gsop_layout, &tick_layout, &gsop_shader);
        let latent_accum_pipeline = Self::create_pipeline(&device, "Latent Accum", &latent_accum_layout, &latent_accum_tick_layout, &latent_accum_shader);
        let latent_distrib_pipeline = Self::create_pipeline(&device, "Latent Distrib", &latent_distrib_layout, &tick_layout, &latent_distrib_shader);

        let titan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Titan Layout"),
            entries: &[
                Self::storage_entry(0, true), // associations
                Self::storage_entry(1, true), // offsets
                Self::storage_entry(2, true), // spikes
                Self::storage_entry(3, true), // block_ids
                Self::storage_entry(4, false), // distal_potentials
                Self::storage_entry(5, true), // context_hashes
                Self::storage_entry(6, true), // lsh_table_l1
                Self::storage_entry(7, true), // lsh_table_l2
                Self::storage_entry(8, true), // projection_matrices
            ],
        });
        let titan_retrieval_pipeline = Self::create_pipeline(&device, "Titan Retrieval", &titan_layout, &tick_layout, &titan_shader);

        let resources = GpuResources {
            neuron_state_buffer: None, distal_buffer: None, proximal_buffer: None, apical_buffer: None, basal_buffer: None,
            spikes_buffer: None, sparse_spike_buffer: None, spike_counter_buffer: None, input_buffer: None,
            interval_buffer: None, gate_threshold_buffer: None, expert_mask_buffer: None, dendritic_gate_buffer: None,
            weight_buffer: None, source_buffer: None, target_buffer: None, compartment_buffer: None,
            pre_spike_buffer: None, post_spike_buffer: None, u_matrix_buffer: None, v_matrix_buffer: None,
            latent_state_buffer: None, config_uniform_buffer: None, tick_buffer: None, lr_buffer: None,
            staging_spikes: None, staging_counter: None, staging_state: None, staging_weights: None, is_excitatory_buffer: None,
            spike_history_buffer: None, delay_buffer: None, stp_resources_buffer: None, stp_calcium_buffer: None,
            modulation_buffer: None,
            block_id_buffer: None,
            block_attn_buffer: None,
            context_hash_buffer: None,
            lsh_table_l1_buffer: None,
            lsh_table_l2_buffer: None,
            projection_matrices_buffer: None,
            segment_potentials_buffer: None,
            segment_gates_buffer: None,
            top_down_modulation_buffer: None,
        };

        Ok(Self {
            device, queue, potential_pipeline, propagation_pipeline, gsop_pipeline, latent_accum_pipeline, latent_distrib_pipeline,
            titan_retrieval_pipeline,
            potential_layout, gsop_layout, latent_accum_layout, latent_distrib_layout, propagation_layout, tick_layout,
            resources,
            bind_group: None, tick_bind_group: None, range_buffer: None, lr_bind_group: None,
            cached_neuron_count: 0, cached_synapse_count: 0,
            state_dirty: true,
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
        device: &wgpu::Device,
        buffer: &mut Option<wgpu::Buffer>,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
        force_realloc: bool
    ) -> bool {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let needs_realloc = force_realloc || buffer.as_ref().map_or(true, |b| b.size() < size);
        if needs_realloc {
            *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            }));
        }
        true
    }

}


impl WgpuBackend {
    fn day_phase_impl(&mut self, model: &mut BakedModel, external_inputs: &[i32], _previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState, range: std::ops::Range<usize>, top_down_modulation: Option<&[i32]>) -> SpikeData {
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();
        if n_count == 0 { return SpikeData::Sparse(Vec::new()); }

        let res = &mut self.resources;

        // 1. Prepare buffers (Device-Resident: only upload if dirty or size changed)
        let needs_upload = self.state_dirty || self.cached_neuron_count != n_count;
        if needs_upload {
            let mut states = Vec::with_capacity(n_count);
            for i in 0..n_count {
                states.push(GpuNeuronState {
                    potential: model.neurons.potential[i],
                    threshold: model.neurons.threshold[i],
                    decay: model.neurons.decay[i],
                    refractory: model.neurons.refractory_timer[i],
                    next_update: model.neurons.next_update_tick[i],
                    last_spike_tick: model.neurons.last_spike_tick[i],
                    backprop_signal: model.neurons.backprop_signal[i],
                    adaptation: model.neurons.adaptation_current[i],
                    activity_ema: model.neurons.activity_ema[i],
                    base_threshold: model.neurons.base_threshold[i],
                    distal_gate: model.neurons.distal_gate[i],
                    apical_gate: model.neurons.apical_gate[i],
                    basal_gate: model.neurons.basal_gate[i],
                    block_id: model.neurons.block_id[i],
                    action: model.neurons.action_potential[i],
                    packed_lo: (model.neurons.packed_params[i] & 0xFFFFFFFF) as u32,
                    packed_hi: (model.neurons.packed_params[i] >> 32) as u32,
                    plasticity_gate: model.neurons.plasticity_gate[i],
                    is_remote: model.neurons.is_remote[i] as u32,
                    origin_node_id: model.neurons.origin_node_id[i],
                    specialization_score: model.neurons.specialization_score[i],
                    liquid_current: model.neurons.liquid_current[i],
                    update_interval: model.neurons.update_interval[i] as i32,
                });
            }
            Self::ensure_buffer(&self.device, &mut res.neuron_state_buffer, "Neuron State", &states, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, false);
            self.queue.write_buffer(res.neuron_state_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&states));
            self.state_dirty = false;
            self.cached_neuron_count = n_count;
        }

        // History buffer: Only upload if we are not maintaining it on-device
        let packed_history_len = (n_count + 63) / 64;
        let mut packed_history = vec![0u64; packed_history_len * 16];
        for (t, step) in history.iter().take(16).enumerate() {
            for (i, &fired) in step.iter().enumerate() {
                if fired {
                    packed_history[t * packed_history_len + (i / 64)] |= 1 << (i % 64);
                }
            }
        }
        Self::ensure_buffer(&self.device, &mut res.spike_history_buffer, "Spike History", &packed_history, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
        self.queue.write_buffer(res.spike_history_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&packed_history));

        // Synapse buffers: Only upload if changed (cached_synapse_count)
        if s_count > 0 && self.cached_synapse_count != s_count {
            if Self::ensure_buffer(&self.device, &mut res.source_buffer, "Source", &model.synapses.source_index, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
                 self.queue.write_buffer(res.source_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.synapses.source_index));
            }
            if Self::ensure_buffer(&self.device, &mut res.target_buffer, "Target", &model.synapses.target_index, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
                 self.queue.write_buffer(res.target_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.synapses.target_index));
            }
            if Self::ensure_buffer(&self.device, &mut res.weight_buffer, "Weight", &model.synapses.weight, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
                 self.queue.write_buffer(res.weight_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.synapses.weight));
            }
            let comp_u32: Vec<u32> = model.synapses.compartment.iter().map(|&c| match c {
                genesis_core::Compartment::Proximal => 0,
                genesis_core::Compartment::Distal => 1,
                genesis_core::Compartment::Apical => 2,
                genesis_core::Compartment::Basal => 3,
                genesis_core::Compartment::Custom(v) => 4 + v as u32,
            }).collect();
            if Self::ensure_buffer(&self.device, &mut res.compartment_buffer, "Compartment", &comp_u32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
                 self.queue.write_buffer(res.compartment_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&comp_u32));
            }
            let delays_u32: Vec<u32> = model.synapses.delay.iter().map(|&d| d as u32).collect();
            if Self::ensure_buffer(&self.device, &mut res.delay_buffer, "Delay", &delays_u32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
                 self.queue.write_buffer(res.delay_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&delays_u32));
            }
            self.cached_synapse_count = s_count;
        }

        // Potential buffers (RW)
        let zeros = vec![0i32; n_count];
        Self::ensure_buffer(&self.device, &mut res.proximal_buffer, "Proximal", &zeros, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
        Self::ensure_buffer(&self.device, &mut res.distal_buffer, "Distal", &zeros, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
        Self::ensure_buffer(&self.device, &mut res.apical_buffer, "Apical", &zeros, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
        Self::ensure_buffer(&self.device, &mut res.basal_buffer, "Basal", &zeros, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
        self.queue.write_buffer(res.proximal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&zeros));
        self.queue.write_buffer(res.distal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&zeros));
        self.queue.write_buffer(res.apical_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&zeros));
        self.queue.write_buffer(res.basal_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&zeros));

        // Inputs & Config
        Self::ensure_buffer(&self.device, &mut res.input_buffer, "Input", external_inputs, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
        self.queue.write_buffer(res.input_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(external_inputs));

        if Self::ensure_buffer(&self.device, &mut res.dendritic_gate_buffer, "Dendritic Gate", &model.neurons.dendritic_gate, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
            self.queue.write_buffer(res.dendritic_gate_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.dendritic_gate));
        }
        if Self::ensure_buffer(&self.device, &mut res.gate_threshold_buffer, "Gate Threshold", &model.neurons.gate_threshold, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
            self.queue.write_buffer(res.gate_threshold_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.gate_threshold));
        }

        // Upload segment potentials and gates for Multi-Segment Dendrites
        Self::ensure_buffer(&self.device, &mut res.segment_potentials_buffer, "Segment Potentials", &model.neurons.segment_potentials, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
        self.queue.write_buffer(res.segment_potentials_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.segment_potentials));

        if Self::ensure_buffer(&self.device, &mut res.segment_gates_buffer, "Segment Gates", &model.neurons.segment_gates, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false) {
            self.queue.write_buffer(res.segment_gates_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&model.neurons.segment_gates));
        }

        // Top-down modulation from HierarchicalModule (Pre-calculated in InputStage/KernelContext)
        if let Some(td_data) = top_down_modulation {
            Self::ensure_buffer(&self.device, &mut res.top_down_modulation_buffer, "Top Down Modulation", td_data, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
            self.queue.write_buffer(res.top_down_modulation_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(td_data));
        } else {
            let zeros = vec![0i32; n_count];
            Self::ensure_buffer(&self.device, &mut res.top_down_modulation_buffer, "Top Down Modulation", &zeros, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
            self.queue.write_buffer(res.top_down_modulation_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&zeros));
        }

        let config_data = [
             model.config.physics.noise_amplitude,
             model.config.physics.theta_rhythm as i32,
             model.config.plasticity.intrinsic_learning_rate as i32,
             0, 0, 0, 0, 0
        ];
        Self::ensure_buffer(&self.device, &mut res.config_uniform_buffer, "Config", &config_data, wgpu::BufferUsages::STORAGE, false);

        let tick_data = [current_tick];
        Self::ensure_buffer(&self.device, &mut res.tick_buffer, "Tick", &tick_data, wgpu::BufferUsages::UNIFORM, false);

        let mod_data = [modulation.dopamine, modulation.noradrenaline, modulation.serotonin, 0];
        Self::ensure_buffer(&self.device, &mut res.modulation_buffer, "Modulation", &mod_data, wgpu::BufferUsages::UNIFORM, false);

        // Result buffers
        let spikes_zeros = vec![0u32; n_count];
        Self::ensure_buffer(&self.device, &mut res.spikes_buffer, "Spikes", &spikes_zeros, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, false);
        let counter_init = [0u32];
        Self::ensure_buffer(&self.device, &mut res.spike_counter_buffer, "Counter", &counter_init, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, false);
        Self::ensure_buffer(&self.device, &mut res.sparse_spike_buffer, "Sparse Spikes", &spikes_zeros, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, false);

        // 2. Dispatch Propagation
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Day Phase Encoder") });
        if s_count > 0 {
            // Prepare Block IDs and Block Attention Gates for Propagation
            Self::ensure_buffer(&self.device, &mut res.block_id_buffer, "Block ID", &model.neurons.block_id, wgpu::BufferUsages::STORAGE, false);

            let mut block_attn_vec = Vec::new();
            for m in model.module_states.values() {
                 if let Ok(attn_mod) = bincode::deserialize::<genesis_core::AttnResModule>(m) {
                     for q in &attn_mod.block_queries {
                         block_attn_vec.push([q[0], q[1], q[2], q[3]]);
                     }
                 }
            }
            if block_attn_vec.is_empty() {
                block_attn_vec.push([1024, 1024, 1024, 1024]);
            }
            Self::ensure_buffer(&self.device, &mut res.block_attn_buffer, "Block Attn", &block_attn_vec, wgpu::BufferUsages::STORAGE, false);

            let prop_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Prop BG"),
                layout: &self.propagation_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: res.source_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: res.target_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: res.weight_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: res.delay_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: res.compartment_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: res.proximal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: res.distal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: res.apical_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: res.basal_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: res.dendritic_gate_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: res.spike_history_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: res.block_id_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: res.block_attn_buffer.as_ref().unwrap().as_entire_binding() },
                ],
            });

            let range_data = [range.start as u32, range.end as u32];
            Self::ensure_buffer(&self.device, &mut self.range_buffer, "Range Buffer", &range_data, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, false);
            self.queue.write_buffer(self.range_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&range_data));

            let tick_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tick BG"),
                layout: &self.tick_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: res.tick_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.range_buffer.as_ref().unwrap().as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Propagation"), timestamp_writes: None });
            pass.set_pipeline(&self.propagation_pipeline);
            pass.set_bind_group(0, &prop_bg, &[]);
            pass.set_bind_group(1, &tick_bg, &[]);
            pass.dispatch_workgroups((s_count as u32 + 63) / 64, 1, 1);
        }

        // 3. Dispatch Titan Retrieval (Sparse Associative Memory)
        if let Some(titan_state) = model.module_states.get("titan").and_then(|s| bincode::deserialize::<genesis_core::titan::BitWiseTitan>(s).ok()) {
            let mut packed_previous = vec![0u32; (n_count + 31) / 32];
            for (i, &s) in _previous_spikes.iter().enumerate() {
                if s { packed_previous[i / 32] |= 1 << (i % 32); }
            }
            Self::ensure_buffer(&self.device, &mut res.pre_spike_buffer, "Titan Spikes", &packed_previous, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, true);
            self.queue.write_buffer(res.pre_spike_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&packed_previous));

            let current_hash = genesis_core::titan::BitWiseTitan::compute_context_hash(&packed_history[..packed_history_len]);
            let hashes = [current_hash];
            Self::ensure_buffer(&self.device, &mut res.context_hash_buffer, "Context Hashes", &hashes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
            self.queue.write_buffer(res.context_hash_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&hashes));

            let max_bid = model.neurons.block_id.iter().max().copied().unwrap_or(0);
            let mut l1_sigs = vec![0u32; (max_bid + 1) as usize];
            let mut l2_sigs = vec![0u32; (max_bid + 1) as usize];

            for bid in 0..=max_bid {
                if let Some(&sig) = titan_state.block_signatures_l1.get(bid as usize) {
                    l1_sigs[bid as usize] = sig;
                }
                if let Some(&sig) = titan_state.block_signatures_l2.get(bid as usize) {
                    l2_sigs[bid as usize] = sig;
                }
            }
            Self::ensure_buffer(&self.device, &mut res.lsh_table_l1_buffer, "LSH L1", &l1_sigs, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
            Self::ensure_buffer(&self.device, &mut res.lsh_table_l2_buffer, "LSH L2", &l2_sigs, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
            self.queue.write_buffer(res.lsh_table_l1_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&l1_sigs));
            self.queue.write_buffer(res.lsh_table_l2_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&l2_sigs));

            let mut flat_matrices = Vec::new();
            for m in &titan_state.projection_matrices { flat_matrices.extend_from_slice(m); }
            Self::ensure_buffer(&self.device, &mut res.projection_matrices_buffer, "Proj Matrices", &flat_matrices, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, false);
            self.queue.write_buffer(res.projection_matrices_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&flat_matrices));

            let tick_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Titan Tick BG"),
                layout: &self.tick_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: res.tick_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.range_buffer.as_ref().unwrap().as_entire_binding() },
                ],
            });

            let mut all_assocs = Vec::new();
            let mut offsets = vec![0u32];

            for bid in 0..=max_bid {
                let start_off = titan_state.block_offsets.get(bid as usize).copied().unwrap_or(0) as usize;
                let count = titan_state.block_counts.get(bid as usize).copied().unwrap_or(0) as usize;
                if start_off + count <= titan_state.associations_flat.len() {
                    for a in &titan_state.associations_flat[start_off..start_off+count] {
                        all_assocs.push(Association { target: a.target, weight: a.weight as i32 });
                    }
                }
                offsets.push(all_assocs.len() as u32);
            }

            if !all_assocs.is_empty() {
                let mut titan_assocs_buffer = None;
                let mut titan_offsets_buffer = None;
                Self::ensure_buffer(&self.device, &mut titan_assocs_buffer, "Titan Assocs", &all_assocs, wgpu::BufferUsages::STORAGE, true);
                Self::ensure_buffer(&self.device, &mut titan_offsets_buffer, "Titan Offsets", &offsets, wgpu::BufferUsages::STORAGE, true);
                self.queue.write_buffer(titan_assocs_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&all_assocs));
                self.queue.write_buffer(titan_offsets_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&offsets));

                let titan_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Titan BG"),
                    layout: &self.titan_retrieval_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: titan_assocs_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: titan_offsets_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: res.pre_spike_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: res.block_id_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: res.distal_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: res.context_hash_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: res.lsh_table_l1_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: res.lsh_table_l2_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: res.projection_matrices_buffer.as_ref().unwrap().as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Titan Retrieval"), timestamp_writes: None });
                pass.set_pipeline(&self.titan_retrieval_pipeline);
                pass.set_bind_group(0, &titan_bg, &[]);
                pass.set_bind_group(1, &tick_bg, &[]);
                pass.dispatch_workgroups((n_count as u32 + 63) / 64, 1, 1);
            }
        }

        // 4. Dispatch Potential Update
        let pot_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pot BG"),
            layout: &self.potential_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: res.neuron_state_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: res.distal_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: res.proximal_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: res.apical_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: res.basal_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: res.config_uniform_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: res.spikes_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: res.sparse_spike_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: res.spike_counter_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: res.input_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: res.interval_buffer.as_ref().unwrap_or(&res.input_buffer.as_ref().unwrap()).as_entire_binding() }, // Placeholder
                wgpu::BindGroupEntry { binding: 11, resource: res.dendritic_gate_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: res.gate_threshold_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: res.expert_mask_buffer.as_ref().unwrap_or(&res.input_buffer.as_ref().unwrap()).as_entire_binding() }, // Placeholder
                wgpu::BindGroupEntry { binding: 14, resource: res.segment_potentials_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: res.segment_gates_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: res.top_down_modulation_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 23, resource: res.modulation_buffer.as_ref().unwrap().as_entire_binding() },
            ],
        });
        let tick_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tick BG"),
            layout: &self.tick_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: res.tick_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.range_buffer.as_ref().unwrap().as_entire_binding() },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Potential"), timestamp_writes: None });
            pass.set_pipeline(&self.potential_pipeline);
            pass.set_bind_group(0, &pot_bg, &[]);
            pass.set_bind_group(1, &tick_bg, &[]);
            let range_len = (range.end - range.start) as u32;
            pass.dispatch_workgroups((range_len + 63) / 64, 1, 1);
        }

        // 4. Read back results (Efficiency: only spikes/counter)
        Self::ensure_buffer(&self.device, &mut res.staging_spikes, "Staging Spikes", &vec![0u32; n_count], wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, false);
        encoder.copy_buffer_to_buffer(res.sparse_spike_buffer.as_ref().unwrap(), 0, res.staging_spikes.as_ref().unwrap(), 0, (n_count * 4) as u64);

        Self::ensure_buffer(&self.device, &mut res.staging_counter, "Staging Counter", &[0u32], wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, false);
        encoder.copy_buffer_to_buffer(res.spike_counter_buffer.as_ref().unwrap(), 0, res.staging_counter.as_ref().unwrap(), 0, 4);

        self.queue.submit(Some(encoder.finish()));

        // Polling and retrieval
        let (tx, rx) = std::sync::mpsc::channel();
        res.staging_counter.as_ref().unwrap().slice(..).map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let counter_data = res.staging_counter.as_ref().unwrap().slice(..).get_mapped_range();
        let spike_count = *bytemuck::from_bytes::<u32>(&counter_data) as usize;
        drop(counter_data);
        res.staging_counter.as_ref().unwrap().unmap();

        let (tx, rx) = std::sync::mpsc::channel();
        res.staging_spikes.as_ref().unwrap().slice(..).map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let spikes_data = res.staging_spikes.as_ref().unwrap().slice(..).get_mapped_range();
        let indices: &[u32] = bytemuck::cast_slice(&spikes_data);
        let result_indices = indices[..spike_count].iter().map(|&i| i as usize).collect();
        drop(spikes_data);
        res.staging_spikes.as_ref().unwrap().unmap();

        SpikeData::Sparse(result_indices)
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "WgpuBackend" }

    fn execute_kernel(&mut self, kernel: crate::SimulationKernel, model: &mut BakedModel, ctx: &crate::KernelContext) -> Option<SpikeData> {
        let n_count = model.neurons.len();
        self.execute_kernel_range(kernel, model, ctx, 0..n_count)
    }

    fn execute_kernel_range(&mut self, kernel: crate::SimulationKernel, model: &mut BakedModel, ctx: &crate::KernelContext, range: std::ops::Range<usize>) -> Option<SpikeData> {
        match kernel {
            crate::SimulationKernel::PropagateSynapses => None,
            crate::SimulationKernel::UpdateMembranePotentials => None,
            crate::SimulationKernel::GenerateSpikes => {
                Some(self.day_phase_impl(model, ctx.external_inputs, ctx.previous_spikes, ctx.history, ctx.current_tick, ctx.modulation, range, ctx.top_down_modulation))
            }
            crate::SimulationKernel::ApplyModulation(_) => None,
        }
    }

    fn day_phase(&mut self, model: &mut BakedModel, external_inputs: &[i32], previous_spikes: &[bool], history: &[Vec<bool>], current_tick: u32, modulation: NeuromodulationState) -> SpikeData {
        let n_count = model.neurons.len();
        self.day_phase_impl(model, external_inputs, previous_spikes, history, current_tick, modulation, 0..n_count, None)
    }

    fn update_weights(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, history: &[Vec<bool>]) {
        self.update_weights_modulated(model, previous_spikes, current_spikes, current_tick, reward, NeuromodulationState::default(), history);
    }

    fn update_weights_modulated(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], current_tick: u32, reward: Option<IValue>, modulation: NeuromodulationState, history: &[Vec<bool>]) {
        let s_count = model.synapses.len();
        self.update_weights_range(model, previous_spikes, current_spikes, current_tick, reward, modulation, history, 0..s_count)
    }

    fn update_weights_range(&mut self, model: &mut BakedModel, previous_spikes: &[bool], current_spikes: &[bool], _current_tick: u32, _reward: Option<IValue>, modulation: NeuromodulationState, _history: &[Vec<bool>], range: std::ops::Range<usize>) {
        let s_count = model.synapses.len();
        if s_count == 0 { return; }

        let res = &mut self.resources;

        // 1. Ensure buffers are ready and uploaded
        Self::ensure_buffer(&self.device, &mut res.weight_buffer, "Weight Buffer", &model.synapses.weight, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, false);
        Self::ensure_buffer(&self.device, &mut res.source_buffer, "Source Buffer", &model.synapses.source_index, wgpu::BufferUsages::STORAGE, false);
        Self::ensure_buffer(&self.device, &mut res.target_buffer, "Target Buffer", &model.synapses.target_index, wgpu::BufferUsages::STORAGE, false);
        let comp_u32: Vec<u32> = model.synapses.compartment.iter().map(|&c| match c {
            genesis_core::Compartment::Proximal => 0,
            genesis_core::Compartment::Distal => 1,
            genesis_core::Compartment::Apical => 2,
            genesis_core::Compartment::Basal => 3,
            genesis_core::Compartment::Custom(v) => 4 + v as u32,
        }).collect();
        Self::ensure_buffer(&self.device, &mut res.compartment_buffer, "Compartment Buffer", &comp_u32, wgpu::BufferUsages::STORAGE, false);

        // Prepare GpuNeuronState buffer
        let n_count = model.neurons.len();
        let mut states = Vec::with_capacity(n_count);
        for i in 0..n_count {
            states.push(GpuNeuronState {
                potential: model.neurons.potential[i],
                threshold: model.neurons.threshold[i],
                decay: model.neurons.decay[i],
                refractory: model.neurons.refractory_timer[i],
                next_update: model.neurons.next_update_tick[i],
                last_spike_tick: model.neurons.last_spike_tick[i],
                backprop_signal: model.neurons.backprop_signal[i],
                adaptation: model.neurons.adaptation_current[i],
                activity_ema: model.neurons.activity_ema[i],
                base_threshold: model.neurons.base_threshold[i],
                distal_gate: model.neurons.distal_gate[i],
                apical_gate: model.neurons.apical_gate[i],
                basal_gate: model.neurons.basal_gate[i],
                block_id: model.neurons.block_id[i],
                action: model.neurons.action_potential[i],
                packed_lo: (model.neurons.packed_params[i] & 0xFFFFFFFF) as u32,
                packed_hi: (model.neurons.packed_params[i] >> 32) as u32,
                plasticity_gate: model.neurons.plasticity_gate[i],
                is_remote: model.neurons.is_remote[i] as u32,
                origin_node_id: model.neurons.origin_node_id[i],
                specialization_score: model.neurons.specialization_score[i],
                liquid_current: model.neurons.liquid_current[i],
                update_interval: model.neurons.update_interval[i] as i32,
            });
        }

        let needs_realloc = res.neuron_state_buffer.is_none() || res.neuron_state_buffer.as_ref().unwrap().size() != (states.len() * std::mem::size_of::<GpuNeuronState>()) as u64;
        if needs_realloc {
            Self::ensure_buffer(&self.device, &mut res.neuron_state_buffer, "Neuron State Buffer", &states, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, true);
        } else {
            self.queue.write_buffer(res.neuron_state_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&states));
        }

        // Convert bool spikes to u32 for WGSL compatibility
        let pre_u32: Vec<u32> = previous_spikes.iter().map(|&s| if s { 1u32 } else { 0u32 }).collect();
        let post_u32: Vec<u32> = current_spikes.iter().map(|&s| if s { 1u32 } else { 0u32 }).collect();

        if res.pre_spike_buffer.as_ref().map_or(true, |b| b.size() != (pre_u32.len() * 4) as u64) {
            Self::ensure_buffer(&self.device, &mut res.pre_spike_buffer, "Pre Spike Buffer", &pre_u32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, true);
        } else {
            self.queue.write_buffer(res.pre_spike_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&pre_u32));
        }

        if res.post_spike_buffer.as_ref().map_or(true, |b| b.size() != (post_u32.len() * 4) as u64) {
            Self::ensure_buffer(&self.device, &mut res.post_spike_buffer, "Post Spike Buffer", &post_u32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, true);
        } else {
            self.queue.write_buffer(res.post_spike_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&post_u32));
        }

        // Upload modulation and learning rate using efficient write_buffer
        let mod_data = [modulation.dopamine, modulation.noradrenaline, modulation.serotonin, 0];
        if res.modulation_buffer.is_none() {
            Self::ensure_buffer(&self.device, &mut res.modulation_buffer, "Modulation Buffer", &mod_data, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, true);
        } else {
            self.queue.write_buffer(res.modulation_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&mod_data));
        }

        let lr_data = [model.config.plasticity.learning_rate];
        if res.lr_buffer.is_none() {
            Self::ensure_buffer(&self.device, &mut res.lr_buffer, "LR Buffer", &lr_data, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, true);
        } else {
            self.queue.write_buffer(res.lr_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&lr_data));
        }

        // 2. Create Bind Groups
        // Ensure excitatory buffer is ready and uploaded (cached)
        let excit_u32: Vec<u32> = model.synapses.source_index.iter().map(|&s| if model.neurons.is_excitatory[s as usize] != 0 { 1u32 } else { 0u32 }).collect();
        Self::ensure_buffer(&self.device, &mut res.is_excitatory_buffer, "Excitatory Buffer", &excit_u32, wgpu::BufferUsages::STORAGE, false);

        let gsop_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GSOP Bind Group"),
            layout: &self.gsop_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: res.weight_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: res.source_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: res.target_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: res.pre_spike_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: res.post_spike_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: res.compartment_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: res.neuron_state_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: res.is_excitatory_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: res.modulation_buffer.as_ref().unwrap().as_entire_binding() },
            ],
        });

        let range_data = [range.start as u32, range.end as u32];
        Self::ensure_buffer(&self.device, &mut self.range_buffer, "Range Buffer", &range_data, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, false);
        self.queue.write_buffer(self.range_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&range_data));

        let lr_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LR Bind Group"),
            layout: &self.tick_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: res.lr_buffer.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.range_buffer.as_ref().unwrap().as_entire_binding() },
            ],
        });

        // 3. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("GSOP Encoder") });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("GSOP Pass"), timestamp_writes: None });
            compute_pass.set_pipeline(&self.gsop_pipeline);
            compute_pass.set_bind_group(0, &gsop_bind_group, &[]);
            compute_pass.set_bind_group(1, &lr_bind_group, &[]);
            let range_len = (range.end - range.start) as u32;
            compute_pass.dispatch_workgroups((range_len + 63) / 64, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
    fn structural_plasticity(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>]) {
        self.structural_plasticity_with_surprise(model, reward, history, &[]);
    }

    fn structural_plasticity_with_surprise(&mut self, model: &mut BakedModel, reward: Option<IValue>, history: &[Vec<bool>], block_surprise: &[f32]) {
        let mut cpu = CpuBackend::default();
        cpu.structural_plasticity_with_surprise(model, reward, history, block_surprise);
    }
    fn sync_state(&mut self, model: &mut BakedModel) {
        let n_count = model.neurons.len();
        let s_count = model.synapses.len();
        let res = &mut self.resources;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Sync Encoder") });

        // 1. Sync Weights
        if let Some(ref buffer) = res.weight_buffer {
            let size = buffer.size();
            Self::ensure_buffer(&self.device, &mut res.staging_weights, "Staging Weights", &vec![0i32; s_count], wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, false);
            encoder.copy_buffer_to_buffer(buffer, 0, res.staging_weights.as_ref().unwrap(), 0, size);
        }

        // 2. Sync Neuron States
        if let Some(ref buffer) = res.neuron_state_buffer {
            let size = (n_count * std::mem::size_of::<GpuNeuronState>()) as u64;
            Self::ensure_buffer(&self.device, &mut res.staging_state, "Staging State", &vec![GpuNeuronState::zeroed(); n_count], wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, false);
            encoder.copy_buffer_to_buffer(buffer, 0, res.staging_state.as_ref().unwrap(), 0, size);
        }

        self.queue.submit(Some(encoder.finish()));

        // Polling and retrieval
        if let Some(ref staging) = res.staging_weights {
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = staging.slice(..).get_mapped_range();
            model.synapses.weight.copy_from_slice(bytemuck::cast_slice(&data));
            drop(data);
            staging.unmap();
        }

        if let Some(ref staging) = res.staging_state {
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = staging.slice(..).get_mapped_range();
            let out_states: &[GpuNeuronState] = bytemuck::cast_slice(&data);
            for i in 0..n_count {
                model.neurons.potential[i] = out_states[i].potential;
                model.neurons.threshold[i] = out_states[i].threshold;
                model.neurons.base_threshold[i] = out_states[i].base_threshold;
                model.neurons.refractory_timer[i] = out_states[i].refractory;
                model.neurons.last_spike_tick[i] = out_states[i].last_spike_tick;
                model.neurons.activity_ema[i] = out_states[i].activity_ema;
                model.neurons.adaptation_current[i] = out_states[i].adaptation;
                model.neurons.backprop_signal[i] = out_states[i].backprop_signal;
                model.neurons.action_potential[i] = out_states[i].action;
                model.neurons.specialization_score[i] = out_states[i].specialization_score;
            }
            drop(data);
            staging.unmap();
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_membrane_potential_calc() {
        use crate::kernels::calculate_membrane_potential;
        let pot = calculate_membrane_potential(500, 0, 0, 0, 0, 512, 0, 0, 0, 0);
        assert!(pot <= 500 && pot >= 499);
    }
}

use crate::{Runtime, SimulationSettings, SimulationObserver, RuntimeError, Observer, telemetry, persistence, engine::SimulationEngine};
use genesis_core::ModuleManager;

pub struct RuntimeBuilder {
    pub model_path: Option<String>,
    pub model_instance: Option<genesis_core::BakedModel>,
    pub settings: SimulationSettings,
    pub observers: Vec<Box<dyn SimulationObserver>>,
    pub peers: Vec<String>,
    pub node_id: Option<String>,
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        Self {
            model_path: None,
            model_instance: None,
            settings: SimulationSettings::default(),
            observers: Vec::new(),
            peers: Vec::new(),
            node_id: None,
        }
    }

    pub fn from_model(mut self, path: &str) -> Self {
        self.model_path = Some(path.to_string());
        self
    }

    pub fn from_model_direct(mut self, model: genesis_core::BakedModel) -> Self {
        self.model_instance = Some(model);
        self
    }

    pub fn with_settings(mut self, settings: SimulationSettings) -> Self {
        self.settings = settings;
        self
    }

    pub fn add_observer(mut self, observer: Box<dyn SimulationObserver>) -> Self {
        self.observers.push(observer);
        self
    }

    pub fn build(self) -> Result<Runtime, RuntimeError> {
        let model = if let Some(m) = self.model_instance {
             m
        } else {
             let path = self.model_path.ok_or_else(|| RuntimeError::ModelLoad("Model path not provided".into()))?;
             persistence::PersistenceManager::load(&path)?
        };
        let n_count = model.neurons.len();

        let backend_name = self.settings.preferred_backend.as_deref()
            .unwrap_or(&model.config.hardware.preferred_backend);

        let registry = genesis_compute::BackendRegistry::new();
        let backend = registry.create(backend_name)
            .or_else(|| {
                log::warn!("Backend '{}' not found, falling back to CPU", backend_name);
                registry.create("cpu")
            })
            .ok_or_else(|| RuntimeError::BackendCreationFailed(backend_name.to_string(), "Could not instantiate backend or CPU fallback".into()))?;

        let mut modules = ModuleManager::new();

        // Register model-specific factories (e.g. pre-initialized Titan from BakedModel)
        #[cfg(feature = "titan")]
        if let Some(ref titan) = model.titan_memory {
            let t = titan.clone();
            modules.register_factory("titan", move || Box::new(t.clone()));
        }

        // Instantiate modules based on model state
        for name in model.module_states.keys() {
            if modules.instantiate(name) {
                if let Some(state) = model.module_states.get(name) {
                    if let Some(m) = modules.modules.last_mut() {
                        m.set_state(state);
                    }
                }
            }
        }

        // Fallback for titan if not in module_states but in titan_memory (migration/legacy)
        #[cfg(feature = "titan")]
        if model.titan_memory.is_some() && !model.module_states.contains_key("titan") {
            modules.instantiate("titan");
        }

        let mut observers = self.observers;
        if observers.is_empty() {
            observers.push(Box::new(Observer::new(n_count)));
        }
        if !observers.iter_mut().any(|o| o.as_any_mut().is::<telemetry::Telemetry>()) {
            observers.push(Box::new(telemetry::Telemetry::default()));
        }

        modules.rebuild_tiers();

        let mut engine = SimulationEngine::new(model, modules, backend, &self.settings)
             .map_err(|e| RuntimeError::StateError("engine_init".into(), e.to_string()))?;

        // Automatic Heterogeneous setup: if primary is WGPU, set secondary to CPU
        if engine.backend.name().contains("Wgpu") {
            if let Some(secondary) = registry.create("cpu") {
                log::info!("Heterogeneous Engine: primary=GPU, secondary=CPU");
                engine.secondary_backend = Some(secondary);
            }
        }

        let (nm, titan_rx) = if !self.peers.is_empty() {
             let node_id = self.node_id.clone().unwrap_or_else(|| "node0".to_string());
             let remote_queue = engine.remote_spike_queue.clone();
             let (tx, rx) = tokio::sync::mpsc::channel(10);
             let nm = pollster::block_on(crate::network::NetworkManager::new(
                 node_id,
                 self.peers.clone(),
                 self.settings.distributed_port,
                 remote_queue,
                 Some(tx)
             )).map_err(|e| RuntimeError::NetworkError("node_init".into(), e.to_string()))?;
             (Some(std::sync::Arc::new(nm)), Some(rx))
        } else {
            (None, None)
        };

        let mut rt = Runtime {
            engine,
            settings: self.settings,
            episode_reward_history: Vec::new(),
            network_manager: nm,
            observers,
            last_surprise: 0,
            surprise_history: Vec::new(),
            titan_rx,
        };
        rt.post_init()?;
        Ok(rt)
    }
}

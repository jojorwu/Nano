use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimulationSettings {
    pub tick_rate_hz: Option<u32>,
    pub checkpoint_interval: u32,
    pub night_phase_interval: u32,
    pub distributed_port: u16,
    pub save_on_exit: bool,
    pub telemetry_enabled: bool,
    pub preferred_backend: Option<String>,
    pub active_pipeline_stages: Option<Vec<String>>,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            tick_rate_hz: None, // As fast as possible
            checkpoint_interval: 1000,
            night_phase_interval: 100,
            distributed_port: 8080,
            save_on_exit: true,
            telemetry_enabled: true,
            preferred_backend: None,
            active_pipeline_stages: None,
        }
    }
}

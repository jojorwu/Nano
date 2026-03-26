use crate::engine::SimulationEngine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalConfig {
    pub simulation: Option<crate::SimulationSettings>,
    pub network: Option<genesis_core::NetworkConfig>,
}

pub struct CommandProcessor;

impl CommandProcessor {
    fn handle_set_think(engine: &mut SimulationEngine, parts: &[&str]) -> String {
        if parts.len() < 3 { return "Usage: set_think <active|ticks> <val>".to_string(); }
        let val = match parts[1] {
            "active" => if parts[2] == "true" || parts[2] == "1" { 1 } else { 0 },
            "ticks" => match parts[2].parse::<i32>() {
                Ok(v) => v,
                Err(_) => return "Invalid numeric value for ticks".to_string(),
            },
            _ => return "Invalid sub-command. Use 'active' or 'ticks'".to_string(),
        };

        let input = genesis_core::ModuleInput::Control(parts[1].to_string(), val);
        let mut found = false;
        for m in &mut engine.modules.modules {
            if m.name() == "think" {
                m.handle_input(&input);
                found = true;
            }
        }
        if found { "Think settings updated".to_string() } else { "Think module not found".to_string() }
    }

    pub fn handle(engine: &mut SimulationEngine, cmd: &str) -> String {
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() { return "No command provided".to_string(); }

        match parts[0] {
            "help" => "Commands: set_lr <val>, set_think <active|ticks>, status, save, exit".to_string(),
            "set_lr" => {
                if parts.len() < 2 { return "Usage: set_lr <val>".to_string(); }
                if let Ok(lr) = parts[1].parse::<i32>() {
                    engine.model.config.plasticity.learning_rate = lr;
                    format!("Learning rate set to {}", lr)
                } else { "Invalid value".to_string() }
            },
            "set_think" => Self::handle_set_think(engine, &parts),
            "status" => {
                format!("Tick: {}, Synapses: {}, Neurons: {}", engine.state.tick_counter, engine.model.synapses.len(), engine.model.neurons.len())
            },
            _ => format!("Unknown command: {}", parts[0]),
        }
    }
}

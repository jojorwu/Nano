use crate::engine::SimulationEngine;

pub struct CommandProcessor;

impl CommandProcessor {
    pub fn handle(engine: &mut SimulationEngine, cmd: &str) -> String {
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() { return "No command provided".to_string(); }

        match parts[0] {
            "help" => "Commands: set_lr <val>, set_think <active|ticks>, status, save, exit".to_string(),
            "set_lr" => {
                if parts.len() < 2 { return "Usage: set_lr <val>".to_string(); }
                if let Ok(lr) = parts[1].parse::<i32>() {
                    engine.model.config.learning_rate = lr;
                    format!("Learning rate set to {}", lr)
                } else { "Invalid value".to_string() }
            },
            "set_think" => {
                if parts.len() < 3 { return "Usage: set_think <active|ticks> <val>".to_string(); }
                let mut found = false;
                let val = if parts[1] == "active" {
                    if parts[2] == "true" || parts[2] == "1" { 1 } else { 0 }
                } else if parts[1] == "ticks" {
                    match parts[2].parse::<i32>() {
                        Ok(v) => v,
                        Err(_) => return "Invalid numeric value for ticks".to_string(),
                    }
                } else {
                    return "Invalid sub-command. Use 'active' or 'ticks'".to_string();
                };

                let input = genesis_core::ModuleInput::Control(parts[1].to_string(), val);
                for m in &mut engine.modules.modules {
                    if m.name() == "think" {
                        m.handle_input(&input);
                        found = true;
                    }
                }
                if found { "Think settings updated".to_string() } else { "Think module not found".to_string() }
            },
            "status" => {
                format!("Tick: {}, Synapses: {}, Neurons: {}", engine.tick_counter, engine.model.synapses.len(), engine.model.neurons.len())
            },
            _ => format!("Unknown command: {}", parts[0]),
        }
    }
}

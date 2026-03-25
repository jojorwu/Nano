use clap::{Parser, Subcommand};
use serde::{Serialize, Deserialize};
use std::fs;
use genesis_baker::ModelBlueprint;
use genesis_node::{Runtime, SimulationSettings};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)] command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new Nano project with default config and blueprint
    Init,
    /// Bake a neural model from a blueprint TOML
    Bake {
        #[arg(short, long)] blueprint: String,
        #[arg(short, long, default_value = "model.state")] output: String,
        #[arg(long)] backend: Option<String>,
    },
    /// Run a multimodal simulation on a model
    Run {
        #[arg(short, long)] model: String,
        #[arg(short, long)] input: Option<String>,
        #[cfg(feature = "vision")]
        #[arg(short = 'g', long)] image: Option<String>,
        #[arg(short, long, default_value_t = false)] byte_level: bool,
        #[arg(short, long, default_value_t = 0)] reasoning: usize,
        #[arg(short, long)] learning_rate: Option<i32>,
        #[arg(long)] backend: Option<String>,
    },
    /// Run a Reinforcement Learning environment (Gym)
    Gym {
        #[arg(short, long)] model: String,
        #[arg(short, long, default_value = "cartpole")] env: String,
        #[arg(short = 'n', long, default_value_t = 100)] episodes: usize,
        #[arg(long)] backend: Option<String>,
    },
    /// Export a model and its configuration for distribution
    Export { #[arg(short, long)] model: String, #[arg(short, long)] name: String },
    /// Start an interactive shell for the model
    Shell {
        #[arg(short, long)] model: String,
        #[arg(long)] backend: Option<String>,
    },
    /// Benchmark simulation performance
    Bench {
        #[arg(short, long)] model: String,
        #[arg(short, long, default_value_t = 1000)] ticks: usize,
        #[arg(long)] backend: Option<String>,
    },
    /// Perform a health check on a model file
    Check {
        #[arg(short, long)] model: String,
    },
    /// Compare two models to see what was learned
    Diff {
        #[arg(short, long)] base: String,
        #[arg(short, long)] current: String,
    },
    /// Auto-tune simulation settings based on surprise
    Tune {
        #[arg(short, long)] model: String,
        #[arg(short, long)] input: String,
        #[arg(short, long, default_value_t = 100)] steps: usize,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct GlobalConfig {
    pub simulation: Option<genesis_node::SimulationSettings>,
    pub network: Option<genesis_core::NetworkConfig>,
}

struct SimulationSession {
    runtime: Runtime,
}

impl SimulationSession {
    fn load_global_config() -> GlobalConfig {
        fs::read_to_string("nano.toml")
            .ok()
            .and_then(|c| toml::from_str(&c).ok())
            .unwrap_or(GlobalConfig { simulation: None, network: None })
    }

    fn new(model_path: &str, lr_override: Option<i32>, backend_override: Option<String>) -> Result<Self> {
        let global = Self::load_global_config();
        let mut settings = global.simulation.unwrap_or_default();

        if let Some(b) = backend_override {
            settings.preferred_backend = Some(b);
        }

        let mut runtime = Runtime::load_with_settings(model_path, settings)
            .with_context(|| format!("Failed to load model from {}", model_path))?;

        runtime.post_init().map_err(|e| anyhow::anyhow!(e)).context("Failed to initialize modules")?;

        if let Some(net_cfg) = global.network {
            runtime.engine.model.config = net_cfg;
        }

        if let Some(lr) = lr_override {
            runtime.engine.model.config.learning_rate = lr;
        }
        Ok(Self { runtime })
    }

    fn run_multimodal(&mut self, text: Option<&str>, img_path: Option<&str>, byte_level: bool) -> Result<()> {
        if let Some(t) = text {
            println!("📝 Injecting Text: '{}' (Mode: {})", t, if byte_level { "Byte-Level" } else { "Modular" });
            self.runtime.inject_text(t);
        }

        #[cfg(feature = "vision")]
        if let Some(path) = img_path {
            println!("🖼️ Injecting Image: '{}'", path);
            let img = image::open(path).with_context(|| format!("Failed to open image at {}", path))?;
            let gray = img.to_luma8();
            self.runtime.inject_image(gray.as_raw());
        }

        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .context("Invalid progress bar template")?);
        pb.set_message("Processing signals...");

        // Execution loop: run until all transient inputs are processed
        loop {
            let spikes = self.runtime.tick(&vec![0; self.runtime.engine.model.neurons.len()]);
            let count = spikes.iter().filter(|&&s| s).count();

            let mut transient_active = false;
            for m in &self.runtime.engine.modules.modules {
                if m.name() == "text_processor" {
                    let state: genesis_core::text::TextProcessorModule = bincode::deserialize(&m.get_state())?;
                    if !state.last_tokens.is_empty() { transient_active = true; }
                }
            }

            pb.set_message(format!("Tick: {} spikes", count));
            if !transient_active { break; }
        }
        pb.finish_with_message("✅ Signal processing complete.");
        Ok(())
    }

    fn finish(&mut self, path: &str) -> Result<()> {
        self.runtime.sync_state();
        self.runtime.engine.model.save(path).with_context(|| format!("Failed to save model to {}", path))?;
        println!("✨ Simulation finished. State saved.");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match &cli.command {
        Commands::Init => {
            let config = r#"[simulation]
checkpoint_interval = 1000
night_phase_interval = 100
save_on_exit = true
preferred_backend = "cpu"

[network]
default_threshold = 1024
learning_rate = 10
neurogenesis_reward_threshold = 200
# Metaplasticity settings
metaplasticity_enabled = true
# Titan Memory settings
titan_surprise_threshold = 100
titan_decay_rate = 1
"#;
            fs::write("nano.toml", config).context("Failed to write nano.toml")?;

            let blueprint = r#"name = "MyFirstNano"
[architecture]
neuron_count = 1000
synapse_count = 5000

[[modules]]
type = "TextProcessor"
vocab_size = 1000
"#;
            fs::write("blueprint.toml", blueprint).context("Failed to write blueprint.toml")?;
            println!("✨ Nano environment initialized. Edit nano.toml and blueprint.toml, then run 'bake'.");
        }
        Commands::Bake { blueprint, output, backend } => {
            let content = fs::read_to_string(blueprint).with_context(|| format!("Failed to read blueprint: {}", blueprint))?;
            let mut bp: ModelBlueprint = toml::from_str(&content).with_context(|| "Invalid blueprint format")?;
            if let Some(b) = backend {
                if bp.config.is_none() { bp.config = Some(Default::default()); }
                if let Some(ref mut cfg) = bp.config { cfg.preferred_backend = b.clone(); }
            }
            let baked = bp.bake();
            baked.save(output).with_context(|| format!("Failed to save model to {}", output))?;
            println!("✅ Model '{}' baked to {} (Backend: {}).", bp.name, output, baked.config.preferred_backend);
        }
        Commands::Run { model, input, #[cfg(feature = "vision")] image, byte_level, reasoning: _, learning_rate, backend } => {
            let mut session = SimulationSession::new(model, *learning_rate, backend.clone())?;
            if let Err(e) = session.run_multimodal(input.as_deref(), image.as_deref(), *byte_level) {
                eprintln!("🔥 Runtime Error: {}. Attempting emergency backup...", e);
                session.finish(&format!("{}.bak", model))?;
                return Err(e);
            }
            session.finish(model)?;
        }
        Commands::Gym { model, env: env_name, episodes, backend } => {
            let settings = SimulationSettings {
                preferred_backend: backend.clone(),
                ..Default::default()
            };
            let mut runtime = Runtime::load_with_settings(model, settings).context("Failed to load model for Gym")?;
            #[cfg(feature = "rl")]
            {
                run_gym_commands(&mut runtime, env_name, *episodes);
            }
            runtime.engine.model.save(model).context("Failed to save model after Gym session")?;
        }
        Commands::Export { model, name } => {
            let dir = format!("models/{}", name);
            fs::create_dir_all(&dir).with_context(|| format!("Failed to create export directory: {}", dir))?;
            fs::copy(model, format!("{}/state.bin", dir)).context("Failed to copy model state")?;

            let config = fs::read_to_string("nano.toml").unwrap_or_default();
            fs::write(format!("{}/config.toml", dir), config).context("Failed to write exported config")?;

            let launcher = format!("#!/bin/bash\ncargo run -p nano-cli -- shell --model state.bin\n");
            fs::write(format!("{}/run.sh", dir), launcher).context("Failed to write exported launcher")?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(format!("{}/run.sh", dir)).context("Metadata error")?.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(format!("{}/run.sh", dir), perms).context("Failed to set perms")?;
            }

            println!("🚀 Model '{}' exported to {}. Use ./run.sh to start the interactive console.", name, dir);
        }
        Commands::Shell { model, backend } => {
            let mut session = SimulationSession::new(model, None, backend.clone())?;
            println!("🐚 Nano Interactive Shell (Backend: {})", session.runtime.engine.backend.name());
            println!("Commands: help, save, run <text>, load_image <path>, reload, consolidate <iters>, exit");

            use std::io::{Write, BufRead};
            let stdin = std::io::stdin();
            let mut stdout = std::io::stdout();

            print!("> ");
            let _ = stdout.flush();
            for line in stdin.lock().lines() {
                let l = line.context("Stdin error")?;
                let cmd = l.trim();
                if cmd == "exit" || cmd == "quit" { break; }
                if cmd == "save" { session.finish(model)?; }
                else if cmd.starts_with("run ") {
                    let input = &cmd[4..];
                    session.run_multimodal(Some(input), None, false)?;
                }
                else if cmd.starts_with("load_image ") {
                    let path = &cmd[11..];
                    session.run_multimodal(None, Some(path), false)?;
                }
                else if cmd == "reload" {
                    if let Err(e) = session.runtime.reload_settings("nano.toml") {
                        println!("!! Failed to reload config: {}", e);
                    } else {
                        println!("++ Settings reloaded from nano.toml");
                    }
                }
                else if cmd.starts_with("consolidate ") {
                    if let Ok(iters) = cmd[12..].parse::<u32>() {
                        println!("⏳ Consolidating memory ({} iterations)...", iters);
                        session.runtime.consolidate_memory(iters);
                        println!("✅ Consolidation complete.");
                    }
                }
                else {
                    let resp = session.runtime.handle_command(cmd);
                    println!(">> {}", resp);
                }
                print!("> ");
                let _ = stdout.flush();
            }
        }
        Commands::Bench { model, ticks, backend } => {
            let mut session = SimulationSession::new(model, None, backend.clone())?;
            println!("⏳ Benchmarking simulation ({} ticks, Backend: {})...", ticks, session.runtime.engine.backend.name());

            let pb = ProgressBar::new(*ticks as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .context("Progress bar error")?);

            let start = std::time::Instant::now();
            let n_count = session.runtime.engine.model.neurons.len();
            let empty_inputs = vec![0; n_count];

            for _ in 0..*ticks {
                session.runtime.tick(&empty_inputs);
                pb.inc(1);
            }

            let duration = start.elapsed();
            pb.finish_with_message("✅ Benchmark complete.");

            let tps = (*ticks as f64) / duration.as_secs_f64();
            println!("\nPerformance Results:");
            println!("  Total Ticks: {}", ticks);
            println!("  Total Time:  {:.2?}", duration);
            println!("  Throughput:  {:.2} ticks/sec (TPS)", tps);
        }
        Commands::Diff { base, current } => {
            println!("⚖️  Comparing base model '{}' with current model '{}'...", base, current);
            let b_baked = genesis_core::BakedModel::load(base).context("Failed to load base model")?;
            let c_baked = genesis_core::BakedModel::load(current).context("Failed to load current model")?;

            // 1. Synapse weight changes
            let mut total_delta = 0i64;
            let mut changed_count = 0;
            let s_len = b_baked.synapses.len().min(c_baked.synapses.len());
            for i in 0..s_len {
                let delta = (c_baked.synapses.weight[i] - b_baked.synapses.weight[i]).abs() as i64;
                if delta > 0 {
                    total_delta += delta;
                    changed_count += 1;
                }
            }
            println!("  - Synapses: {} changed, total weight delta = {}", changed_count, total_delta);

            // 2. Associative memory changes (Titan)
            if let (Some(b_titan), Some(c_titan)) = (&b_baked.titan_memory, &c_baked.titan_memory) {
                let b_count = b_titan.associations_flat.len();
                let c_count = c_titan.associations_flat.len();
                println!("  - Titan Associations: {} -> {} (delta: {})", b_count, c_count, c_count as i32 - b_count as i32);
            }

            // 3. Byte Memory changes
            if let (Some(b_titan), Some(c_titan)) = (&b_baked.titan_memory, &c_baked.titan_memory) {
                let mut bytes_changed = 0;
                let m_len = b_titan.byte_memory.len().min(c_titan.byte_memory.len());
                for i in 0..m_len {
                    if b_titan.byte_memory[i] != c_titan.byte_memory[i] {
                        bytes_changed += 1;
                    }
                }
                println!("  - Byte RAM (1MB): {} bytes modified", bytes_changed);
            }

            println!("✅ Knowledge comparison complete.");
        }
        Commands::Tune { model, input, steps } => {
            println!("🛠️  Auto-tuning model '{}' with input data...", model);
            let mut session = SimulationSession::new(model, None, None)?;

            let mut avg_surprise = 0f32;
            let mut best_lr = session.runtime.engine.model.config.learning_rate;
            let mut best_interval = session.runtime.settings.night_phase_interval;

            println!("  Initial State: LR={}, SleepInterval={}", best_lr, best_interval);

            // Tuning loop
            for step in 1..=*steps {
                session.runtime.inject_text(input);
                let _spikes = session.runtime.tick(&vec![0; session.runtime.engine.model.neurons.len()]);

                let surprise = session.runtime.last_surprise;
                avg_surprise = avg_surprise * 0.9 + surprise as f32 * 0.1;

                // Simple auto-tuning heuristic:
                // If surprise is consistently high (> 500), the model is struggling to learn or too unstable.
                if avg_surprise > 500.0 {
                    best_lr = (best_lr - 1).max(1);
                    best_interval = (best_interval - 5).max(10);
                } else if avg_surprise < 50.0 {
                    // If surprise is very low, we can increase LR to speed up learning.
                    best_lr = (best_lr + 1).min(100);
                    best_interval = (best_interval + 5).min(1000);
                }

                if step % 10 == 0 {
                    println!("    Step {}: AvgSurprise={:.2}, Suggesting LR={}, SleepInterval={}",
                        step, avg_surprise, best_lr, best_interval);
                }
            }

            session.runtime.engine.model.config.learning_rate = best_lr;
            session.runtime.settings.night_phase_interval = best_interval;

            println!("✅ Auto-tuning complete. Suggested settings applied.");
            session.finish(model)?;
        }
        Commands::Check { model } => {
            println!("🔍 Performing Health Check on model: {}...", model);
            let baked = genesis_core::BakedModel::load(model).context("Health Check Failed: Could not load model")?;

            println!("  - Neurons: {} (Structure: SoA)", baked.neurons.len());
            println!("  - Synapses: {}", baked.synapses.len());

            // Basic consistency check
            baked.neurons.validate().map_err(|e| anyhow::anyhow!(e)).context("Health Check Failed: Neuron state is inconsistent")?;

            // Check for NaN or infinite potentials (if applicable, but IValue is i32)

            // Check modules
            println!("  - Modules ({}):", baked.module_states.len());
            for name in baked.module_states.keys() {
                println!("    * {}", name);
            }

            if let Some(titan) = &baked.titan_memory {
                 println!("  - Titan Memory: {} associations, {} bytes buffer", titan.associations_flat.len(), titan.byte_memory.len());
            }

            println!("✅ Model Health Check PASSED.");
        }
    }
    Ok(())
}

#[cfg(feature = "rl")]
fn run_gym_commands(runtime: &mut Runtime, env_name: &str, episodes: usize) {
    let mut episode_rewards = Vec::new();

    if env_name == "arm" {
        let mut env = genesis_node::examples_rl::arm_sim::RobotArmEnv::new();
        run_gym_loop(runtime, &mut env, episodes, &mut episode_rewards);
    } else {
        let mut env = genesis_node::examples_rl::balance::SimpleBalanceEnv::new();
        run_gym_loop(runtime, &mut env, episodes, &mut episode_rewards);
    }
}

#[cfg(feature = "rl")]
fn run_gym_loop(runtime: &mut Runtime, env: &mut dyn genesis_core::rl::Environment, episodes: usize, history: &mut Vec<i32>) {
    use genesis_core::rl::RLAgent;
    let agent = RLAgent::new(env.observation_space(), env.action_space(), runtime.engine.model.neurons.len());

    for ep in 0..episodes {
        let mut obs = env.reset();
        let mut total_reward = 0;
        let mut done = false;
        while !done {
            let inputs = agent.encode_observation(&obs, runtime.engine.model.neurons.len());
            let spikes = runtime.tick_with_reward(&inputs, None);
            let actions = agent.decode_action(&spikes);
            let (next_obs, reward, is_done) = env.step(&actions);

            let mean = if history.is_empty() { 0 } else { history.iter().sum::<i32>() / history.len() as i32 };
            runtime.tick_with_reward(&vec![0; runtime.engine.model.neurons.len()], Some(reward - mean));

            total_reward += reward;
            obs = next_obs;
            done = is_done;
        }
        history.push(total_reward);
        if history.len() > 10 { history.remove(0); }
        if ep % 10 == 0 { println!("   Episode {}: Reward = {}", ep, total_reward); }
    }
}

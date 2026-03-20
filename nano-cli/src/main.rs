use clap::{Parser, Subcommand};
use serde::{Serialize, Deserialize};
use std::fs;
use genesis_baker::ModelBlueprint;
use genesis_node::Runtime;
#[cfg(feature = "text")]
use genesis_core::text::{SpikingTextModule, ByteSpikingModule};
#[cfg(feature = "vision")]
use genesis_core::vision::SpikingVisionModule;
#[cfg(feature = "vision")]
use image::GenericImageView;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)] command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Bake { #[arg(short, long)] blueprint: String, #[arg(short, long, default_value = "model.state")] output: String },
    Run {
        #[arg(short, long)] model: String,
        #[arg(short, long)] input: Option<String>,
        #[cfg(feature = "vision")]
        #[arg(short = 'g', long)] image: Option<String>,
        #[arg(short, long, default_value_t = false)] byte_level: bool,
        #[arg(short, long, default_value_t = 0)] reasoning: usize,
        #[arg(short, long)] learning_rate: Option<i32>,
    },
    Gym {
        #[arg(short, long)] model: String,
        #[arg(short, long, default_value = "cartpole")] env: String,
        #[arg(short, long, default_value_t = 100)] episodes: usize,
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
    fn new(model_path: &str, lr_override: Option<i32>) -> Self {
        let settings = if let Ok(content) = fs::read_to_string("nano.toml") {
            let global: GlobalConfig = toml::from_str(&content).unwrap_or_else(|_| GlobalConfig { simulation: None, network: None });
            global.simulation.unwrap_or_default()
        } else {
            genesis_node::SimulationSettings::default()
        };

        let mut runtime = Runtime::load_with_settings(model_path, settings).expect("Failed to load model");

        if let Ok(content) = fs::read_to_string("nano.toml") {
             if let Ok(global) = toml::from_str::<GlobalConfig>(&content) {
                 if let Some(net_cfg) = global.network {
                     runtime.model.config = net_cfg;
                 }
             }
        }

        if let Some(lr) = lr_override {
            runtime.model.config.learning_rate = lr;
        }
        Self { runtime }
    }

    fn run_text(&mut self, text: &str, byte_level: bool, reasoning: usize) {
        let mut combined_inputs = vec![0; self.runtime.model.neurons.len()];
        println!("📝 Text Input: '{}' (Mode: {})", text, if byte_level { "Byte-Level" } else { "Word-Based" });

        #[cfg(feature = "text")]
        {
            if byte_level {
                let pattern_len = (self.runtime.model.neurons.len() / 4).min(256).max(10);
                let patterns = ByteSpikingModule::encode_text(text, pattern_len);
                for (i, pattern) in patterns.iter().enumerate() {
                    for (j, &spiked) in pattern.iter().enumerate() {
                        if spiked { combined_inputs[j] = 1024; }
                    }
                    let mut spikes = self.runtime.tick(&combined_inputs);
                    for _ in 0..reasoning {
                        spikes = self.runtime.tick(&vec![0; self.runtime.model.neurons.len()]);
                    }
                    println!("   Byte {}: Generated {} spikes", text.as_bytes()[i] as char, spikes.iter().filter(|&&s| s).count());
                }
            } else {
                let tokens = {
                    let mut text_mod = SpikingTextModule::new(&mut self.runtime.model.vocabulary);
                    text_mod.tokenize(text)
                };
                for token in tokens {
                    let mut inputs = vec![0; self.runtime.model.neurons.len()];
                    let pattern_len = (self.runtime.model.neurons.len() / 4).min(256).max(10);
                    let text_mod = SpikingTextModule::new(&mut self.runtime.model.vocabulary);
                    let pattern = text_mod.encode(token, pattern_len);
                    for (i, &spiked) in pattern.iter().enumerate() {
                        if spiked { inputs[i] = 1024; }
                    }
                    let mut spikes = self.runtime.tick(&inputs);
                    for _ in 0..reasoning {
                        spikes = self.runtime.tick(&vec![0; self.runtime.model.neurons.len()]);
                    }
                    println!("   Token {}: Generated {} spikes", token, spikes.iter().filter(|&&s| s).count());
                }
            }
        }
    }

    #[cfg(feature = "vision")]
    fn run_image(&mut self, img_path: &str) {
        println!("🖼️ Image Input: '{}'", img_path);
        let img = image::open(img_path).expect("Failed to open image");
        let (w, h) = img.dimensions();
        let vision_mod = SpikingVisionModule::new(w, h);
        let gray = img.to_luma8();
        let pixel_potentials = vision_mod.rate_encode(gray.as_raw());
        let mut inputs = vec![0; self.runtime.model.neurons.len()];
        for (i, &pot) in pixel_potentials.iter().enumerate() {
            if i < inputs.len() { inputs[i] = pot; }
        }
        let spikes = self.runtime.tick(&inputs);
        println!("   Generated {} spikes from image", spikes.iter().filter(|&&s| s).count());
    }

    fn finish(&mut self, path: &str) {
        self.runtime.sync_state();
        self.runtime.model.save(path).expect("Failed to save model");
        println!("✨ Simulation finished. State saved.");
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let cli = Cli::parse();
    match &cli.command {
        Commands::Init => {
            let config = r#"[simulation]
checkpoint_interval = 1000
night_phase_interval = 100
save_on_exit = true

[network]
default_threshold = 1024
learning_rate = 10
neurogenesis_reward_threshold = 200
"#;
            fs::write("nano.toml", config).expect("Failed to write nano.toml");

            let blueprint = r#"name = "MyFirstNano"
[architecture]
neuron_count = 1000
synapse_count = 5000

[[modules]]
type = "TextProcessor"
vocab_size = 1000
"#;
            fs::write("blueprint.toml", blueprint).expect("Failed to write blueprint.toml");
            println!("✨ Nano environment initialized. Edit nano.toml and blueprint.toml, then run 'bake'.");
        }
        Commands::Bake { blueprint, output } => {
            let content = fs::read_to_string(blueprint).expect("Failed to read");
            let bp: ModelBlueprint = toml::from_str(&content).expect("Invalid");
            let baked = bp.bake();
            baked.save(output).expect("Failed to save");
            println!("✅ Model '{}' baked to {}.", bp.name, output);
        }
        Commands::Run { model, input, #[cfg(feature = "vision")] image, byte_level, reasoning, learning_rate } => {
            let mut session = SimulationSession::new(model, *learning_rate);

            if let Some(text) = input {
                session.run_text(text, *byte_level, *reasoning);
            }

            #[cfg(feature = "vision")]
            if let Some(img_path) = image {
                session.run_image(img_path);
            }

            session.finish(model);
        }
        Commands::Gym { model, env: env_name, episodes } => {
            let mut runtime = Runtime::load(model).expect("Failed to load");
            #[cfg(feature = "rl")]
            {
                run_gym_commands(&mut runtime, env_name, *episodes);
            }
            runtime.model.save(model).expect("Failed to save");
        }
    }
}

#[cfg(feature = "rl")]
fn run_gym_commands(runtime: &mut Runtime, env_name: &str, episodes: usize) {
    let mut episode_rewards = Vec::new();

    // Simple factory-like selection
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
    let agent = RLAgent::new(env.observation_space(), env.action_space(), runtime.model.neurons.len());

    for ep in 0..episodes {
        let mut obs = env.reset();
        let mut total_reward = 0;
        let mut done = false;
        while !done {
            let inputs = agent.encode_observation(&obs, runtime.model.neurons.len());
            let spikes = runtime.tick_with_reward(&inputs, None);
            let actions = agent.decode_action(&spikes);
            let (next_obs, reward, is_done) = env.step(&actions);

            let mean = if history.is_empty() { 0 } else { history.iter().sum::<i32>() / history.len() as i32 };
            runtime.tick_with_reward(&vec![0; runtime.model.neurons.len()], Some(reward - mean));

            total_reward += reward;
            obs = next_obs;
            done = is_done;
        }
        history.push(total_reward);
        if history.len() > 10 { history.remove(0); }
        if ep % 10 == 0 { println!("   Episode {}: Reward = {}", ep, total_reward); }
    }
}

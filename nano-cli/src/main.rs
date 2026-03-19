use clap::{Parser, Subcommand};
use std::fs;
use genesis_baker::ModelBlueprint;
use genesis_node::Runtime;
#[cfg(feature = "text")]
use genesis_core::text::SpikingTextModule;
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

#[tokio::main]
async fn main() {
    env_logger::init();
    let cli = Cli::parse();
    match &cli.command {
        Commands::Bake { blueprint, output } => {
            let content = fs::read_to_string(blueprint).expect("Failed to read");
            let bp: ModelBlueprint = toml::from_str(&content).expect("Invalid");
            let baked = bp.bake();
            baked.save(output).expect("Failed to save");
            println!("✅ Model '{}' baked to {}.", bp.name, output);
            println!("   Neurons: {}, Synapses: {}", baked.neurons.len(), baked.synapses.len());
        }
        Commands::Run { model, input, #[cfg(feature = "vision")] image, byte_level, reasoning, learning_rate } => {
            println!("🚀 Loading model: {}", model);
            let mut runtime = Runtime::load(model).expect("Failed to load model");

            if let Some(lr) = learning_rate {
                runtime.model.config.learning_rate = *lr;
                println!("   Overriding learning rate: {}", lr);
            }
            let initial_synapses = runtime.model.synapses.len();

            let mut combined_inputs = vec![0; runtime.model.neurons.len()];

            if let Some(text) = input {
                println!("📝 Text Input: '{}' (Mode: {})", text, if *byte_level { "Byte-Level" } else { "Word-Based" });
                #[cfg(feature = "text")]
                {
                    if *byte_level {
                        use genesis_core::text::ByteSpikingModule;
                        let pattern_len = (runtime.model.neurons.len() / 4).min(256).max(10);
                        let patterns = ByteSpikingModule::encode_text(text, pattern_len);
                        for (i, pattern) in patterns.iter().enumerate() {
                            for (j, &spiked) in pattern.iter().enumerate() {
                                if spiked { combined_inputs[j] = 1000; }
                            }
                            let mut byte_spikes = runtime.tick(&combined_inputs);
                            for _ in 0..*reasoning {
                                byte_spikes = runtime.tick(&vec![0; runtime.model.neurons.len()]);
                            }
                            println!("   Byte {}: Generated {} spikes", text.as_bytes()[i] as char, byte_spikes.iter().filter(|&&s| s).count());
                        }
                    } else {
                        let tokens = {
                            let mut text_mod = SpikingTextModule::new(&mut runtime.model.vocabulary);
                            text_mod.tokenize(text)
                        };
                        for token in tokens {
                            let mut inputs = vec![0; runtime.model.neurons.len()];
                            {
                                let text_mod = SpikingTextModule::new(&mut runtime.model.vocabulary);
                                let pattern_len = (runtime.model.neurons.len() / 4).min(256).max(10);
                                let pattern = text_mod.encode(token, pattern_len);
                                for (i, &spiked) in pattern.iter().enumerate() {
                                    if spiked { inputs[i] = 1000; }
                                }
                            }
                            let mut spikes_res = runtime.tick(&inputs);
                            for _ in 0..*reasoning {
                                spikes_res = runtime.tick(&vec![0; runtime.model.neurons.len()]);
                            }
                            println!("   Token {}: Generated {} spikes", token, spikes_res.iter().filter(|&&s| s).count());
                        }
                    }
                }
                #[cfg(not(feature = "text"))]
                {
                    println!("❌ Text processing is disabled in this build.");
                }
            }

            #[cfg(feature = "vision")]
            if let Some(img_path) = image {
                println!("🖼️ Image Input: '{}'", img_path);
                let img = image::open(img_path).expect("Failed to open image");
                let (w, h) = img.dimensions();
                println!("   Resolution: {}x{}", w, h);
                let vision_mod = SpikingVisionModule::new(w, h);
                let gray = img.to_luma8();
                let pixel_potentials = vision_mod.rate_encode(gray.as_raw());
                for (i, &pot) in pixel_potentials.iter().enumerate() {
                    if i < combined_inputs.len() { combined_inputs[i] = pot; }
                }
                let spikes = runtime.tick(&combined_inputs);
                println!("   Generated {} spikes from image", spikes.iter().filter(|&&s| s).count());
            }

            let final_synapses = runtime.model.synapses.len();
            println!("✨ Simulation finished.");
            println!("   Structural Evolution: {} -> {} synapses", initial_synapses, final_synapses);

            runtime.model.save(model).expect("Failed to auto-save model");
            println!("💾 Model state and vocabulary saved.");
        }
        Commands::Gym { model, env: env_name, episodes } => {
            println!("🏋️ Training in Gym: {}", env_name);
            let mut runtime = Runtime::load(model).expect("Failed to load");

            #[cfg(feature = "rl")]
            {
                use genesis_core::rl::{RLAgent, Environment};
                // For demonstration, use a hardcoded environment
                // In a real scenario, this would be a dynamic registry
                let mut env = examples_rl::SimpleBalanceEnv::new();
                let agent = RLAgent::new(env.observation_space(), env.action_space(), runtime.model.neurons.len());

                let mut episode_rewards = Vec::new();
                for ep in 0..*episodes {
                    let mut obs = env.reset();
                    let mut total_reward = 0;
                    let mut done = false;
                    while !done {
                        let inputs = agent.encode_observation(&obs, runtime.model.neurons.len());
                        let (next_obs, reward, is_done) = {
                            let spikes = runtime.tick_with_reward(&inputs, None);
                            let actions = agent.decode_action(&spikes);
                            env.step(&actions)
                        };

                        let mean_reward = if episode_rewards.is_empty() { 0 } else {
                            episode_rewards.iter().sum::<i32>() / episode_rewards.len() as i32
                        };
                        let relative_reward = reward - mean_reward;

                        runtime.tick_with_reward(&vec![0; runtime.model.neurons.len()], Some(relative_reward));
                        total_reward += reward;
                        obs = next_obs;
                        done = is_done;
                    }
                    if ep % 10 == 0 { println!("   Episode {}: Total Reward = {}", ep, total_reward); }
                    episode_rewards.push(total_reward);
                    if episode_rewards.len() > 10 { episode_rewards.remove(0); }
                }
            }
            runtime.model.save(model).expect("Failed to save trained state");
            println!("✨ Gym session finished. Evolution saved.");
        }
    }
}

#[cfg(feature = "rl")]
mod examples_rl {
    use genesis_core::rl::Environment;
    use genesis_core::IValue;
    include!("../../examples/rl/balance.rs");
    include!("../../examples/robotics/arm_sim.rs");
}

use clap::Parser;
use std::fs;
use genesis_baker::ModelBlueprint;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    blueprint: String,
    #[arg(short, long, default_value = "model.state")]
    output: String,
}

fn main() {
    let args = Args::parse();
    let content = fs::read_to_string(&args.blueprint).expect("Failed to read blueprint");
    let blueprint: ModelBlueprint = toml::from_str(&content).expect("Invalid TOML");
    let baked = blueprint.bake();
    baked.save(&args.output).expect("Failed to save model");
    println!("Baked model '{}' to {} with {} neurons", blueprint.name, args.output, baked.neurons.len());
}

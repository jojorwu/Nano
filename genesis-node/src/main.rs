use genesis_node::Runtime;

fn main() {
    let runtime = Runtime::load("model.state").expect("Failed to load");
    println!("Runtime started with {} neurons", runtime.model.neurons.len());
}

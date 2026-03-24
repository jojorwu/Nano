from genesis_client import NanoRuntime, BrainBuilder

# 1. Build a new brain (Model)
builder = BrainBuilder("PythonModel")
builder.neurons = 2000
builder.synapses = 10000
builder.add_module("TextProcessor", vocab_size=5000)
builder.build("python_model.toml")

# 2. Run simulation with Python SDK
# Note: This requires a pre-baked model (python_model.state)
try:
    rt = NanoRuntime("model.state", backend="cpu")
    print(f"Loaded Nano model with {rt.neuron_count} neurons.")

    # 3. Inject text and simulate
    rt.inject_text("hello python world")
    for _ in range(10):
        spikes = rt.tick([0] * rt.neuron_count)
        active_count = sum(spikes)
        print(f"Simulation tick: {active_count} spikes")

    rt.save("model_updated.state")
except Exception as e:
    print(f"Could not load model: {e}")
    print("Please run 'nano-cli bake --blueprint blueprint.toml' first.")

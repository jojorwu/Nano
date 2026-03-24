import sys
import os

# Add build directory to path to find the .so
sys.path.append(os.path.abspath('target/debug'))

def test_python_sdk():
    print("Testing Python SDK integration...")
    from genesis_client import NanoRuntime, BrainBuilder

    model_path = "test.model"
    if not os.path.exists(model_path):
        print(f"Skipping load test as {model_path} does not exist.")
        return

    try:
        print("Loading runtime...")
        rt = NanoRuntime(model_path, backend="cpu")
        print(f"Successfully loaded model with {rt.neuron_count} neurons.")

        # Test tick
        print("Running tick...")
        inputs = [0] * rt.neuron_count
        spikes = rt.tick(inputs)
        print("Tick test passed.")

        # Test callback module
        print("Testing callback module...")
        ticks_received = []
        def my_callback(tick):
            ticks_received.append(tick)

        rt.add_module("PythonTest", my_callback)
        print("Running tick with callback...")
        rt.tick(inputs)
        print(f"Callback received: {ticks_received}")

        print("Python SDK integration tests PASSED.")

    except Exception as e:
        print(f"Python SDK test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_python_sdk()

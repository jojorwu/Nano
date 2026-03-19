import os
import subprocess
import time

# Training dataset
COMMANDS = [
    "ВКЛЮЧИТЬ СВЕТ",
    "ВЫКЛЮЧИТЬ СВЕТ",
    "ОСТАНОВИТЬ РОБОТА",
    "ИДИ ВПЕРЕД",
    "ПОВЕРНИ НАЛЕВО",
    "ПОВЕРНИ НАПРАВО"
]

TEST_COMMANDS = [
    "ВКЛЮЧИТЬ СВЕТ",  # Trained
    "НЕИЗВЕСТНАЯ КОМАНДА", # Unknown
    "ОСТАНОВИТЬ РОБОТА" # Trained
]

MODEL_PATH = "examples/commands/classifier.state"
BLUEPRINT_PATH = "examples/commands/blueprint.toml"

def run_cli_cmd(cli_args):
    cmd = ["cargo", "run", "-q", "-p", "axicor-cli", "--all-features", "--"] + cli_args
    # print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def main():
    # 1. Bake the model
    print("--- STEP 1: BAKING MODEL ---")
    run_cli_cmd(["bake", "--blueprint", BLUEPRINT_PATH, "--output", MODEL_PATH])

    # 2. Train (multiple exposures)
    print("\n--- STEP 2: TRAINING (3 EPOCHS) ---")
    for epoch in range(1, 4):
        print(f"\nEPOCH {epoch}:")
        for cmd_text in COMMANDS:
            output = run_cli_cmd(["run", "--model", MODEL_PATH, "--input", cmd_text])
            for line in output.split("\n"):
                if "Generated" in line:
                    print(f"   {cmd_text}: {line.strip()}")

    # 3. Verify
    print("\n--- STEP 3: VERIFYING ---")
    for cmd_text in TEST_COMMANDS:
        output = run_cli_cmd(["run", "--model", MODEL_PATH, "--input", cmd_text])
        for line in output.split("\n"):
            if "Generated" in line:
                print(f"   [{cmd_text}] activity: {line.strip()}")

    print("\n--- TRAINING FINISHED ---")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# encoding: utf-8

import os
import subprocess
import itertools
import time
from datetime import datetime

DATA_PATH = "./data/us_hhs/data.txt"
SAVE_DIR = "./save"

# 48-run focused search around current best
grid = {
    '--horizon': [12],
    '--window': [154, 158],           # Slight variation around 156
    '--hidRNN': [38, 42],              # Around 40
    '--dropout': [0.14, 0.15, 0.16],
    '--epilambda': [0.14, 0.16],       # Test slightly lower/higher physics weight
    '--lr': [0.00065, 0.00075],        # Around 0.0007
    '--lambda_t': [0.01],             # Keep strong value
    '--batch_size': [128],
    '--epochs': [600],
}

MAX_RUNS = None  # Set to 8 for testing

keys = list(grid.keys())
values = list(grid.values())
total_combinations = len(list(itertools.product(*values)))  # = 48

print(f"Total combinations: {total_combinations} (48 runs — fast & focused)")

print(f"Total combinations: {total_combinations}")
if MAX_RUNS:
    print(f"Limiting to {MAX_RUNS} runs for testing.")
    total_combinations = min(total_combinations, MAX_RUNS)

print("="*70)
print("STARTING FINE-GRAINED GRID SEARCH (v2)")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Centered on new best config:")
print("  window=156, hidRNN=40, dropout=0.15, epilambda=0.15, lambda_t=0.01, lr=0.0007")
print("="*70)

# =========================================================
# Run experiments
# =========================================================

count = 0
for combo in itertools.product(*values):
    count += 1
    if MAX_RUNS and count > MAX_RUNS:
        break

    args_list = []
    config_str = []

    for k, v in zip(keys, combo):
        args_list.extend([k, str(v)])
        config_str.append(f"{k.split('--')[1]}-{v}")

    run_name = "seir_ultrafine." + ".".join(config_str)  # Clear naming

    cmd = [
        ".venv/Scripts/python.exe", "main.py",
        "--data", DATA_PATH,
        "--model", "SEIRmodel",
        "--save_dir", SAVE_DIR,
        "--save_name", run_name,
        "--ifPlot", "0",
        "--seed", "54321",
        "--normalize", "2",
        "--train", "0.6",
        "--valid", "0.2",
    ]

    cmd.extend(args_list)

    print(f"\n[{count}/{total_combinations}] Launching: {run_name}")
    print("Command:", " ".join(cmd[:10]) + " ...")  # Truncated for readability

    try:
        result = subprocess.run(cmd, check=False)
        status = "✓ SUCCESS" if result.returncode == 0 else f"✗ FAILED (code {result.returncode})"
        print(f"{status}: {run_name}")
    except KeyboardInterrupt:
        print("\nGrid search interrupted by user.")
        break
    except Exception as e:
        print(f"✗ ERROR launching {run_name}: {e}")

    time.sleep(2)

print("\n" + "="*70)
print("FINE-GRAINED GRID SEARCH (v2) COMPLETED")
print(f"Ran {count} experiments.")
print("Next step:")
print("    python generatesummary.py")
print("New top models will appear with prefix 'seir_ultrafine.'")
print("Logs → ./logs/ | Models → ./save/")
print("="*70)
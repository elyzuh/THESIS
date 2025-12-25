#!/usr/bin/env python
# encoding: utf-8

import os
import subprocess
import itertools
import time
from datetime import datetime

# =========================================================
# Grid Search Configuration for PINN-SEIR Model
# =========================================================

# Fixed arguments
DATA_PATH = "./data/us_hhs/data.txt"
SAVE_DIR = "./save"
LOG_DIR = "./logs"

# Reduced grid — carefully chosen values based on common good practices
grid = {
    '--horizon': [12],               # Fixed: most common forecasting horizon in your logs
    '--window': [168, 252],          # 24 weeks (6 months) and 36 weeks (~9 months) lookback
    '--hidRNN': [50, 100],           # Standard sizes for RNN in time series
    '--dropout': [0.1, 0.2],          # Reasonable regularization
    '--epilambda': [0.05, 0.1, 0.2, 0.5],   # Physics constraint weight — key parameter!
    '--lambda_t': [0.001, 0.01],      # Temporal smoothness — small values usually best
    '--lr': [0.001, 0.0005],         # Common learning rates
    '--batch_size': [128],           # Fixed: larger is usually stable
    '--epochs': [300],               # Slightly more than before for convergence
}
# Optional: limit total runs for testing
MAX_RUNS = 5  # Set to e.g. 50 for quick testing

# =========================================================
# Generate all combinations
# =========================================================

keys = list(grid.keys())
values = list(grid.values())

total_combinations = len(list(itertools.product(*values)))
print(f"Total combinations: {total_combinations}")
if MAX_RUNS:
    print(f"Limiting to {MAX_RUNS} runs for testing.")
    total_combinations = min(total_combinations, MAX_RUNS)

print("="*60)
print("STARTING GRID SEARCH")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

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

    # Fixed args
    run_name = "seir." + ".".join(config_str)
    cmd = [
        ".venv/Scripts/python.exe", "main.py",
        "--data", DATA_PATH,
        "--model", "SEIRmodel",
        "--save_dir", SAVE_DIR,
        "--save_name", run_name,
        "--ifPlot", "0",                     # Change to 0 if no plots wanted
        "--seed", "54321",
        "--normalize", "2",                 # As in your original runs
        "--train", "0.6",
        "--valid", "0.2",
    ]

    # Add grid params
    cmd.extend(args_list)

    # Run the training
    try:
        result = subprocess.run(cmd, check=False)  # check=True would raise on error
        if result.returncode == 0:
            print(f"✓ SUCCESS: {run_name}")
        else:
            print(f"✗ FAILED (code {result.returncode}): {run_name}")
    except KeyboardInterrupt:
        print("\nGrid search interrupted by user.")
        break
    except Exception as e:
        print(f"✗ ERROR launching {run_name}: {e}")

    # Optional: small delay to avoid filesystem/GPU overload
    time.sleep(1)

print("\n" + "="*60)
print("GRID SEARCH COMPLETED")
print(f"Ran {count} experiments.")
print("All logs saved in ./logs/")
print("Models saved in ./save/")
print("Figures in ./Figures/<run_name>/")
print("="*60)
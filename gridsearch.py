#!/usr/bin/env python
# encoding: utf-8

import os
import subprocess
import itertools
import time
from datetime import datetime

DATA_PATH = "./data/us_hhs/data.txt"
SAVE_DIR = "./save"

# 27-run ultra-focused search around current champion
grid = {
    '--horizon': [12],
    '--window': [156],                     # Locked to best
    '--hidRNN': [40],                      # Locked to best
    '--dropout': [0.14, 0.15, 0.16],        # Small steps
    '--epilambda': [0.14, 0.15, 0.16],      # Small steps around 0.15
    '--lr': [0.00065, 0.0007, 0.00075],     # Fine learning rate tuning
    '--lambda_t': [0.01],                  # Locked
    '--batch_size': [128],
    '--epochs': [650],
}

MAX_RUNS = None  # Set to 9 for quick test

keys = list(grid.keys())
values = list(grid.values())
total_combinations = len(list(itertools.product(*values)))  # 27 runs

print(f"Total combinations: {total_combinations} (27 ultra-fast runs)")

if MAX_RUNS:
    total_combinations = min(total_combinations, MAX_RUNS)

print("="*70)
print("STARTING ULTRA-FOCUSED FINAL SEARCH")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Locked: window=156, hidRNN=40, lambda_t=0.01")
print("Tuning: dropout, epilambda, lr")
print("="*70)

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

    run_name = "seir_final." + ".".join(config_str)

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

    try:
        result = subprocess.run(cmd, check=False)
        status = "✓ SUCCESS" if result.returncode == 0 else f"✗ FAILED ({result.returncode})"
        print(f"{status}: {run_name}")
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(2)

print("\n" + "="*70)
print("FINAL SEARCH COMPLETED")
print(f"Ran {count} experiments.")
print("Run: python generatesummary.py")
print("Look for runs with prefix 'seir_final.'")
print("="*70)
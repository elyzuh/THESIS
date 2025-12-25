#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import math
import time
import torch
import torch.nn as nn
import numpy as np
import sys
import os

from models import SEIRmodel
from utils import *
from utils_ModelTrainEval import *
import Optim

import matplotlib.pyplot as plt
from PlotFunc import *

from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# Argument Parser
# =========================================================
parser = argparse.ArgumentParser(description='PINN-based Epidemiology Forecasting')

# --- Data options
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--train', type=float, default=0.6)
parser.add_argument('--valid', type=float, default=0.2)

# --- Model options
parser.add_argument('--model', type=str, default='SEIRmodel')

# misc
parser.add_argument('--sim_mat', type=str,help='file of similarity measurement (Required for CNNRNN_Res_epi)')
parser.add_argument('--metric', type=int, default=1, help='whether (1) or not (0) normalize rse and rae with global variance/deviation ')
parser.add_argument('--ratio', type=float, default=1.,help='The ratio between CNNRNN and residual')
parser.add_argument('--hidRNN', type=int, default=50, help='number of RNN hidden units')
parser.add_argument('--residual_window', type=int, default=4,help='The window size of the residual component')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--output_fun', type=str, default=None, help='the output function of neural net')

# --- Optimization
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip', type=float, default=1.)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--weight_decay', type=float, default=0)

# --- PINN parameters
parser.add_argument('--epilambda', type=float, default=0.2,
                    help='Weight of epidemiological (SEIR) constraint loss')
parser.add_argument('--lambda_t', type=float, default=0.01,
                    help='Temporal smoothness weight for epidemic parameters')

# --- Forecast settings
parser.add_argument('--window', type=int, default=24 * 7)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--normalize', type=int, default=0)

# --- System
parser.add_argument('--seed', type=int, default=54321)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--save_dir', type=str, default='./save')
parser.add_argument('--save_name', type=str, default='pinn_seir')

# plots
parser.add_argument('--ifPlot', type=int, default=1, help='1: plot figures, 0: no plots (evaluation only)')

args = parser.parse_args()
print(args)

# =========================================================
# Logging: redirect stdout to .out file
# =========================================================
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"{args.save_name}.out"
)

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

log_f = open(log_file, "w")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

# =========================================================
# Log command-line arguments at the top of the .out file
# =========================================================
print("=========================================================")
print(f"Run timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("Command-line Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print("=========================================================\n")



# =========================================================
# Environment setup
# =========================================================
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

torch.manual_seed(args.seed)
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

# =========================================================
# Load data
# =========================================================
Data = Data_utility(args)

# =========================================================
# Build model
# =========================================================
model = eval(args.model).Model(args, Data)
if args.cuda:
    model.cuda()

print(model)
print('* number of parameters:', sum(p.nelement() for p in model.parameters()))

# =========================================================
# Loss functions
# =========================================================
criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')

if args.cuda:
    criterion.cuda()
    evaluateL2.cuda()
    evaluateL1.cuda()

# =========================================================
# Optimizer
# =========================================================
optim = Optim.Optim(
    model.parameters(),
    args.optim,
    args.lr,
    args.clip,
    model.named_parameters(),
    weight_decay=args.weight_decay
)

# =========================================================
# Training Loop
# =========================================================
best_val = float('inf')

print("\nBEGIN TRAINING (PINN-SEIR)\n")

for epoch in range(1, args.epochs + 1):
    start_time = time.time()

    # ---- TRAIN STEP (MODIFIED train() EXPECTED)
    train_total, train_data, train_epi = train(
        Data,
        Data.train,
        model,
        criterion,
        optim,
        args.batch_size,
        args.model,
        args.epilambda,
        args.lambda_t
    )

    # ---- VALIDATION
    val_rse, val_rae, val_corr,val_r2 = evaluate(
        Data, Data.valid, model,
        evaluateL2, evaluateL1,
        args.batch_size, args.model
    )

    print(
        f"| Epoch {epoch:03d} | "
        f"Time {time.time() - start_time:.2f}s | "
        f"Total {train_total:.6f} | "
        f"Data {train_data:.6f} | "
        f"Epi {train_epi:.6f} | "
        f"Val RSE {val_rse:.4f} |"
        f"Val R² {val_r2:.4f}"
    )

    if math.isnan(train_total):
        sys.exit("NaN encountered")

    # ---- SAVE BEST MODEL
    if val_rse < best_val:
        best_val = val_rse
        model_path = f"{args.save_dir}/{args.save_name}.pt"
        torch.save(model.state_dict(), model_path)

        test_rse, test_rae, test_corr,test_r2  = evaluate(
            Data, Data.test, model,
            evaluateL2, evaluateL1,
            args.batch_size, args.model
        )

        print(
            f"  >> BEST MODEL SAVED | "
            f"Test RSE {test_rse:.4f} | "
            f"RAE {test_rae:.4f} | "
            f"CORR {test_corr:.4f} | "
            f"R² {test_r2:.4f}"
        )

# =========================================================
# Load Best Model (ALWAYS)
# =========================================================
best_model_path = f"{args.save_dir}/{args.save_name}.pt"
assert os.path.exists(best_model_path), "Best model not found!"

model.load_state_dict(torch.load(best_model_path))
model.eval()

print("\n================ FINAL EVALUATION ================\n")

test_rse, test_rae, test_corr, test_r2 = evaluate(
    Data, Data.test, model,
    evaluateL2, evaluateL1,
    args.batch_size, args.model
)

print(
    f"Test RSE  : {test_rse:.4f}\n"
    f"Test RAE  : {test_rae:.4f}\n"
    f"Test CORR : {test_corr:.4f}\n"
    f"Test R²   : {test_r2:.4f}\n"
)

# =========================================================
# Prediction & Interpretability (NO PLOTTING YET)
# =========================================================
X_true, Y_pred, Y_true, BetaList, GammaList, SigmaList = GetPrediction(
    Data, Data.test, model,
    evaluateL2, evaluateL1,
    args.batch_size, args.model
)

# =========================================================
# Conditional Plotting
# =========================================================
if args.ifPlot == 1:
    print("Plotting enabled — generating figures...")

    fig_dir = f"./Figures/{args.save_name}/"
    os.makedirs(fig_dir, exist_ok=True)

    PlotPredictionTrends(Y_true.T, Y_pred.T, fig_dir)
    PlotParameters(BetaList.T, GammaList.T, fig_dir)
    PlotSigma(SigmaList.T, fig_dir)

    print(f"Figures saved to {fig_dir}")

else:
    print("Plotting disabled (ifPlot=0). Evaluation only.")

print("\n==================== DONE ====================\n")

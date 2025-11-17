# file: main_pinn.py (FINAL)

from __future__ import print_function
import argparse
import time
import os
import torch
import numpy as np
import sys

from models import SEIR_PINN
from utils import Data_utility
from pinn_train_eval import train_pinn, evaluate_pinn
import Optim

parser = argparse.ArgumentParser(description='SEIR PINN Epidemiology Forecasting with NGM')
parser.add_argument('--data', type=str, required=True, help='location of the data file')
parser.add_argument('--train', type=float, default=0.6)
parser.add_argument('--valid', type=float, default=0.2)
parser.add_argument('--save_dir', type=str,  default='./save')
parser.add_argument('--save_name', type=str,  default='pinn_ngm_model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--lr', type=float, default=0.0001)
# --- NEW: NGM Loss Weight ---
parser.add_argument('--lambda_ngm', type=float, default=0.1, help='weight of the NGM loss')
parser.add_argument('--lambda_physics', type=float, default=0.01)
parser.add_argument('--model', type=str, default='SEIR_PINN')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--window', type=int, default=1)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--sim_mat', type=str, default=None)
parser.add_argument('--metric', type=int, default=0)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--cuda', type=str, default=None)
parser.add_argument('--clip', type=float, default=1.0)

args = parser.parse_args()
print(args)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args)
model = SEIR_PINN.Model(args, Data)
if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

try:
    print('Begin training SEIR PINN with NGM...')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # UPDATED: The training function now returns the NGM loss as well
        train_loss, data_loss, physics_loss, ngm_loss = train_pinn(Data, model, optim, args)
        
        if epoch % 100 == 0:
            print('| epoch {:5d} | time: {:5.2f}s | total_loss {:5.6f} | data_loss {:5.6f} | physics_loss {:5.6f} | ngm_loss {:5.6f}'
                  .format(epoch, (time.time() - epoch_start_time), train_loss, data_loss, physics_loss, ngm_loss))
            
            test_rmse, test_mae, test_corr = evaluate_pinn(Data, model, args)
            print('| VALIDATION | test rmse {:5.4f} | test mae {:5.4f} | test corr {:5.4f}'.format(test_rmse, test_mae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
with open(model_path, 'wb') as f:
    torch.save(model.state_dict(), f)

print("\n" + "-" * 89)
print("Training finished. Final Evaluation on Test Set:")
final_rmse, final_mae, final_corr = evaluate_pinn(Data, model, args)
print("Final Test RMSE: {:5.4f} | Final Test MAE: {:5.4f} | Final Test Corr: {:5.4f}".format(final_rmse, final_mae, final_corr))
print("-" * 89)
print("Learned Epidemiological Parameters (one per location):")
print("Beta (Transmission Rate):", model.beta.data.cpu().numpy().flatten())
print("Sigma (Latency Rate):", model.sigma.data.cpu().numpy().flatten())
print("Gamma (Recovery Rate):", model.gamma.data.cpu().numpy().flatten())
print("-" * 89)
print("Final Learned Mobility Matrix (Pi):")
final_mobility_matrix = F.softmax(model.mobility_matrix.data, dim=1).cpu().numpy()
# Print formatted to 2 decimal places for readability
print(np.array2string(final_mobility_matrix, formatter={'float_kind':lambda x: "%.2f" % x}))
print("-" * 89)
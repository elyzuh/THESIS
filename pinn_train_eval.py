# file: pinn_train_eval.py (UPDATED)

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def train_pinn(data_loader, model, optim, args):
    model.train()
    t_data, i_data, loc_data, t_physics, loc_physics = data_loader.get_pinn_batches()
    t_physics.requires_grad = True
    model.zero_grad()
    
    # --- 1. Data Loss (MSE) ---
    input_for_data = torch.cat([t_data, loc_data], dim=1)
    seir_pred_for_data = model(input_for_data)
    i_pred_for_data = seir_pred_for_data[:, 2]
    loss_data = torch.mean((i_pred_for_data - i_data.squeeze(1))**2)

    # --- 2. Physics Loss ---
    input_for_physics = torch.cat([t_physics, loc_physics], dim=1)
    seir_pred_for_physics = model(input_for_physics)
    S, E, I, R = seir_pred_for_physics[:, 0], seir_pred_for_physics[:, 1], seir_pred_for_physics[:, 2], seir_pred_for_physics[:, 3]
    loc_indices = loc_physics.long().squeeze(1)
    beta = model.beta[loc_indices]
    sigma = model.sigma[loc_indices]
    gamma = model.gamma[loc_indices]
    dS_dt = torch.autograd.grad(S, t_physics, torch.ones_like(S), create_graph=True)[0]
    dE_dt = torch.autograd.grad(E, t_physics, torch.ones_like(E), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I, t_physics, torch.ones_like(I), create_graph=True)[0]
    dR_dt = torch.autograd.grad(R, t_physics, torch.ones_like(R), create_graph=True)[0]
    N = 1.0
    residual_S = dS_dt - (-beta * S * I / N)
    residual_E = dE_dt - (beta * S * I / N - sigma * E)
    residual_I = dI_dt - (sigma * E - gamma * I)
    residual_R = dR_dt - (gamma * I)
    loss_physics = torch.mean(residual_S**2) + torch.mean(residual_E**2) + torch.mean(residual_I**2) + torch.mean(residual_R**2)

    # --- 3. NEW: NGM Loss ---
    # This loss enforces consistency between the FNN dynamics and NGM dynamics
    loss_ngm = 0
    if args.lambda_ngm > 0:
        # Get the NGM for the current state of learnable parameters
        K = model.compute_ngm()

        # Get the FNN's prediction of I at the previous time step (t-1)
        # We need to reshape I from a flat list to a (time x location) grid
        num_times = len(torch.unique(t_physics.flatten()))
        num_locs = model.m
        I_grid = I.view(num_times, num_locs)
        
        # Get infection data from the previous step (I_t-1)
        I_prev = I_grid[:-1, :] # All but the last time step
        # Get the NGM prediction for the current step: I_pred = I_t-1 * K^T
        I_pred_ngm = torch.matmul(I_prev, K.T)
        
        # Get the FNN's prediction for the current step (I_t)
        I_curr_fnn = I_grid[1:, :] # All but the first time step

        # The NGM loss is the difference between the two predictions
        loss_ngm = torch.mean((I_curr_fnn - I_pred_ngm)**2)

    # --- 4. Total Loss ---
    loss = loss_data + args.lambda_physics * loss_physics + args.lambda_ngm * loss_ngm
    
    loss.backward()
    optim.step()
    
    return loss.item(), loss_data.item(), loss_physics.item(), loss_ngm.item() if isinstance(loss_ngm, torch.Tensor) else loss_ngm

# The evaluation function remains the same, no changes needed here.
def evaluate_pinn(data_loader, model, args):
    model.eval()
    t_test, i_test, loc_test = data_loader.get_pinn_test_batches()
    with torch.no_grad():
        input_test = torch.cat([t_test, loc_test], dim=1)
        seir_pred_test = model(input_test)
        i_pred_test = seir_pred_test[:, 2]
    scale = data_loader.scale.mean().cpu().numpy()
    i_pred_unscaled = i_pred_test.cpu().numpy() * scale
    i_test_unscaled = i_test.cpu().numpy() * scale
    rmse = np.sqrt(mean_squared_error(i_test_unscaled, i_pred_unscaled))
    mae = mean_absolute_error(i_test_unscaled, i_pred_unscaled)
    corr = np.corrcoef(i_test_unscaled.flatten(), i_pred_unscaled.flatten())[0, 1]
    return rmse, mae, corr
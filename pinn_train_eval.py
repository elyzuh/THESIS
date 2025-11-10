# file: pinn_train_eval.py

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def train_pinn(data_loader, model, optim, args):
    model.train()

    # Get the specially prepared batches for PINN training
    t_data, i_data, loc_data, t_physics, loc_physics = data_loader.get_pinn_batches()
    
    # We need to compute gradients with respect to the time input for the physics loss
    t_physics.requires_grad = True

    model.zero_grad()
    
    # --- 1. Data Loss (MSE) ---
    # This loss measures how well the model's 'I' prediction matches the real data
    input_for_data = torch.cat([t_data, loc_data], dim=1)
    seir_pred_for_data = model(input_for_data)
    i_pred_for_data = seir_pred_for_data[:, 2] # Index 2 corresponds to the 'I' compartment
    
    loss_data = torch.mean((i_pred_for_data - i_data.squeeze(1))**2)

    # --- 2. Physics Loss ---
    # This loss measures how well the model's outputs obey the SEIR differential equations
    input_for_physics = torch.cat([t_physics, loc_physics], dim=1)
    seir_pred_for_physics = model(input_for_physics)
    S, E, I, R = seir_pred_for_physics[:, 0], seir_pred_for_physics[:, 1], seir_pred_for_physics[:, 2], seir_pred_for_physics[:, 3]

    # Get the correct epidemiological parameter (beta, sigma, gamma) for each point's location
    loc_indices = loc_physics.long().squeeze(1)
    beta = model.beta[loc_indices]
    sigma = model.sigma[loc_indices]
    gamma = model.gamma[loc_indices]
    
    # Use PyTorch's automatic differentiation to get the derivatives
    dS_dt = torch.autograd.grad(S, t_physics, torch.ones_like(S), create_graph=True)[0]
    dE_dt = torch.autograd.grad(E, t_physics, torch.ones_like(E), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I, t_physics, torch.ones_like(I), create_graph=True)[0]
    dR_dt = torch.autograd.grad(R, t_physics, torch.ones_like(R), create_graph=True)[0]

    # These are the SEIR equations from your thesis (Figure 4) written as residuals
    # We assume N=1 since the data is normalized
    N = 1.0
    residual_S = dS_dt - (-beta * S * I / N)
    residual_E = dE_dt - (beta * S * I / N - sigma * E)
    residual_I = dI_dt - (sigma * E - gamma * I)
    residual_R = dR_dt - (gamma * I)
    
    # The physics loss is the mean squared error of these residuals. The goal is to make them zero.
    loss_physics = torch.mean(residual_S**2) + torch.mean(residual_E**2) + torch.mean(residual_I**2) + torch.mean(residual_R**2)

    # --- 3. Total Loss ---
    # Combine the data loss and the physics loss using a weighting factor
    loss = loss_data + args.lambda_physics * loss_physics
    
    loss.backward()
    optim.step()
    
    return loss.item(), loss_data.item(), loss_physics.item()

def evaluate_pinn(data_loader, model, args):
    model.eval()
    # Get all test data points at once
    t_test, i_test, loc_test = data_loader.get_pinn_test_batches()

    with torch.no_grad():
        input_test = torch.cat([t_test, loc_test], dim=1)
        seir_pred_test = model(input_test)
        i_pred_test = seir_pred_test[:, 2] # Get the 'I' predictions

    # Un-normalize the predictions and ground truth to calculate meaningful metrics
    # We use the mean of the scaling factors as a representative scale
    scale = data_loader.scale.mean().cpu().numpy()
    i_pred_unscaled = i_pred_test.cpu().numpy() * scale
    i_test_unscaled = i_test.cpu().numpy() * scale
    
    rmse = np.sqrt(mean_squared_error(i_test_unscaled, i_pred_unscaled))
    mae = mean_absolute_error(i_test_unscaled, i_pred_unscaled)
    corr = np.corrcoef(i_test_unscaled.flatten(), i_pred_unscaled.flatten())[0, 1]
    
    return rmse, mae, corr
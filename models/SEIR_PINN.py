# file: models/SEIR_PINN.py (UPDATED)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.m = data.m  # Number of locations (e.g., 10 HHS regions)

        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 4)   # Output: [S, E, I, R]
        )

        # --- Learnable Epidemiological Parameters ---
        self.beta = nn.Parameter(torch.rand(self.m, 1) * 0.5)
        self.sigma = nn.Parameter(torch.rand(self.m, 1) * 0.5)
        self.gamma = nn.Parameter(torch.rand(self.m, 1) * 0.5)

        # --- NEW: Learnable Mobility Matrix ---
        # This is the equivalent of Φ_G in the paper. It's a learnable
        # (m x m) matrix representing cross-location transmission.
        self.mobility_matrix = nn.Parameter(torch.rand(self.m, self.m))

        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        seir_output = self.net(x)
        return torch.sigmoid(seir_output)

    # --- NEW: Method to compute the Next-Generation Matrix ---
    def compute_ngm(self):
        """
        This method implements Equation (5) from the reference paper:
        K = β * (γ + A)^-1
        where A = W - Π^T
        """
        # 1. Create diagonal matrices for beta and gamma
        # Clamp values to prevent numerical instability (e.g., negative rates)
        beta_diag = torch.diag(self.beta.clamp(0, 1).squeeze())
        gamma_diag = torch.diag(self.gamma.clamp(0, 1).squeeze())

        # 2. Create the mobility matrix Π (Pi) from our learnable parameter
        # We apply softmax as described in the paper to get probabilities
        Pi = F.softmax(self.mobility_matrix, dim=1)

        # 3. Calculate the A matrix
        # W is the degree matrix of Pi
        W = torch.diag(torch.sum(Pi, dim=0))
        # A = W - Pi^T
        A = W - Pi.T

        # 4. Calculate the NGM (K)
        # We add a small identity matrix for numerical stability during inversion
        identity = torch.eye(self.m, device=A.device) * 1e-5
        try:
            # K = β_diag @ (γ_diag + A)^-1
            K = torch.matmul(beta_diag, torch.inverse(gamma_diag + A + identity))
        except torch.linalg.LinAlgError:
            # If the matrix is singular, return an identity matrix as a fallback
            return torch.eye(self.m, device=A.device)
            
        return K
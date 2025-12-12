import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.ratio = args.ratio
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m

        # --- Deep Learning Component (Data Driven) ---
        self.hidR = args.hidRNN
        self.GRU1 = nn.GRU(self.m, self.hidR)
        self.residual_window = args.residual_window
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        
        # Graph Structure
        self.mask_mat = nn.Parameter(torch.Tensor(self.m, self.m))
        self.adj = data.adj

        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1)
        
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

        # --- Physics Component (SEIR Parameter Prediction) ---
        # We need independent GRUs to predict dynamic parameters Beta, Gamma, and Sigma
        self.GRU_Physics = nn.GRU(1, self.hidR, batch_first=True)
        
        # Predict Beta (Infection Rate)
        self.PredBeta = nn.Sequential(
            nn.Linear(self.hidR, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid() 
        )

        # Predict Gamma (Recovery Rate)
        self.PredGamma = nn.Sequential(
            nn.Linear(self.hidR, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        # [NEW] Predict Sigma (Incubation Rate for SEIR)
        self.PredSigma = nn.Sequential(
            nn.Linear(self.hidR, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        # [NEW] Latent State Estimator
        # Since input 'x' is likely just 'Infected', we need to estimate S, E, R 
        # based on the history of I. 
        # Input: Window size (self.P) -> Output: 3 states (S, E, R)
        self.state_estimator = nn.Linear(self.P, 3)

    def forward(self, x):
        # x: batch (b) x window (P) x locations (m)
        xOriginal = x[:,:,:]
        b = x.shape[0]

        # 1. Graph Adjacency Processing
        masked_adj_soft = F.softmax(self.mask_mat, dim=1)
        masked_adj_epi = self.adj * masked_adj_soft
        masked_adj_deep = self.adj * masked_adj_soft

        # --------------------------------------------------
        # ------------ Physics-Informed SEIR Block ---------
        # --------------------------------------------------

        # Reshape for Parameter Prediction: (batch*location, window, 1)
        x_reshaped = x.permute(0, 2, 1).contiguous().view(-1, self.P, 1)

        # Run GRU once for physics embeddings
        ROut, _ = self.GRU_Physics(x_reshaped)
        RoutFinalStep = ROut[:,-1,:] # (batch*location, hidden)

        # Predict Parameters (Time-varying ODE parameters)
        # View as (batch, locations)
        Beta = self.PredBeta(RoutFinalStep).view(b, self.m)
        Gamma = self.PredGamma(RoutFinalStep).view(b, self.m)
        Sigma = self.PredSigma(RoutFinalStep).view(b, self.m)

        # --- SEIR State Estimation ---
        # We assume x contains 'I' (Infected). We need S, E, R.
        # Use the history of I to infer current S, E, R
        # Input: (batch, locations, window)
        history_I = x.permute(0, 2, 1) 
        estimated_states = F.relu(self.state_estimator(history_I)) # (batch, locations, 3)
        
        # Normalize states to ensure physical validity (fractions summing to <= 1)
        # Note: This is a soft constraint strategy. 
        # Current 'I' is the last value in the window
        Current_I = x[:, -1, :] 
        Current_S = estimated_states[:, :, 0]
        Current_E = estimated_states[:, :, 1]
        Current_R = estimated_states[:, :, 2]

        # --- Network Diffusion (Force of Infection) ---
        # In a network, S is infected by local I AND neighbor I.
        # Transmission = Beta * S * (I_local + sum(Adj * I_neighbor))
        
        # Calculate Neighbor Effect: (batch, locations)
        # (batch, locations) @ (locations, locations) -> (batch, locations)
        Neighbor_I = torch.matmul(Current_I, masked_adj_epi)
        
        # Total infective force
        Total_Exposure = Current_I + Neighbor_I

        # --- Explicit SEIR Euler Integration Step ---
        # dS = -Beta * S * Total_Exposure
        # dE = Beta * S * Total_Exposure - Sigma * E
        # dI = Sigma * E - Gamma * I
        # dR = Gamma * I
        
        new_S = -Beta * Current_S * Total_Exposure
        new_E = (Beta * Current_S * Total_Exposure) - (Sigma * Current_E)
        new_I = (Sigma * Current_E) - (Gamma * Current_I)
        new_R = Gamma * Current_I

        # Apply updates (Assuming dt=1)
        Next_I = Current_I + new_I
        
        # Ensure positivity (Physics constraint)
        EpiOutput = F.relu(Next_I)

        # ----------------------------------------
        # ------------ Deep Learning Block -------
        # ----------------------------------------
        # (Standard GNN+RNN for residual correction)
        
        xTrans = x.matmul(masked_adj_deep)
        r = xTrans.permute(1, 0, 2).contiguous() # (window, batch, loc)
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))
        res = self.linear1(r)

        # Residual connection
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(-1,self.m)
            res = res * self.ratio + z

        if self.output is not None:
            res = self.output(res).float()

        # Returns:
        # res: The pure deep learning prediction
        # EpiOutput: The Physics-Informed prediction (Next I)
        # Beta, Gamma, Sigma: The learned parameters (useful for interpretability)
        return res, EpiOutput, Beta, Gamma, Sigma
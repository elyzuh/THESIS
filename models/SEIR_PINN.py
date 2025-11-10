# file: models/SEIR_PINN.py

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        # self.m is the number of locations, which will be 10 from your data.txt
        self.m = data.m

        # This simple Feedforward Neural Network (FNN) is the core of the PINN.
        # It takes a 2-element tensor [time, location_index] as input.
        # It outputs a 4-element tensor [S, E, I, R].
        self.net = nn.Sequential(
            nn.Linear(2, 100), # Using 100 neurons as per your thesis description
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 4)   # Output layer for the 4 SEIR compartments
        )

        # --- Learnable Epidemiological Parameters ---
        # These are the unknown parameters we want the network to learn from data
        # and physics. We create one parameter for each of the 'm' locations.
        self.beta = nn.Parameter(torch.rand(self.m, 1) * 0.5)  # Transmission rate
        self.sigma = nn.Parameter(torch.rand(self.m, 1) * 0.5) # Latency rate
        self.gamma = nn.Parameter(torch.rand(self.m, 1) * 0.5) # Recovery rate

        # Initialize the network weights for better training stability
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        """
        x: A tensor of shape [batch_size, 2] where columns are [time, location_index]
        """
        seir_output = self.net(x)
        
        # We use a sigmoid function to ensure the outputs are positive and
        # scaled between 0 and 1, as they represent fractions of a population.
        return torch.sigmoid(seir_output)
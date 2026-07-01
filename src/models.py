import torch
import torch.nn as nn

class LensPredictorNetwork(nn.Module):
    def __init__(self, input_dim):
        
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim,256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256,128),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128,27)

        )
        
    def forward(self, x):
        return self.network(x)

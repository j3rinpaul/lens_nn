import torch
import torch.nn as nn

class LensPredictorNetwork(nn.Module):
    def __init__(self, input_dim=40, output_dim=27):
        super().__init__()
        
        
        self.network = nn.Sequential(
            # Layer 1: 160 Inputs -> 256 Neurons
            nn.Dropout(0.1), # Prevents overfitting
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x is your batch of 160-element histograms
        logits = self.network(x)
        return logits # Shape: [Batch_Size, 27]
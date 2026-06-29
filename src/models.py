import torch
import torch.nn as nn

class LensPredictorNetwork(nn.Module):
    def __init__(self, input_dim=74):
        # input_dim = 74 from extract_features()
        # 60 (RGB spatial histograms) + 9 (per-channel stats) + 5 (spatial contrast)
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.iso_head      = nn.Linear(64, 3)  # ISO:      250 / 2000 / 16000
        self.shutter_head  = nn.Linear(64, 3)  # Shutter:  1/4  / 1/60 / 1/1000
        self.aperture_head = nn.Linear(64, 3)  # Aperture: f5.0 / f9.0 / f16

    def forward(self, x):
        feat = self.backbone(x)
        return self.iso_head(feat), self.shutter_head(feat), self.aperture_head(feat)

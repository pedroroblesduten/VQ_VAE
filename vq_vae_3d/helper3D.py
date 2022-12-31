import torch 
import torch.nn as nn
import torch.nn.functional as F

class Residual3D(nn.Module):
    def __init__(self, in_channels, res_channels, dropout):
        super().__init__()
        self._res_block = nn.Sequential(
            nn.Conv3d(in_channels, res_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv3d(res_channels, in_channels,
                      kernel_size=1, stride=1)
        )

    def forward(self, x):
        return x + self._res_block(x)

class ResidualStack3D(nn.Module):
    def __init__(self, in_channels, res_channels, dropout, num_residual_layers):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._res_layers = nn.ModuleList([
            Residual3D(in_channels, res_channels, dropout)
            for _ in range(self._num_residual_layers)
        ])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._res_layers[i](x)
        x = F.relu(x)
        return x



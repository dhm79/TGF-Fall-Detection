"""
Arquitectura usada:
Input: (B, 1, 10, 256, 256)

Bloque 1:
- Conv3D(1 -> 32, kernel=3x3x3, padding=0)
- LeakyReLU(0.1)
- MaxPool3D(1,2,2)
- Dropout(0.25)

Bloque 2:
- Conv3D(32 -> 64, kernel=3x3x3, padding=0)
- LeakyReLU(0.1)
- MaxPool3D(1,2,2)
- Dropout(0.25)

Bloque 3:
- Conv3D(64 -> 128, kernel=3x3x3, padding=0)
- LeakyReLU(0.1)
- MaxPool3D(1,2,2)
- Dropout(0.25)

Clasificador:
- Flatten -> 460800
- Linear(460800 -> 64)
- Dropout(0.5)
- Linear(64 -> 1)
"""

import torch
import torch.nn as nn


class TF66Baseline3DCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # -----------------------------
        # Bloque 1
        # Entrada: (B, 1, 10, 256, 256)
        # Conv valid -> (B, 32, 8, 254, 254)
        # Pool       -> (B, 32, 8, 127, 127)
        # -----------------------------
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=0)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.drop1 = nn.Dropout(p=0.25)

        # -----------------------------
        # Bloque 2
        # Entrada: (B, 32, 8, 127, 127)
        # Conv valid -> (B, 64, 6, 125, 125)
        # Pool       -> (B, 64, 6, 62, 62)
        # -----------------------------
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.drop2 = nn.Dropout(p=0.25)

        # -----------------------------
        # Bloque 3
        # Entrada: (B, 64, 6, 62, 62)
        # Conv valid -> (B, 128, 4, 60, 60)
        # Pool       -> (B, 128, 4, 30, 30)
        # -----------------------------
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0)
        self.act3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.drop3 = nn.Dropout(p=0.25)

        # 128 * 4 * 30 * 30 = 460800
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(460800, 64)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bloque 1
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Bloque 2
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Bloque 3
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Clasificador
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop_fc(x)
        x = self.fc_out(x)

        return x
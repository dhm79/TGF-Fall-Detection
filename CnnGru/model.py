import torch
import torch.nn as nn


class FrameCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


class FallNet(nn.Module):
    def __init__(self, feature_dim=128, hidden_size=128):
        super().__init__()

        self.cnn = FrameCNN(feature_dim)

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape

        feats = []
        for t in range(T):
            f = self.cnn(x[:, :, t])
            feats.append(f)

        x = torch.stack(feats, dim=1)

        out, _ = self.gru(x)

        last = out[:, -1]

        return self.classifier(last)
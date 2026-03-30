import math
import torch
import torch.nn as nn


# =========================================
# CNN PARA PROCESAR CADA FRAME
# =========================================
class FrameCNN(nn.Module):
    """
    Extrae características espaciales de cada frame térmico.

    Entrada por frame: (B, 1, H, W)
    Salida por frame: (B, feature_dim)
    """

    def __init__(self, feature_dim=128):
        super().__init__()

        # Bloque convolucional
        self.features = nn.Sequential(
            # Capa 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Capa 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Capa 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Resume cada mapa de características a tamaño 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Convierte el vector de 64 componentes a feature_dim
        self.projection = nn.Linear(64, feature_dim)

    def forward(self, x):
        # x: (B, 1, H, W)

        x = self.features(x)      # (B, 64, 1, 1)
        x = x.flatten(1)          # (B, 64)
        x = self.projection(x)    # (B, feature_dim)

        return x


# =========================================
# POSITIONAL ENCODING
# =========================================
class PositionalEncoding(nn.Module):
    """
    Añade información posicional a cada frame de la secuencia.
    El Transformer por sí solo no sabe qué frame va primero o después,
    así que necesitamos codificar la posición temporal.
    """

    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Forma final: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer hace que pe forme parte del modelo,
        # pero no sea un parámetro entrenable
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)

        # Sumamos codificación posicional a la secuencia
        return x + self.pe[:, :seq_len, :]


# =========================================
# MODELO COMPLETO: CNN + TRANSFORMER
# =========================================
class TransformerFallNet(nn.Module):
    """
    Modelo CNN + Transformer para detección de caídas.

    Entrada: (B, 1, T, H, W)
    Salida: logits (B, 1)
    """

    def __init__(
        self,
        feature_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.3
    ):
        super().__init__()

        # CNN por frame
        self.frame_cnn = FrameCNN(feature_dim=feature_dim)

        # Positional encoding temporal
        self.pos_encoder = PositionalEncoding(d_model=feature_dim, max_len=200)

        # Una capa base del Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,       # dimensión del embedding
            nhead=num_heads,           # número de cabezas de atención
            dim_feedforward=ff_dim,    # tamaño de la red feed-forward interna
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )

        # Apilamos varias capas Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, 1, T, H, W)

        batch_size, channels, seq_len, height, width = x.shape

        frame_features = []

        # =========================================
        # 1. Procesar cada frame con la CNN
        # =========================================
        for t in range(seq_len):
            frame = x[:, :, t, :, :]            # (B, 1, H, W)
            feat = self.frame_cnn(frame)        # (B, feature_dim)
            frame_features.append(feat)

        # =========================================
        # 2. Construir secuencia temporal
        # =========================================
        x_seq = torch.stack(frame_features, dim=1)   # (B, T, feature_dim)

        # =========================================
        # 3. Añadir información de posición
        # =========================================
        x_seq = self.pos_encoder(x_seq)              # (B, T, feature_dim)

        # =========================================
        # 4. Procesar secuencia completa con Transformer
        # =========================================
        x_seq = self.transformer(x_seq)              # (B, T, feature_dim)

        # =========================================
        # 5. Pooling temporal global
        # =========================================
        # En vez de usar solo el último frame, hacemos media de toda la secuencia
        x_seq = x_seq.mean(dim=1)                    # (B, feature_dim)

        # =========================================
        # 6. Clasificación final
        # =========================================
        logits = self.classifier(x_seq)              # (B, 1)

        return logits
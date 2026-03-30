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
        # Aquí se extraen características espaciales (formas, contornos, calor, etc.)
        self.features = nn.Sequential(

            # Capa 1: detecta patrones básicos (bordes, zonas calientes)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # reduce tamaño a la mitad

            # Capa 2: patrones más complejos
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Capa 3: características más abstractas
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Reduce todo a 1x1 → resumen global del frame
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Proyección final a un vector de características
        self.projection = nn.Linear(64, feature_dim)

    def forward(self, x):
        # x: (B, 1, H, W)

        # Pasamos por la CNN
        x = self.features(x)          # (B, 64, 1, 1)

        # Aplanamos → vector
        x = x.flatten(1)              # (B, 64)

        # Proyectamos a dimensión deseada
        x = self.projection(x)        # (B, feature_dim)

        return x


# =========================================
# MODELO COMPLETO: CNN + BiGRU
# =========================================
class FallNet(nn.Module):
    """
    Modelo CNN + BiGRU para detección de caídas.

    Entrada: (B, 1, T, H, W)
    Salida: logits (B, 1)
    """

    def __init__(self, feature_dim=128, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()

        # CNN que procesa cada frame individualmente
        self.frame_cnn = FrameCNN(feature_dim=feature_dim)

        # GRU bidireccional (procesa la secuencia temporal)
        self.gru = nn.GRU(
            input_size=feature_dim,     # entrada: vector de cada frame
            hidden_size=hidden_size,    # tamaño del estado interno
            num_layers=num_layers,
            batch_first=True,           # entrada: (B, T, features)
            bidirectional=True,         # procesa hacia adelante y hacia atrás
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # *2 por bidireccional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # salida binaria (caída / no caída)
        )

    def forward(self, x):
        # x: (B, 1, T, H, W)

        # Extraemos dimensiones
        batch_size, channels, seq_len, height, width = x.shape

        frame_features = []

        # =========================================
        # 1. Procesar cada frame con la CNN
        # =========================================
        for t in range(seq_len):

            # Seleccionamos el frame t
            frame = x[:, :, t, :, :]              # (B, 1, H, W)

            # Extraemos características espaciales
            feat = self.frame_cnn(frame)          # (B, feature_dim)

            frame_features.append(feat)

        # =========================================
        # 2. Crear secuencia temporal
        # =========================================
        # Convertimos lista → tensor
        # Resultado: (B, T, feature_dim)
        x_seq = torch.stack(frame_features, dim=1)

        # =========================================
        # 3. Procesar secuencia con GRU
        # =========================================
        # Salida: (B, T, hidden_size * 2)
        gru_out, _ = self.gru(x_seq)

        # =========================================
        # 4. Usar último timestep
        # =========================================
        last_out = gru_out[:, -1, :]              # (B, hidden_size*2)

        # =========================================
        # 5. Clasificación final
        # =========================================
        logits = self.classifier(last_out)        # (B, 1)

        return logits
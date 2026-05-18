import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Convierte una imagen (B, 1, H, W) en una secuencia de patches embebidos.
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=128):
        super().__init__()

        assert img_size % patch_size == 0, "img_size debe ser divisible por patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2

        # Proyección de patches usando una conv con stride = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.proj(x)                  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                  # (B, embed_dim, N)
        x = x.transpose(1, 2)             # (B, N, embed_dim)
        return x


class SpatialTransformerEncoder(nn.Module):
    """
    Procesa los patches de un frame con Transformer y devuelve un embedding por frame.
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1,
        embed_dim=128,
        num_heads=4,
        depth=2,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Token CLS para resumir cada frame
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding espacial
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 1, H, W)
        B = x.size(0)

        x = self.patch_embed(x)                              # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)               # (B, N+1, D)

        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)

        # Nos quedamos con CLS token como embedding del frame
        frame_embedding = x[:, 0, :]                        # (B, D)
        return frame_embedding


class TemporalTransformerEncoder(nn.Module):
    """
    Procesa la secuencia de embeddings temporales.
    """
    def __init__(
        self,
        seq_len=10,
        embed_dim=96,
        num_heads=4,
        depth=2,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, T, D)
        B = x.size(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)      # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)              # (B, T+1, D)

        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)

        seq_embedding = x[:, 0, :]                         # (B, D)
        return seq_embedding


class FallTransformer(nn.Module):
    """
    Modelo puro Transformer:
    1. Transformer espacial por frame
    2. Transformer temporal por secuencia
    3. Clasificación binaria
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        seq_len=10,
        in_chans=1,
        embed_dim=128,
        spatial_depth=2,
        temporal_depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.spatial_encoder = SpatialTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=spatial_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        self.temporal_encoder = TemporalTransformerEncoder(
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=temporal_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, 1, T, H, W)
        B, C, T, H, W = x.shape

        frame_embeddings = []

        for t in range(T):
            frame = x[:, :, t, :, :]                      # (B, 1, H, W)
            emb = self.spatial_encoder(frame)            # (B, D)
            frame_embeddings.append(emb)

        x_seq = torch.stack(frame_embeddings, dim=1)     # (B, T, D)
        seq_emb = self.temporal_encoder(x_seq)           # (B, D)

        logits = self.classifier(seq_emb)                # (B, 1)
        return logits
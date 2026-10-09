from __future__ import annotations

import torch
from torch import nn


class CTGEncoder(nn.Module):
    """CNN + Transformer encoder for CTG windows shaped [batch, time, 2]."""

    def __init__(self, input_channels: int = 2, d_model: int = 128, transformer_layers: int = 2, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        if transformer_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        else:
            # transformer_layers=0 -> CNN-only ablation: skip attention entirely,
            # go straight from the CNN feature map to the temporal mean-pool.
            self.transformer = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # [B, D, T'] -> [B, T', D]
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling over time
        return self.projection(x)


class CTGOnlyModel(nn.Module):
    def __init__(self, input_channels: int = 2, **encoder_kwargs):
        super().__init__()
        self.encoder = CTGEncoder(input_channels=input_channels, **encoder_kwargs)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x_ctg: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x_ctg)
        return self.classifier(emb).squeeze(-1)

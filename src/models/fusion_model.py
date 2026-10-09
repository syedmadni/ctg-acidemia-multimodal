from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from src.models.ctg_cnn import CTGEncoder


class MaternalEncoder(nn.Module):
    def __init__(self, num_features: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaternalOnlyModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = MaternalEncoder(num_features=num_features)
        self.classifier = nn.Linear(32, 1)

    def forward(self, x_maternal: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x_maternal)
        return self.classifier(emb).squeeze(-1)


class FusionModel(nn.Module):
    def __init__(self, num_maternal_features: int, input_channels: int = 2, dropout: float = 0.3, **encoder_kwargs):
        super().__init__()
        ctg_dropout = encoder_kwargs.pop("dropout", dropout)
        self.ctg_encoder = CTGEncoder(input_channels=input_channels, dropout=ctg_dropout, **encoder_kwargs)
        self.maternal_encoder = MaternalEncoder(num_features=num_maternal_features, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_ctg: torch.Tensor, x_maternal: torch.Tensor) -> torch.Tensor:
        ctg_emb = self.ctg_encoder(x_ctg)
        mat_emb = self.maternal_encoder(x_maternal)
        fused = torch.cat([ctg_emb, mat_emb], dim=1)
        return self.classifier(fused).squeeze(-1)


class GatedFusionModel(nn.Module):
    """CTG + maternal fusion via a learned per-record scalar gate.

    Unlike FusionModel's plain concatenation, a small gate MLP looks at both
    embeddings and outputs g in (0,1) that reweights each modality's
    contribution before concatenation. Fewer added parameters than
    cross-attention; the gate value itself is directly interpretable as
    per-record modality weighting.
    """

    def __init__(self, num_maternal_features: int, input_channels: int = 2, dropout: float = 0.3, **encoder_kwargs):
        super().__init__()
        ctg_dropout = encoder_kwargs.pop("dropout", dropout)
        self.ctg_encoder = CTGEncoder(input_channels=input_channels, dropout=ctg_dropout, **encoder_kwargs)
        self.maternal_encoder = MaternalEncoder(num_features=num_maternal_features, dropout=0.2)
        self.gate = nn.Sequential(
            nn.Linear(128 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode_and_gate(self, x_ctg: torch.Tensor, x_maternal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctg_emb = self.ctg_encoder(x_ctg)
        mat_emb = self.maternal_encoder(x_maternal)
        concat = torch.cat([ctg_emb, mat_emb], dim=1)
        gate = self.gate(concat)  # [B, 1] in (0, 1); weight on the CTG branch
        fused = torch.cat([gate * ctg_emb, (1.0 - gate) * mat_emb], dim=1)
        return fused, gate.squeeze(-1), concat

    def forward(self, x_ctg: torch.Tensor, x_maternal: torch.Tensor) -> torch.Tensor:
        fused, _gate, _concat = self.encode_and_gate(x_ctg, x_maternal)
        return self.classifier(fused).squeeze(-1)

    def forward_with_gate(self, x_ctg: torch.Tensor, x_maternal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Also returns the per-record gate value, for interpretability analysis."""
        fused, gate, _concat = self.encode_and_gate(x_ctg, x_maternal)
        return self.classifier(fused).squeeze(-1), gate

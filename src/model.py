"""Projection heads and the InfoNCE loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 128, hidden_dim: int | None = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # L2-normalise so the dot product is a cosine. Without it the model can
        # shrink the loss by inflating magnitudes instead of separating pairs.
        return F.normalize(self.net(x), dim=-1)


class AlignmentModel(nn.Module):
    def __init__(
        self,
        image_dim: int,
        expr_dim: int,
        embed_dim: int = 128,
        init_temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_head = ProjectionHead(image_dim, embed_dim, dropout=dropout)
        self.expr_head = ProjectionHead(expr_dim, embed_dim, dropout=dropout)

        # log(1/T) rather than T, as in CLIP: stays positive under unconstrained
        # SGD and is better conditioned.
        self.log_scale = nn.Parameter(
            torch.tensor(np.log(1.0 / init_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self):
        return float(1.0 / self.log_scale.detach().exp().clamp(max=100.0))

    def forward(self, images, expressions):
        return self.image_head(images), self.expr_head(expressions)

    def logit_scale(self):
        # Clamp at CLIP's ceiling. Unbounded, the model sharpens the softmax to
        # cut the loss instead of improving the embedding.
        return self.log_scale.exp().clamp(max=100.0)


def info_nce_loss(image_emb, expr_emb, logit_scale):
    """Symmetric InfoNCE; row i of each input is the same spot."""
    logits = logit_scale * image_emb @ expr_emb.T
    targets = torch.arange(len(logits), device=logits.device)

    # Both directions, since retrieval is evaluated both ways. Training one
    # alone leaves the other worse.
    loss_i2e = F.cross_entropy(logits, targets)
    loss_e2i = F.cross_entropy(logits.T, targets)
    loss = (loss_i2e + loss_e2i) / 2

    with torch.no_grad():
        stats = {
            "loss_i2e": float(loss_i2e.detach()),
            "loss_e2i": float(loss_e2i.detach()),
            "acc_i2e": float((logits.argmax(dim=1) == targets).float().mean()),
            "acc_e2i": float((logits.argmax(dim=0) == targets).float().mean()),
        }
    return loss, stats

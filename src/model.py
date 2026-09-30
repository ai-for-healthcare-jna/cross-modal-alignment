"""The only trainable part of the system: two small projection heads.

Both encoders are frozen — the image side is UNI, the expression side is just a
normalised gene vector. All this model does is learn two maps into a shared
128-d space where matched image/expression pairs sit close together.

Keeping the trainable surface this small is a deliberate choice. With a few
thousand spots from a handful of cohorts, a large trainable model would have
plenty of capacity to memorise cohort identity, and our whole question is about
generalisation. A two-layer MLP per modality is roughly the least expressive
thing that can still learn a useful alignment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """in_dim -> hidden -> embed_dim, L2-normalised on the way out.

    Normalising to the unit sphere makes the dot product a cosine similarity,
    which is what InfoNCE expects and what keeps the temperature parameter
    meaningful — without it, the model could shrink the loss by inflating vector
    magnitudes instead of actually separating pairs.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class AlignmentModel(nn.Module):
    """Two heads plus a learned temperature."""

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

        # Stored as log(1/T) and exponentiated at use time, following CLIP. In
        # log space the parameter stays positive under unconstrained gradient
        # descent, and the optimiser sees a better-conditioned surface than it
        # would optimising T directly.
        self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 / init_temperature), dtype=torch.float32))

    @property
    def temperature(self) -> float:
        return float(1.0 / self.log_scale.detach().exp().clamp(max=100.0))

    def forward(self, images: torch.Tensor, expressions: torch.Tensor):
        return self.image_head(images), self.expr_head(expressions)

    def logit_scale(self) -> torch.Tensor:
        # Clamped because an unbounded scale lets the model drive the loss down
        # by sharpening the softmax rather than improving the embedding. 100 is
        # CLIP's ceiling (temperature floor of 0.01).
        return self.log_scale.exp().clamp(max=100.0)


def info_nce_loss(
    image_emb: torch.Tensor, expr_emb: torch.Tensor, logit_scale: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric InfoNCE over a batch of paired embeddings.

    What a positive pair is
        One image patch and the expression vector measured at the *same physical
        spot* on the same tissue slice. Row i of `image_emb` and row i of
        `expr_emb` are the same location seen two ways. That pairing is free —
        it comes from the assay, not from any labelling effort — which is what
        makes this problem tractable without annotation.

    What the negatives are
        Every other pair in the batch. For image i we treat expressions
        j != i as wrong answers. Note these are only *approximately* negative:
        two spots of the same tissue type can be genuinely similar, so some
        "negatives" are near-duplicates and the loss can never reach zero. That
        noise is accepted in contrastive learning, but it is worth being aware
        of when reading absolute loss values.

    What the temperature does
        `logit_scale` is 1/T. It controls how sharply the softmax distinguishes
        the positive from the negatives. Low T (high scale) makes the loss focus
        almost entirely on the hardest negatives — fast separation, but unstable
        and prone to latching onto whatever superficial cue separates them
        quickest, which for us would be cohort identity. High T spreads gradient
        across all negatives and learns a smoother, more forgiving space. We
        learn it rather than fixing it, but clamp the ceiling.

    Why batch size matters
        The number of negatives *is* the batch size minus one. With a batch of
        16, picking the right answer out of 16 is easy and the loss saturates
        early with a weak embedding; with 1024 the task is genuinely hard and
        the gradient is informative. This is the main reason contrastive methods
        are batch-hungry. We can afford large batches precisely because both
        encoders are frozen and we train on cached vectors — the memory cost per
        example is a couple of KB, not a forward pass through a ViT.

    Symmetric because retrieval is evaluated in both directions: image->
    expression and expression->image. Training only one direction would leave
    the other systematically worse.
    """
    logits = logit_scale * image_emb @ expr_emb.T  # (B, B)
    # The correct match for row i is column i, so the targets are the diagonal.
    targets = torch.arange(len(logits), device=logits.device)

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

"""Contrastive training pieces, reused by the alignment step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from evaluate import embed_split, retrieval_metrics
from model import AlignmentModel, info_nce_loss


@dataclass
class TrainConfig:
    embed_dim: int = 128
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 15
    seed: int = 0
    device: str = "cpu"


def standardize(features, train_idx):
    """Zero-mean unit-variance per dimension, from training stats only."""
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def _batches(n, batch_size, rng):
    # Drop the short tail batch. InfoNCE difficulty scales with batch size, so
    # a stray batch of 3 contributes ~zero loss and flatters training accuracy.
    idx = rng.permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        yield idx[start : start + batch_size]


@torch.no_grad()
def val_score(model, image_features, expr_features, val_idx, device="cpu"):
    """Mean Recall@1 over both directions, used for early stopping."""
    img_emb, expr_emb = embed_split(
        model, image_features[val_idx], expr_features[val_idx], device=device
    )
    i2e = retrieval_metrics(img_emb, expr_emb)["recall@1"]
    e2i = retrieval_metrics(expr_emb, img_emb)["recall@1"]
    return (i2e + e2i) / 2


def train_alignment(image_features, expr_features, train_idx, val_idx, cfg, verbose=True):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = AlignmentModel(
        image_dim=image_features.shape[1],
        expr_dim=expr_features.shape[1],
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    ).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    img = torch.as_tensor(image_features[train_idx], dtype=torch.float32, device=cfg.device)
    expr = torch.as_tensor(expr_features[train_idx], dtype=torch.float32, device=cfg.device)
    batch_size = min(cfg.batch_size, len(train_idx))  # else _batches yields nothing

    history, best, best_state, stale = [], -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for b in _batches(len(train_idx), batch_size, rng):
            loss, _ = info_nce_loss(*model(img[b], expr[b]), model.logit_scale())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        score = val_score(model, image_features, expr_features, val_idx, cfg.device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_recall@1": score, "temperature": model.temperature})

        if score > best:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val R@1 {best:.3f})")
                break

    # Best checkpoint, not the last - val is small and noisy here.
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

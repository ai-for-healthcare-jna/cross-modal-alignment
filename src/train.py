"""Contrastive training on cached embeddings.

Runs on CPU in seconds once embed_images.py has populated the cache — the only
tensors involved are a few thousand rows of a couple of thousand floats. That
cheapness is the whole reason the caching step exists.

One model is trained per split fold. Everything that is fit from data — the
gene panel, the image feature standardisation — is fit on that fold's training
spots only and applied to val/test, so the leave-one-out numbers mean what they
claim to mean.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from data_loading import SpotData, build_expression_features, make_dummy_dataset
from embed_images import DEFAULT_ENCODER, load_cached_embeddings
from evaluate import evaluate_split, retrieval_metrics
from model import AlignmentModel, info_nce_loss
from splits import STRATEGIES, Split, build_splits


@dataclass
class TrainConfig:
    embed_dim: int = 128
    n_genes: int = 50
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 15  # epochs without val improvement before stopping
    seed: int = 0
    device: str = "cpu"


def standardize(features: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per dimension, using training statistics only.

    UNI's output dimensions have very different scales and offsets. Feeding
    them raw makes the first linear layer spend its capacity undoing that.
    Statistics come from train because computing them over the held-out cohort
    would let the model see the test distribution's mean — a small leak, but the
    kind that matters when the claim is specifically about cross-cohort
    transfer.
    """
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    """Shuffled batches, dropping a trailing short batch.

    drop_last matters here in a way it doesn't for supervised training: InfoNCE
    difficulty is set by batch size, so a stray batch of 3 would contribute a
    near-zero loss and a misleadingly high training accuracy.
    """
    idx = rng.permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        yield idx[start : start + batch_size]


@torch.no_grad()
def _val_score(model, image_features, expr_features, split: Split, device: str) -> float:
    """Mean Recall@1 over both directions on the val fold — the early-stopping signal.

    Retrieval rather than loss, because val loss depends on val set size (more
    candidates = harder) and isn't comparable across folds of different sizes.
    """
    from evaluate import embed_split

    img_emb, expr_emb = embed_split(
        model, image_features[split.val], expr_features[split.val], device=device
    )
    i2e = retrieval_metrics(img_emb, expr_emb)["recall@1"]
    e2i = retrieval_metrics(expr_emb, img_emb)["recall@1"]
    return (i2e + e2i) / 2


def train_one_split(
    image_features: np.ndarray,
    expr_features: np.ndarray,
    split: Split,
    cfg: TrainConfig,
    verbose: bool = True,
) -> tuple[AlignmentModel, list[dict]]:
    """Train a single alignment model on one fold."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device

    model = AlignmentModel(
        image_dim=image_features.shape[1],
        expr_dim=expr_features.shape[1],
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    img_train = torch.as_tensor(image_features[split.train], dtype=torch.float32, device=device)
    expr_train = torch.as_tensor(expr_features[split.train], dtype=torch.float32, device=device)

    # A batch larger than the training set would silently produce zero batches.
    batch_size = min(cfg.batch_size, len(split.train))

    history: list[dict] = []
    best_score, best_state, epochs_since_best = -1.0, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for batch_idx in _batches(len(split.train), batch_size, rng):
            img_emb, expr_emb = model(img_train[batch_idx], expr_train[batch_idx])
            loss, _ = info_nce_loss(img_emb, expr_emb, model.logit_scale())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        val_score = _val_score(model, image_features, expr_features, split, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_recall@1": val_score,
            "temperature": model.temperature,
        })

        if val_score > best_score:
            best_score, epochs_since_best = val_score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                if verbose:
                    print(f"      early stop at epoch {epoch} (best val R@1 {best_score:.3f})")
                break

    # Restore the best checkpoint rather than the last one — with small val
    # folds the score is noisy and the final epoch is often not the best.
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_strategy(
    data: SpotData,
    image_features_raw: np.ndarray,
    strategy: str,
    cfg: TrainConfig,
    verbose: bool = True,
) -> list[dict]:
    """Train and evaluate every fold of one split strategy."""
    results = []
    for split in build_splits(data.metadata, strategy, seed=cfg.seed):
        # Refit per fold: both of these are data-dependent and must not see test.
        expr_features, genes = build_expression_features(data, split.train, n_genes=cfg.n_genes)
        image_features = standardize(image_features_raw, split.train)

        if verbose:
            print(f"    fold '{split.name}' {split.sizes()}")
        model, history = train_one_split(image_features, expr_features, split, cfg, verbose)

        metrics = evaluate_split(
            model, image_features, expr_features, data.metadata, split.test,
            device=cfg.device,
        )
        results.append({
            "strategy": strategy,
            "fold": split.name,
            "held_out": split.held_out,
            "in_distribution_val": split.in_distribution_val,
            **split.sizes(),
            "epochs_run": len(history),
            "n_genes": len(genes),
            **metrics,
        })
        if verbose:
            print(f"      test i2e R@1 {metrics['i2e_recall@1']:.3f} "
                  f"(chance {metrics['i2e_chance_recall@1']:.3f}) | "
                  f"silhouette organ {metrics['silhouette_organ']:.3f} "
                  f"cohort {metrics['silhouette_cohort']:.3f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cross-modal alignment heads.")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--samples", nargs="+", help="HEST sample ids (omit to use dummy data)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--embeddings", default="embeddings")
    parser.add_argument("--out", default="results/metrics.json")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      seed=args.seed, device=args.device)

    if args.samples:
        from data_loading import load_hest_subset

        data = load_hest_subset(args.samples, args.data_dir)
        image_features = load_cached_embeddings(
            data.metadata["spot_id"].tolist(), encoder=args.encoder, root=args.embeddings
        )
    else:
        print("no --samples given: running on dummy data")
        data = make_dummy_dataset()
        image_features = np.random.default_rng(cfg.seed).normal(
            size=(len(data), 1024)).astype(np.float32)

    print(data.summary().to_string(index=False))

    results = []
    for strategy in args.strategies:
        print(f"\n  strategy: {strategy}")
        results.extend(run_strategy(data, image_features, strategy, cfg))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": asdict(cfg), "results": results}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

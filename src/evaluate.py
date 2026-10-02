"""Retrieval and clustering metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

RECALL_KS = (1, 5, 10)


def _cosine_sim(a, b):
    # numpy linked against Apple Accelerate throws spurious "divide by zero in
    # matmul" on finite input - it fires on x @ x.T for random normals. Check
    # finiteness ourselves so the real thing still surfaces, then mute BLAS.
    # Drop the errstate on an OpenBLAS build.
    assert np.isfinite(a).all() and np.isfinite(b).all(), "non-finite embeddings"
    with np.errstate(all="ignore"):
        return a @ b.T


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def retrieval_metrics(query_emb, gallery_emb, ks=RECALL_KS):
    """Recall@k and MRR where query i should retrieve gallery i."""
    n = len(query_emb)
    sim = _cosine_sim(query_emb, gallery_emb)
    correct = np.diag(sim)[:, None]

    # Count how many score strictly higher than the true match. Cheaper than
    # argsort and ties don't get an arbitrary winner.
    ranks = (sim > correct).sum(axis=1) + 1

    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks if k <= n}
    out["mrr"] = float((1.0 / ranks).mean())
    # Recall@1 means nothing without gallery size: 0.05 is strong at n=1000 and
    # worthless at n=10.
    out["n_candidates"] = float(n)
    out["chance_recall@1"] = float(1.0 / n)
    return out


def cross_modal_retrieval(image_emb, expr_emb):
    """Retrieval in both directions; an asymmetry is itself informative."""
    out = {}
    for prefix, (q, g) in {"i2e": (image_emb, expr_emb), "e2i": (expr_emb, image_emb)}.items():
        for name, value in retrieval_metrics(q, g).items():
            out[f"{prefix}_{name}"] = value
    return out


def safe_silhouette(embeddings, labels):
    labels = np.asarray(labels)
    n_labels = len(np.unique(labels))
    # The pilot has one platform, so that silhouette is undefined. Return NaN
    # rather than raising - it should read as "not measurable" in the table.
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    with np.errstate(all="ignore"):
        return float(silhouette_score(embeddings, labels, metric="cosine"))


def structure_metrics(embeddings, metadata):
    return {
        "silhouette_organ": safe_silhouette(embeddings, metadata["organ"].to_numpy()),
        "silhouette_cohort": safe_silhouette(embeddings, metadata["cohort_id"].to_numpy()),
        "silhouette_platform": safe_silhouette(embeddings, metadata["platform"].to_numpy()),
    }


def raw_baseline(image_features, expr_features, metadata, n_sample=2000, seed=0):
    """What's already there before training anything.

    Cross-modal Recall@k on raw features isn't defined - image features are
    1024-d and expression vectors 50-d, so there is no shared space to rank in
    without fitting a map, and fitting one is the thing we're trying to
    baseline. What needs no fitting is neighbourhood agreement: if morphology
    and expression covary at all, spots that look alike should also have
    similar expression.
    """
    rng = np.random.default_rng(seed)
    n = len(image_features)
    idx = rng.choice(n, min(n_sample, n), replace=False)  # pairwise, so cap it

    img = _unit(image_features[idx])
    expr = _unit(expr_features[idx])
    s_img = _cosine_sim(img, img)
    s_expr = _cosine_sim(expr, expr)

    off = ~np.eye(len(idx), dtype=bool)
    rho = spearmanr(s_img[off], s_expr[off]).statistic

    # Rank of each spot's nearest expression neighbour within its image-space
    # ranking. Self-similarity excluded or it would trivially rank first.
    np.fill_diagonal(s_img, -np.inf)
    np.fill_diagonal(s_expr, -np.inf)
    top_expr = s_expr.argmax(axis=1)
    img_rank = (s_img > s_img[np.arange(len(idx)), top_expr][:, None]).sum(axis=1) + 1

    out = {
        "raw_structure_spearman": float(rho),
        "raw_n_sample": float(len(idx)),
        "silhouette_organ": safe_silhouette(_unit(image_features), metadata["organ"].to_numpy()),
        "silhouette_cohort": safe_silhouette(_unit(image_features), metadata["cohort_id"].to_numpy()),
        "silhouette_platform": float("nan"),
    }
    for k in RECALL_KS:
        out[f"raw_neighbour_recall@{k}"] = float((img_rank <= k).mean())
        out[f"raw_neighbour_chance@{k}"] = float(k / (len(idx) - 1))
    return out


@torch.no_grad()
def embed_split(model, image_features, expr_features, device="cpu"):
    model.eval().to(device)
    img = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    expr = torch.as_tensor(expr_features, dtype=torch.float32, device=device)
    image_emb, expr_emb = model(img, expr)
    return image_emb.cpu().numpy(), expr_emb.cpu().numpy()


def evaluate_split(model, image_features, expr_features, metadata, test_idx, device="cpu"):
    image_emb, expr_emb = embed_split(model, image_features, expr_features, device=device)

    metrics = cross_modal_retrieval(image_emb[test_idx], expr_emb[test_idx])

    # Silhouette over all spots, not just test: under cross_organ the test set
    # is a single organ, so an organ silhouette there has one label and is
    # undefined. It's a question about the whole space anyway.
    shared = np.vstack([image_emb, expr_emb])
    doubled = pd.concat([metadata, metadata], ignore_index=True)
    metrics.update(structure_metrics(shared, doubled))
    return metrics

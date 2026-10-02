"""Retrieval and clustering metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
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

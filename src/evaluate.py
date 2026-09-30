"""Two families of metric: does retrieval work, and what is it keying on.

Retrieval tells us whether the alignment learned anything. Silhouette tells us
what the shared space is organised by — and that second question is the point
of the project, because a model can score well on in-distribution retrieval
purely by recognising which cohort a spot came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

RECALL_KS = (1, 5, 10)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a @ b.T, with a check for real non-finite values and a guard for fake ones.

    numpy built against Apple's Accelerate BLAS raises spurious "divide by zero
    encountered in matmul" warnings on perfectly finite float input — it fires
    on `x @ x.T` for plain random normals, and the results are correct. Rather
    than let that noise bury a genuine problem, we assert the inputs are finite
    ourselves and then silence the BLAS-level warning. Drop the errstate if this
    ever runs on a numpy linked against OpenBLAS.
    """
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite values in embeddings — training likely diverged")
    with np.errstate(all="ignore"):
        return a @ b.T


def retrieval_metrics(
    query_emb: np.ndarray, gallery_emb: np.ndarray, ks: tuple[int, ...] = RECALL_KS
) -> dict[str, float]:
    """Recall@k and MRR for matched rows: query i should retrieve gallery i.

    Both inputs are already L2-normalised by the projection heads, so a dot
    product is cosine similarity and ranking by it is ranking by closeness.
    """
    n = len(query_emb)
    if n != len(gallery_emb):
        raise ValueError("query and gallery must be aligned row-for-row")

    sim = _cosine_sim(query_emb, gallery_emb)
    correct = np.diag(sim)[:, None]
    # Rank = how many gallery items score strictly higher than the true match,
    # plus one. Computed this way rather than by argsort because it is O(n^2)
    # instead of O(n^2 log n) and handles ties without an arbitrary tie-break.
    ranks = (sim > correct).sum(axis=1) + 1

    metrics = {f"recall@{k}": float((ranks <= k).mean()) for k in ks if k <= n}
    metrics["mrr"] = float((1.0 / ranks).mean())
    # Chance level depends on gallery size, so report it — Recall@1 of 0.05 is
    # excellent with 1000 candidates and meaningless with 10.
    metrics["n_candidates"] = float(n)
    metrics["chance_recall@1"] = float(1.0 / n)
    return metrics


def cross_modal_retrieval(image_emb: np.ndarray, expr_emb: np.ndarray) -> dict[str, float]:
    """Retrieval in both directions. They can differ — one modality is often
    more discriminative than the other, and an asymmetry is itself informative.
    """
    out = {}
    for prefix, (q, g) in {
        "i2e": (image_emb, expr_emb),
        "e2i": (expr_emb, image_emb),
    }.items():
        for name, value in retrieval_metrics(q, g).items():
            out[f"{prefix}_{name}"] = value
    return out


def safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score, or NaN when the label set can't support one.

    Guards rather than raises because with leave-one-out splits a test fold can
    legitimately contain a single organ or a single platform, and that should
    show up as "not measurable" in the results table rather than crash the run.
    """
    labels = np.asarray(labels)
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    # errstate for the same Accelerate BLAS false positive as _cosine_sim,
    # raised here from inside sklearn's pairwise distance computation.
    with np.errstate(all="ignore"):
        return float(silhouette_score(embeddings, labels, metric="cosine"))


def structure_metrics(embeddings: np.ndarray, metadata) -> dict[str, float]:
    """How well is the shared space clustered by biology vs by technical batch?

    Silhouette ranges from -1 to 1: high means points sit close to their own
    label group and far from others, ~0 means the labels don't explain the
    geometry at all.

    Reading the numbers:

      high organ silhouette      the space is organised by tissue type. This is
                                 what we want — it means the alignment captured
                                 something biological.

      high cohort or platform    the space is organised by *where the sample was
      silhouette                 processed*. This is the failure mode the whole
                                 project is built to detect. It means the model
                                 found that spots from the same study share a
                                 stain colour, scanner profile or sequencing
                                 depth signature, and used that to match pairs
                                 instead of learning any image/expression
                                 relationship. Retrieval scores can look strong
                                 while this is happening, which is exactly why
                                 we measure it separately.

    The two are not cleanly separable in HEST: cohorts are not organ-balanced,
    so a cohort can act as a proxy for its organ and inflate the technical
    number. Treat a high cohort silhouette alongside a high organ silhouette as
    ambiguous; a high cohort silhouette with a *low* organ silhouette is the
    clear bad case.
    """
    return {
        "silhouette_organ": safe_silhouette(embeddings, metadata["organ"].to_numpy()),
        "silhouette_cohort": safe_silhouette(embeddings, metadata["cohort_id"].to_numpy()),
        "silhouette_platform": safe_silhouette(embeddings, metadata["platform"].to_numpy()),
    }


@torch.no_grad()
def embed_split(model, image_features: np.ndarray, expr_features: np.ndarray,
                device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run the projection heads over a whole split in eval mode."""
    model.eval().to(device)
    img = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    expr = torch.as_tensor(expr_features, dtype=torch.float32, device=device)
    image_emb, expr_emb = model(img, expr)
    return image_emb.cpu().numpy(), expr_emb.cpu().numpy()


def evaluate_split(
    model, image_features: np.ndarray, expr_features: np.ndarray, metadata,
    test_idx: np.ndarray, device: str = "cpu",
) -> dict[str, float]:
    """Full metric set for one fold.

    The two metric families are deliberately computed over different sets.

    Retrieval is a held-out measurement, so it runs on the test fold only.

    Silhouette runs over *all* spots — train and test together. It has to: under
    leave-one-cohort-out the test fold contains exactly one cohort, so a
    cohort silhouette computed there has a single label and is undefined. The
    question "is this space organised by batch or by biology?" is a question
    about the geometry of the whole embedding space, not about held-out
    prediction, so evaluating it on everything is both necessary and correct.
    """
    image_emb, expr_emb = embed_split(model, image_features, expr_features, device=device)

    metrics = cross_modal_retrieval(image_emb[test_idx], expr_emb[test_idx])

    # Both modalities stacked into one point cloud: if the alignment worked, a
    # patch and its own expression vector are neighbours, so it makes sense to
    # ask how that single cloud is clustered.
    shared = np.vstack([image_emb, expr_emb])
    doubled = pd.concat([metadata, metadata], ignore_index=True)
    metrics.update(structure_metrics(shared, doubled))
    return metrics

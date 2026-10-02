"""Steps 2-3: contrastive alignment per fold, then kNN gene prediction and retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import load_fold_side, pearson_per_gene
from data_loading import BENCH_ROOT, N_FOLDS, load_gene_list, load_split
from embed_images import DEFAULT_ENCODER
from evaluate import cross_modal_retrieval, embed_split
from train import TrainConfig, standardize, train_alignment

KNN_KS = (10, 50, 100)
VAL_FRAC = 0.1
# Test patients range from 3,005 to 20,761 spots and Recall@k depends on gallery
# size, so retrieval is scored on equal-sized subsets instead.
SUBSET_SIZE = 1000
N_DRAWS = 10


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_predict(query, gallery, y_gallery, ks=KNN_KS, chunk=2048):
    """For each query row, average y_gallery over its k most cosine-similar gallery rows."""
    q, g = _l2(query), _l2(gallery)
    k_max = max(ks)
    out = {k: np.empty((len(q), y_gallery.shape[1]), dtype=np.float64) for k in ks}

    # Chunked: fold 0 would otherwise need a 20,761 x 14,775 similarity matrix.
    with np.errstate(all="ignore"):  # Accelerate BLAS false positive
        for s in range(0, len(q), chunk):
            sim = q[s : s + chunk] @ g.T
            top = np.argpartition(-sim, k_max, axis=1)[:, :k_max]
            rows = np.arange(len(top))[:, None]
            # argpartition leaves the k_max unordered; sort so a prefix is a true top-k.
            top = top[rows, np.argsort(-sim[rows, top], axis=1)]
            for k in ks:
                out[k][s : s + chunk] = y_gallery[top[:, :k]].mean(axis=1)
    return out


def score_genes(y_true, y_pred):
    """Mean Pearson over genes, scored as in step 1."""
    per_gene = pearson_per_gene(y_true, y_pred)
    # A kNN prediction can be constant for a near-silent gene; that is no
    # correlation, not a missing measurement.
    per_gene = np.nan_to_num(per_gene, nan=0.0)
    return float(per_gene.mean()), per_gene


def subset_retrieval(image_emb, expr_emb, rng):
    """Retrieval averaged over random equal-sized subsets of the test patient."""
    n = min(SUBSET_SIZE, len(image_emb))
    draws = [
        cross_modal_retrieval(image_emb[sel], expr_emb[sel])
        for sel in (rng.choice(len(image_emb), n, replace=False) for _ in range(N_DRAWS))
    ]
    return {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}


def run_fold(fold, genes, root, encoder, embeddings_root, cfg, verbose=True):
    X_train, y_train = load_fold_side(load_split(fold, "train", root), genes, root,
                                      encoder, embeddings_root)
    X_test, y_test = load_fold_side(load_split(fold, "test", root), genes, root,
                                    encoder, embeddings_root)
    n_train = len(X_train)
    if verbose:
        print(f"\nfold {fold}: {n_train} train / {len(X_test)} test spots")

    # Standardise on the training fold only; the test patient never informs the stats.
    X = np.vstack([X_train, X_test])
    Y = np.vstack([y_train, y_test])
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(X))
    Xs, Ys = standardize(X, train_idx), standardize(Y, train_idx)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_train)
    n_val = int(round(VAL_FRAC * n_train))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    model, history = train_alignment(Xs, Ys, fit_idx, val_idx, cfg, verbose=verbose)

    # Gallery is every training spot, including the 10% held out for early stopping.
    _, gallery_expr_emb = embed_split(model, Xs[train_idx], Ys[train_idx], device=cfg.device)
    test_image_emb, test_expr_emb = embed_split(model, Xs[test_idx], Ys[test_idx],
                                                device=cfg.device)

    knn_raw = knn_predict(X_test, X_train, y_train)
    knn_aligned = knn_predict(test_image_emb, gallery_expr_emb, y_train)

    result = {
        "fold": fold,
        "n_train": n_train,
        "n_test": len(X_test),
        "n_val": int(n_val),
        "epochs_run": len(history),
        "best_val_recall@1": max(h["val_recall@1"] for h in history),
        "temperature": model.temperature,
        "knn_raw": {}, "knn_aligned": {},
        "retrieval": subset_retrieval(test_image_emb, test_expr_emb, rng),
    }
    for k in KNN_KS:
        mean_raw, per_raw = score_genes(y_test, knn_raw[k])
        mean_al, per_al = score_genes(y_test, knn_aligned[k])
        result["knn_raw"][str(k)] = {"pearson_mean": mean_raw,
                                     "per_gene": dict(zip(genes, per_raw.tolist()))}
        result["knn_aligned"][str(k)] = {"pearson_mean": mean_al,
                                         "per_gene": dict(zip(genes, per_al.tolist()))}
        if verbose:
            print(f"  k={k:<4} raw {mean_raw:.4f}   aligned {mean_al:.4f}")
    return result


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def print_tables(folds, ridge_folds):
    n = len(folds)
    header = f"{'method':<22}" + "".join(f"{'fold ' + str(i):>10}" for i in range(n)) + f"{'mean ± std':>18}"

    print("\n=== Gene prediction (Pearson, mean over 50 genes) ===")
    print(header)
    rows = [("ridge (step 1)", ridge_folds)]
    for name, key in [("kNN raw image", "knn_raw"), ("kNN aligned", "knn_aligned")]:
        for k in KNN_KS:
            rows.append((f"{name}, k={k}", [f[key][str(k)]["pearson_mean"] for f in folds]))
    for label, values in rows:
        m, s = _mean_std(values)
        print(f"{label:<22}" + "".join(f"{v:>10.4f}" for v in values) + f"{m:>12.4f} ± {s:.4f}")

    print(f"\n=== Retrieval on the held-out patient ({SUBSET_SIZE} spots, {N_DRAWS} draws) ===")
    print(header)
    metrics = ["recall@1", "recall@5", "recall@10", "mrr"]
    for direction, label in [("i2e", "image→expr"), ("e2i", "expr→image")]:
        for metric in metrics:
            values = [f["retrieval"][f"{direction}_{metric}"] for f in folds]
            m, s = _mean_std(values)
            print(f"{label + ' ' + metric:<22}" + "".join(f"{v:>10.4f}" for v in values)
                  + f"{m:>12.4f} ± {s:.4f}")
    chance = [f["retrieval"]["i2e_chance_recall@1"] for f in folds]
    print(f"{'chance recall@1':<22}" + "".join(f"{v:>10.4f}" for v in chance))


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate the IDC alignment.")
    ap.add_argument("--root", default=str(BENCH_ROOT))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--embeddings", default="embeddings")
    ap.add_argument("--benchmark", default="results/benchmark_idc.json")
    ap.add_argument("--out", default="results/alignment_idc.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = ap.parse_args()

    # One config for every fold - per-fold tuning on 4 patients would just fit noise.
    cfg = TrainConfig(device=args.device, epochs=args.epochs)
    genes = load_gene_list(args.root)
    folds = [run_fold(f, genes, args.root, args.encoder, args.embeddings, cfg)
             for f in range(N_FOLDS)]

    ridge = json.loads(Path(args.benchmark).read_text())
    ridge_folds = [f["pearson_mean"] for f in ridge["folds"]]
    print_tables(folds, ridge_folds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "encoder": args.encoder,
        "config": vars(cfg),
        "ridge_folds": ridge_folds,
        "folds": folds,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

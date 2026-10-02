"""HEST-Benchmark regression baseline: PCA + ridge, scored by Pearson per gene.

Mirrors hest/bench (trainer.py, benchmark.py) without installing the package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loading import (
    BENCH_ROOT,
    N_FOLDS,
    load_gene_list,
    load_split,
    read_expression,
    read_patch_barcodes,
)
from embed_images import DEFAULT_ENCODER, load_cached_embeddings

# Benchmark defaults from its BenchmarkConfig.
LATENT_DIM = 256
SEED = 1


def load_fold_side(split_df, genes, root, encoder, embeddings_root):
    """Stack embeddings and expression for every sample in one split."""
    xs, ys = [], []
    for _, row in split_df.iterrows():
        sid = row.sample_id
        barcodes = read_patch_barcodes(Path(root) / row.patches_path)
        xs.append(load_cached_embeddings(
            [f"{sid}:{b}" for b in barcodes], encoder=encoder, root=embeddings_root
        ))
        ys.append(read_expression(Path(root) / row.expr_path, genes, barcodes))
    return np.vstack(xs), np.vstack(ys)


def fit_predict(X_train, y_train, X_test):
    # errstate: numpy on Apple Accelerate emits bogus matmul FP warnings from
    # inside sklearn. Values are fine.
    with np.errstate(all="ignore"):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=LATENT_DIM, random_state=SEED)),
        ])
        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)

        # The benchmark derives alpha from the shapes rather than tuning it.
        alpha = 100 / (X_train.shape[1] * y_train.shape[1])
        reg = Ridge(solver="lsqr", alpha=alpha, fit_intercept=False,
                    max_iter=1000, random_state=SEED)
        reg.fit(X_train, y_train)
        return reg.predict(X_test), alpha


def pearson_per_gene(y_true, y_pred):
    with np.errstate(all="ignore"):  # Accelerate BLAS false positive, as above
        return np.array([pearsonr(y_true[:, g], y_pred[:, g])[0]
                         for g in range(y_true.shape[1])])


def run_fold(fold, genes, root, encoder, embeddings_root):
    X_train, y_train = load_fold_side(load_split(fold, "train", root), genes, root,
                                      encoder, embeddings_root)
    X_test, y_test = load_fold_side(load_split(fold, "test", root), genes, root,
                                    encoder, embeddings_root)

    preds, alpha = fit_predict(X_train, y_train, X_test)
    per_gene = pearson_per_gene(y_test, preds)
    return {
        "fold": fold,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "alpha": alpha,
        "pearson_mean": float(np.mean(per_gene)),
        "pearson_std_genes": float(np.std(per_gene)),
        "per_gene": {g: float(v) for g, v in zip(genes, per_gene)},
    }


def main():
    ap = argparse.ArgumentParser(description="Reproduce the HEST-Benchmark IDC baseline.")
    ap.add_argument("--root", default=str(BENCH_ROOT))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--embeddings", default="embeddings")
    ap.add_argument("--out", default="results/benchmark_idc.json")
    args = ap.parse_args()

    genes = load_gene_list(args.root)
    folds = [run_fold(f, genes, args.root, args.encoder, args.embeddings)
             for f in range(N_FOLDS)]

    means = [f["pearson_mean"] for f in folds]
    print(f"\n{'fold':<6}{'n_train':>9}{'n_test':>8}{'pearson':>10}")
    for f in folds:
        print(f"{f['fold']:<6}{f['n_train']:>9}{f['n_test']:>8}{f['pearson_mean']:>10.4f}")
    # Across folds, matching the benchmark's merge_fold_results: mean and std
    # of the per-fold means, not of the per-gene values.
    print(f"\nPearson mean +- std across folds: {np.mean(means):.4f} +- {np.std(means):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "encoder": args.encoder,
        "pearson_mean": float(np.mean(means)),
        "pearson_std": float(np.std(means)),
        "folds": folds,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

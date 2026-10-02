"""Load the HEST-Benchmark IDC task."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

BENCH_ROOT = Path("data/hest-bench/IDC")
GENE_LIST = "var_50genes.json"
N_FOLDS = 4


def load_gene_list(root=BENCH_ROOT):
    """The 50 target genes the benchmark fixes for this task."""
    with open(Path(root) / GENE_LIST) as f:
        return json.load(f)["genes"]


def load_split(fold, which, root=BENCH_ROOT):
    """One official split CSV. Folds are patient-stratified."""
    return pd.read_csv(Path(root) / "splits" / f"{which}_{fold}.csv")


def read_patch_barcodes(path):
    with h5py.File(path, "r") as f:
        return [_decode(b) for b in f["barcode"][:].ravel()]


def _decode(x):
    # h5py hands back bytes; str() on those gives "b'AAACCT'", which still
    # looks like a valid id and then silently fails to match anything.
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def read_expression(expr_path, genes, barcodes):
    """log1p counts for `genes`, rows ordered to match `barcodes`.

    log1p only, no library-size scaling. The benchmark's normalize_adata
    docstring claims it scales by total counts but the code just calls
    sc.pp.log1p, and the published numbers come from the code.
    """
    with h5py.File(expr_path, "r") as f:
        X = f["X"]
        # hest-bench stores X dense; the main HEST repo stores it CSR.
        if isinstance(X, h5py.Dataset):
            counts = X[:]
        else:
            counts = csr_matrix(
                (X["data"][:], X["indices"][:], X["indptr"][:]),
                shape=tuple(X.attrs["shape"]),
            ).toarray()
        all_genes = [_decode(g) for g in f["var"]["_index"][:]]
        all_barcodes = [_decode(b) for b in f["obs"]["_index"][:]]

    gene_pos = {g: i for i, g in enumerate(all_genes)}
    row_pos = {b: i for i, b in enumerate(all_barcodes)}
    sub = counts[np.ix_([row_pos[b] for b in barcodes], [gene_pos[g] for g in genes])]
    return np.log1p(sub.astype(np.float64))


def normalize_counts(counts):
    """Scale each spot to 10,000 counts, then log1p.

    Kept for the alignment step; the benchmark baseline uses log1p alone.
    """
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(counts / totals * 1e4).astype(np.float32)

"""Load HEST cohorts and build the expression vectors."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HEST_REPO = "MahmoodLab/hest"
METADATA_CSV = "HEST_v1_3_0.csv"

HEST_COLUMNS = {
    "id": "sample_id",
    "dataset_title": "cohort_id",
    "organ": "organ",
    "st_technology": "platform",
    "spots_under_tissue": "n_spots",
    "nb_genes": "n_genes_measured",
    "species": "species",
}

# HEST has a few targeted panels (one is 1,056 genes) that can't share a gene
# panel with whole-transcriptome cohorts.
MIN_GENES_MEASURED = 5000


@dataclass
class SpotData:
    """Spots pooled across cohorts; metadata and counts align row by row."""

    metadata: pd.DataFrame
    counts: np.ndarray
    genes: list[str]

    def __len__(self):
        return len(self.metadata)

    def summary(self):
        return (
            self.metadata.groupby(["cohort_id", "organ", "platform"], as_index=False)
            .agg(n_spots=("spot_id", "size"))
            .reset_index(drop=True)
        )


def load_hest_metadata(data_dir="data"):
    """HEST's per-sample metadata, renamed to our column names."""
    path = Path(data_dir) / METADATA_CSV
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(HEST_REPO, METADATA_CSV, repo_type="dataset"), path)

    raw = pd.read_csv(path, low_memory=False)
    return raw.rename(columns=HEST_COLUMNS)[list(HEST_COLUMNS.values())]


def select_starter_cohorts(meta, max_spots=6000):
    """Largest breast and prostate Visium cohorts under max_spots."""
    visium = meta[
        (meta.species == "Homo sapiens")
        & (meta.platform == "Visium")
        & (meta.n_genes_measured >= MIN_GENES_MEASURED)
    ]
    cohorts = visium.groupby(["cohort_id", "organ"], as_index=False).agg(
        n_samples=("sample_id", "nunique"), n_spots=("n_spots", "sum")
    )

    picks = []
    for organ in ("Breast", "Prostate"):
        options = cohorts[(cohorts.organ == organ) & (cohorts.n_spots < max_spots)]
        picks.append(options.loc[options.n_spots.idxmax()])
    return pd.DataFrame(picks).reset_index(drop=True)


def read_counts(path):
    """Raw counts, gene names and barcodes from a HEST .h5ad."""
    # h5py not anndata: .h5ad is plain HDF5 and we need three arrays, which
    # isn't worth the dependency.
    with h5py.File(path, "r") as f:
        X = f["X"]
        shape = tuple(X.attrs["shape"])
        counts = csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=shape)
        genes = [g.decode() for g in f["var"]["_index"][:]]
        barcodes = [b.decode() for b in f["obs"]["_index"][:]]
    return counts.toarray().astype(np.float32), genes, barcodes


def load_cohorts(sample_ids, data_dir="data/hest", meta=None):
    """Pool the given HEST samples into one SpotData, intersecting genes."""
    data_dir = Path(data_dir)
    meta = (load_hest_metadata() if meta is None else meta).set_index("sample_id")

    frames, blocks = [], []
    for sid in sample_ids:
        counts, genes, barcodes = read_counts(data_dir / "st" / f"{sid}.h5ad")
        patch_file = data_dir / "patches" / f"{sid}.h5"

        with h5py.File(patch_file, "r") as f:
            patch_barcodes = [b.decode() for b in f["barcode"][:].ravel()]

        # Match on barcode rather than row order, and drop spots with no patch:
        # HEST skips patches that would run off the slide edge, so the .h5ad can
        # have slightly more spots than the patch file (19 of 4,992 in NCBI776).
        where = {b: i for i, b in enumerate(patch_barcodes)}
        keep = [i for i, b in enumerate(barcodes) if b in where]
        barcodes = [barcodes[i] for i in keep]
        counts = counts[keep]
        patch_index = [where[b] for b in barcodes]

        row = meta.loc[sid]
        frames.append(pd.DataFrame({
            "spot_id": [f"{sid}:{b}" for b in barcodes],
            "sample_id": sid,
            "cohort_id": row.cohort_id,
            "organ": row.organ,
            "platform": row.platform,
            "patch_file": str(patch_file),
            "patch_index": patch_index,
        }))
        blocks.append(pd.DataFrame(counts, columns=genes))

    shared = sorted(set.intersection(*(set(b.columns) for b in blocks)))
    return SpotData(
        metadata=pd.concat(frames, ignore_index=True),
        counts=np.vstack([b[shared].to_numpy(np.float32) for b in blocks]),
        genes=shared,
    )


def normalize_counts(counts):
    """Scale each spot to 10,000 counts, then log1p.

    Without the rescale, cosine similarity would partly be measuring
    sequencing depth rather than biology.
    """
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(counts / totals * 1e4).astype(np.float32)


def select_hvg(normalized, train_idx, n_genes=50):
    """Most variable genes, measured on training spots only.

    Our own panel rather than HEST-Benchmark's: theirs is picked per organ, so
    gene 7 in a breast task isn't gene 7 in a prostate one. Train-only because
    picking genes with the test organ visible leaks what we're measuring.
    """
    variances = normalized[train_idx].var(axis=0)
    return np.sort(np.argsort(variances)[-n_genes:])


def build_expression_features(data, train_idx, n_genes=50):
    """Counts -> the normalised gene vector used as the expression embedding."""
    normalized = normalize_counts(data.counts)
    genes = select_hvg(normalized, train_idx, n_genes)
    return normalized[:, genes], [data.genes[i] for i in genes]


def make_dummy_dataset(n_per_cohort=20, n_genes=200, seed=0):
    """Fake SpotData shaped like the pilot, for the smoke test."""
    rng = np.random.default_rng(seed)
    cohorts = [("cohort_breast", "Breast"), ("cohort_prostate", "Prostate")]

    rows, blocks = [], []
    for i, (cohort, organ) in enumerate(cohorts):
        # Per-cohort rate offset: a synthetic batch effect for the silhouette
        # to respond to.
        blocks.append(rng.poisson(2.0 + i, size=(n_per_cohort, n_genes)))
        rows += [{
            "spot_id": f"{cohort}:spot{s}",
            "sample_id": f"{cohort}_s0",
            "cohort_id": cohort,
            "organ": organ,
            "platform": "Visium",
            "patch_file": f"dummy/{cohort}.h5",
            "patch_index": s,
        } for s in range(n_per_cohort)]

    return SpotData(
        metadata=pd.DataFrame(rows),
        counts=np.vstack(blocks).astype(np.float32),
        genes=[f"GENE{i}" for i in range(n_genes)],
    )

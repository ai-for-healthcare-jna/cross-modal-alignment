"""Loading HEST-1k spots and preparing the expression side of each pair.

A "spot" is one physical location on a tissue slice. HEST-1k gives us, for that
same location, a 224x224 image patch at 0.5 um/px and a gene count vector. Those
two views of one spot are the positive pair the contrastive model learns from.

This module produces a single flat table of spots pooled across samples, plus
the raw count matrix. Expression normalisation and gene selection live here too,
but they are deliberately *not* applied at load time: the gene panel has to be
refit per split (see `select_hvg`), so the loader hands back raw counts and the
training script decides what to do with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# How HEST names the fields we care about, mapped in one place so the rest of
# the codebase speaks our vocabulary rather than HEST's.
HEST_SAMPLE_COL = "id"
HEST_COHORT_COL = "dataset_title"  # a "cohort" here = one HEST study/submission
HEST_ORGAN_COL = "organ"
HEST_PLATFORM_COL = "st_technology"

METADATA_COLUMNS = [
    "spot_id",
    "sample_id",
    "cohort_id",
    "organ",
    "platform",
    "patch_file",
    "patch_index",
]


@dataclass
class SpotData:
    """Spots pooled across samples. `metadata` and `counts` align row-by-row."""

    metadata: pd.DataFrame
    counts: np.ndarray  # (n_spots, n_genes), raw integer counts
    gene_names: list[str]

    def __post_init__(self) -> None:
        missing = set(METADATA_COLUMNS) - set(self.metadata.columns)
        if missing:
            raise ValueError(f"metadata is missing columns: {sorted(missing)}")
        if len(self.metadata) != len(self.counts):
            raise ValueError(
                f"metadata has {len(self.metadata)} rows but counts has {len(self.counts)}"
            )
        if self.counts.shape[1] != len(self.gene_names):
            raise ValueError("counts columns do not match gene_names")

    def __len__(self) -> int:
        return len(self.metadata)

    def summary(self) -> pd.DataFrame:
        """One row per cohort — what we actually pulled in. Printed before runs."""
        return (
            self.metadata.groupby(["cohort_id", "organ", "platform"], as_index=False)
            .agg(n_spots=("spot_id", "size"), n_samples=("sample_id", "nunique"))
            .sort_values("cohort_id")
            .reset_index(drop=True)
        )


# --------------------------------------------------------------------------
# Expression preprocessing
# --------------------------------------------------------------------------


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Library-size normalise to 10,000 counts per spot, then log1p.

    Two separate problems, one standard fix for each.

    Library size: total counts per spot vary several-fold for reasons unrelated
    to biology — capture efficiency, how many cells happen to sit under the
    spot, sequencing depth. Without rescaling, a deeply sequenced spot and a
    shallow one with identical cell composition look far apart. Dividing by the
    row total and multiplying by a fixed constant (10,000, the scRNA-seq
    convention) puts every spot on the same footing.

    Skew: expression is long-tailed, so a few very high-count genes would
    dominate any distance metric. log1p compresses that. We use log1p rather
    than log because the data is mostly zeros and log(0) is undefined — log1p
    maps zero to zero and leaves the sparsity structure intact.

    This matters for us specifically because InfoNCE operates on cosine
    similarity: unnormalised counts would make the loss largely a function of
    sequencing depth, which is exactly the technical signal we are trying to
    show the model is *not* relying on.
    """
    counts = np.asarray(counts, dtype=np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1.0)  # a spot with zero counts must not divide by zero
    return np.log1p(counts / totals * 1e4).astype(np.float32)


def select_hvg(
    normalized: np.ndarray, train_idx: np.ndarray, n_genes: int = 50
) -> np.ndarray:
    """Pick the `n_genes` most variable genes, measured on training spots only.

    HEST-Benchmark ships a 50-gene panel, but it is chosen *per organ/indication*
    — gene 7 in a breast task is not gene 7 in a kidney task. That makes those
    panels unusable for us, because our whole question is whether alignment
    transfers across organs and cohorts. So we build our own panel from the full
    transcriptome instead.

    Fitting it on `train_idx` only is the point. Highly-variable-gene selection
    is a data-dependent choice; picking genes using the held-out cohort would
    leak information about the test distribution into the model's input space
    and inflate exactly the number we are trying to measure honestly. Callers
    refit this for every split.
    """
    if n_genes > normalized.shape[1]:
        raise ValueError(
            f"asked for {n_genes} genes but only {normalized.shape[1]} are available"
        )
    variances = normalized[train_idx].var(axis=0)
    # argsort is ascending; take the tail and sort the indices so gene order is
    # stable and reproducible rather than variance-ordered.
    top = np.argsort(variances)[-n_genes:]
    return np.sort(top)


def build_expression_features(
    data: SpotData, train_idx: np.ndarray, n_genes: int = 50
) -> tuple[np.ndarray, list[str]]:
    """Raw counts -> the normalised n_genes vector used as the expression embedding.

    v1 has no learned expression encoder: this vector goes straight into the
    projection head. Returns features for *all* spots (the model still needs to
    embed val/test spots) but the gene choice came from train only.
    """
    normalized = normalize_counts(data.counts)
    genes = select_hvg(normalized, train_idx, n_genes=n_genes)
    return normalized[:, genes], [data.gene_names[i] for i in genes]


# --------------------------------------------------------------------------
# Cohort selection
# --------------------------------------------------------------------------


def check_cohort_span(metadata: pd.DataFrame, min_organs: int = 2, min_platforms: int = 2) -> None:
    """Fail loudly if the selected cohorts can't support the splits we plan to run.

    Leave-one-organ-out and leave-one-platform-out are meaningless with a single
    organ or platform, and silently producing an empty test set is the kind of
    bug that survives all the way into a results table.
    """
    organs = sorted(metadata["organ"].unique())
    platforms = sorted(metadata["platform"].unique())
    if len(organs) < min_organs:
        raise ValueError(f"need >= {min_organs} organs, got {organs}")
    if len(platforms) < min_platforms:
        raise ValueError(f"need >= {min_platforms} platforms, got {platforms}")


def select_starter_cohorts(
    sample_meta: pd.DataFrame,
    n_cohorts: int = 4,
    min_spots: int = 200,
) -> pd.DataFrame:
    """Choose the first few HEST cohorts to work with, maximising organ/platform spread.

    Deliberately programmatic rather than a hard-coded list of sample IDs: the
    constraint that matters is coverage (>= 2 organs, >= 2 platforms, Visium
    plus at least one other), not any particular study. Greedy — at each step
    take the cohort that adds the most unseen (organ, platform) coverage, break
    ties on spot count so we prefer better-populated cohorts.

    `sample_meta` is HEST's per-sample metadata table, already renamed to our
    column vocabulary.
    """
    candidates = sample_meta[sample_meta["n_spots"] >= min_spots].copy()
    if candidates.empty:
        raise ValueError(f"no cohorts with >= {min_spots} spots")

    cohorts = (
        candidates.groupby("cohort_id", as_index=False)
        .agg(
            organ=("organ", "first"),
            platform=("platform", "first"),
            n_spots=("n_spots", "sum"),
            n_samples=("sample_id", "nunique"),
        )
    )

    chosen: list[str] = []
    seen_organs: set[str] = set()
    seen_platforms: set[str] = set()

    while len(chosen) < n_cohorts and len(chosen) < len(cohorts):
        remaining = cohorts[~cohorts["cohort_id"].isin(chosen)].copy()
        remaining["gain"] = remaining.apply(
            lambda r: (r["organ"] not in seen_organs) + (r["platform"] not in seen_platforms),
            axis=1,
        )
        pick = remaining.sort_values(["gain", "n_spots"], ascending=[False, False]).iloc[0]
        chosen.append(pick["cohort_id"])
        seen_organs.add(pick["organ"])
        seen_platforms.add(pick["platform"])

    selected = cohorts[cohorts["cohort_id"].isin(chosen)].reset_index(drop=True)
    check_cohort_span(selected)
    if not any("visium" in p.lower() for p in selected["platform"]):
        raise ValueError(f"expected Visium among selected platforms, got {list(selected['platform'])}")
    return selected


# --------------------------------------------------------------------------
# HEST loading
# --------------------------------------------------------------------------


def load_hest_metadata(data_dir: str | Path) -> pd.DataFrame:
    """HEST's per-sample metadata table, renamed to our column vocabulary.

    Imported lazily: `hest` pulls in a heavy stack (openslide, scanpy) that the
    smoke test and the training loop do not need. Only this one step does.
    """
    import datasets  # noqa: F401  (hest depends on it; fail early with a clear error)
    from hest import iter_hest  # noqa: F401

    meta_path = Path(data_dir) / "HEST_v1_1_0.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Download the HEST metadata first — see README."
        )
    raw = pd.read_csv(meta_path)
    return raw.rename(
        columns={
            HEST_SAMPLE_COL: "sample_id",
            HEST_COHORT_COL: "cohort_id",
            HEST_ORGAN_COL: "organ",
            HEST_PLATFORM_COL: "platform",
        }
    )


def load_hest_subset(
    sample_ids: list[str],
    data_dir: str | Path,
    patch_dir: str | Path | None = None,
) -> SpotData:
    """Load the given HEST samples into one pooled SpotData.

    Genes are intersected across samples. Different platforms and different
    submissions do not measure the same gene set, and a gene that is absent in
    one cohort but present in another would be a free cohort label for the model
    to latch onto — the union-plus-zero-fill alternative literally encodes batch
    identity in the input.
    """
    from hest import iter_hest

    data_dir = Path(data_dir)
    patch_dir = Path(patch_dir) if patch_dir else data_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    meta = load_hest_metadata(data_dir).set_index("sample_id")

    per_sample: list[tuple[pd.DataFrame, "pd.DataFrame"]] = []
    for st in iter_hest(str(data_dir), id_list=sample_ids):
        sample_id = st.meta[HEST_SAMPLE_COL]
        adata = st.adata

        patch_file = patch_dir / f"{sample_id}.h5"
        if not patch_file.exists():
            # HEST cuts patches from the whole-slide image on demand. 224 px at
            # 0.5 um/px (20x) is the magnification Phikon-v2 was trained at.
            st.dump_patches(str(patch_dir), name=sample_id, target_patch_size=224,
                            target_pixel_size=0.5)

        row = meta.loc[sample_id]
        spot_meta = pd.DataFrame(
            {
                "spot_id": [f"{sample_id}:{b}" for b in adata.obs_names],
                "sample_id": sample_id,
                "cohort_id": row["cohort_id"],
                "organ": row["organ"],
                "platform": row["platform"],
                "patch_file": str(patch_file),
                "patch_index": np.arange(adata.n_obs),
            }
        )
        # AnnData -> dense float. Fine at our scale (thousands of spots); would
        # need revisiting if we scaled to all 1,229 HEST samples.
        counts = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
        per_sample.append((spot_meta, pd.DataFrame(counts, columns=list(adata.var_names))))

    if not per_sample:
        raise ValueError(f"iter_hest returned nothing for {sample_ids}")

    shared_genes = set(per_sample[0][1].columns)
    for _, counts_df in per_sample[1:]:
        shared_genes &= set(counts_df.columns)
    gene_names = sorted(shared_genes)
    if not gene_names:
        raise ValueError("selected samples share no genes")

    metadata = pd.concat([m for m, _ in per_sample], ignore_index=True)
    counts = np.vstack([c[gene_names].to_numpy(dtype=np.float32) for _, c in per_sample])
    return SpotData(metadata=metadata, counts=counts, gene_names=gene_names)


# --------------------------------------------------------------------------
# Dummy data for the smoke test
# --------------------------------------------------------------------------


def make_dummy_dataset(
    n_spots_per_cohort: int = 10,
    n_genes: int = 200,
    seed: int = 0,
) -> SpotData:
    """Fake SpotData with the same shape and metadata as the real thing.

    Four cohorts spanning three organs and three platforms. More than the bare
    minimum on purpose: every leave-one-out strategy needs at least three groups
    to put train, val and test in different groups, so this shape exercises the
    full protocol rather than the reduced two-group fallback. One organ appears
    in two cohorts, which mirrors the real HEST layout.

    Counts are Poisson with a per-cohort rate offset, so there *is* a synthetic
    batch effect in the data. That is intentional: it means the smoke test
    produces a non-degenerate technical silhouette and we can see the metric
    respond rather than just checking it returns a float.
    """
    rng = np.random.default_rng(seed)
    cohorts = [
        ("cohort_a", "breast", "Visium"),
        ("cohort_b", "kidney", "Visium"),
        ("cohort_c", "lung", "Spatial Transcriptomics"),
        ("cohort_d", "breast", "VisiumHD"),
    ]

    rows, blocks = [], []
    for c_i, (cohort, organ, platform) in enumerate(cohorts):
        base_rate = 2.0 + c_i  # the synthetic batch effect
        blocks.append(rng.poisson(base_rate, size=(n_spots_per_cohort, n_genes)))
        for s in range(n_spots_per_cohort):
            rows.append(
                {
                    "spot_id": f"{cohort}:spot{s}",
                    "sample_id": f"{cohort}_sample0",
                    "cohort_id": cohort,
                    "organ": organ,
                    "platform": platform,
                    "patch_file": f"dummy/{cohort}.h5",
                    "patch_index": s,
                }
            )

    return SpotData(
        metadata=pd.DataFrame(rows),
        counts=np.vstack(blocks).astype(np.float32),
        gene_names=[f"GENE{i}" for i in range(n_genes)],
    )

"""End-to-end run on random data. CPU only, no downloads, a couple of seconds.

Checks plumbing, not science. Image features are random noise unrelated to the
expression vectors, so retrieval should land near chance — a high score here
would mean a leak, not a result.

Run directly for the report, or under pytest for pass/fail.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loading import (  # noqa: E402
    build_expression_features,
    make_dummy_dataset,
    normalize_counts,
)
from embed_images import load_cached_embeddings, save_embeddings  # noqa: E402
from splits import STRATEGIES, build_split  # noqa: E402
from train import TrainConfig, run_strategy  # noqa: E402

SEED = 0
IMAGE_DIM = 1024  # Phikon-v2's CLS width, so head sizes match the real run


def check_data(verbose=True):
    data = make_dummy_dataset(seed=SEED)
    assert data.metadata.spot_id.is_unique

    # Round-tripping the normalisation catches an argument-order slip that
    # would otherwise pass silently.
    totals = np.expm1(normalize_counts(data.counts)).sum(axis=1)
    assert np.allclose(totals, 1e4, rtol=1e-3), totals[:3]

    if verbose:
        print("[1/5] dummy data")
        print(data.summary().to_string(index=False))
        print(f"      normalised spot totals ~ {totals.mean():.0f}")
    return data


def check_gene_panel(data, verbose=True):
    a = build_split(data.metadata, "within_cohort", seed=SEED)
    b = build_split(data.metadata, "cross_organ", seed=SEED)

    _, genes_a = build_expression_features(data, a.train, n_genes=50)
    _, genes_b = build_expression_features(data, b.train, n_genes=50)

    # Identical panels would mean select_hvg is ignoring train_idx.
    overlap = len(set(genes_a) & set(genes_b))
    assert overlap < 50, "gene panel did not change with the training set"

    if verbose:
        print("[2/5] gene panel refit per split")
        print(f"      within_cohort vs cross_organ share {overlap}/50 genes")


def check_cache(data, verbose=True):
    rng = np.random.default_rng(SEED)
    spot_ids = data.metadata.spot_id.tolist()

    with tempfile.TemporaryDirectory() as tmp:
        for sample_id, group in data.metadata.groupby("sample_id"):
            feats = rng.normal(size=(len(group), IMAGE_DIM)).astype(np.float32)
            save_embeddings(sample_id, group.spot_id.tolist(), feats, encoder="dummy", root=tmp)

        # Ask in a shuffled order: the loader keys on spot id, so it must return
        # rows as requested rather than as stored.
        shuffled = list(rng.permutation(spot_ids))
        loaded = load_cached_embeddings(shuffled, encoder="dummy", root=tmp)
        in_order = load_cached_embeddings(spot_ids, encoder="dummy", root=tmp)
        remap = [shuffled.index(s) for s in spot_ids]
        assert np.allclose(in_order, loaded[remap]), "cache loader misaligned rows"

        try:
            load_cached_embeddings(spot_ids + ["nope"], encoder="dummy", root=tmp)
        except KeyError:
            pass
        else:
            raise AssertionError("cache loader should raise on a missing spot")

    if verbose:
        print("[3/5] embedding cache")
        print(f"      {len(data)} x {IMAGE_DIM} vectors, order preserved, missing key raises")
    return in_order


def check_splits(data, verbose=True):
    if verbose:
        print("[4/5] splits")
    for strategy in STRATEGIES:
        s = build_split(data.metadata, strategy, seed=SEED)
        assert not (set(s.train) & set(s.test)), "train/test overlap"
        assert not (set(s.train) & set(s.val)), "train/val overlap"
        if strategy == "cross_organ":
            organs = data.metadata.organ
            assert not (set(organs[s.train]) & set(organs[s.test])), "organ leaked into test"
        if verbose:
            print(f"      {s.name:<32} {s.sizes()}")


def check_train_eval(data, image_features, verbose=True):
    cfg = TrainConfig(epochs=20, batch_size=8, patience=5, seed=SEED, device="cpu")
    if verbose:
        print("[5/5] train + evaluate")

    results = []
    for strategy in STRATEGIES:
        r = run_strategy(data, image_features, strategy, cfg, verbose=False)
        results.append(r)
        for key in ("i2e_recall@1", "e2i_recall@1", "i2e_mrr", "silhouette_organ"):
            assert key in r, key
        assert 0.0 <= r["i2e_recall@1"] <= 1.0
        if verbose:
            print(f"      {strategy:<14} i2e R@1 {r['i2e_recall@1']:.3f} "
                  f"(chance {r['i2e_chance_recall@1']:.3f}) MRR {r['i2e_mrr']:.3f} "
                  f"sil organ {r['silhouette_organ']:+.3f} cohort {r['silhouette_cohort']:+.3f}")
    return results


def run(verbose=True):
    start = time.time()
    data = check_data(verbose)
    check_gene_panel(data, verbose)
    image_features = check_cache(data, verbose)
    check_splits(data, verbose)
    results = check_train_eval(data, image_features, verbose)
    elapsed = time.time() - start

    if verbose:
        print(f"\nOK — {len(results)} strategies in {elapsed:.1f}s on CPU")
        print("Near chance as expected: the image features are random noise.")
    return elapsed


def test_smoke():
    assert run(verbose=False) < 30


if __name__ == "__main__":
    run()

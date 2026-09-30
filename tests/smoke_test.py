"""End-to-end smoke test on random data. CPU only, no downloads, seconds to run.

Purpose is plumbing, not science: every stage runs, shapes line up, nothing
silently drops rows. The numbers it produces are meaningless by construction —
the image features are random noise with no relationship to the expression
vectors, so retrieval should land near chance. A high score here would mean a
leak, not a result.

Run directly (`python tests/smoke_test.py`) for the printed report, or under
pytest for pass/fail.
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
    check_cohort_span,
    make_dummy_dataset,
    normalize_counts,
)
from embed_images import load_cached_embeddings, save_embeddings  # noqa: E402
from evaluate import evaluate_split  # noqa: E402
from splits import STRATEGIES, build_all_splits, leave_one_group_out  # noqa: E402
from train import TrainConfig, run_strategy  # noqa: E402

SEED = 0
IMAGE_DIM = 1024  # Phikon-v2's CLS width, so head sizes match the real run


def check_data(verbose=True):
    data = make_dummy_dataset(seed=SEED)
    check_cohort_span(data.metadata)

    assert len(data) == 40, len(data)
    assert data.counts.shape == (40, 200)
    assert data.metadata["spot_id"].is_unique

    # Library-size normalisation should make every spot sum to ~10,000 before
    # the log. Checking the inverse transform catches an argument-order slip
    # that would otherwise pass silently.
    normalized = normalize_counts(data.counts)
    totals = np.expm1(normalized).sum(axis=1)
    assert np.allclose(totals, 1e4, rtol=1e-3), totals[:3]

    if verbose:
        print("[1/5] dummy data")
        print(data.summary().to_string(index=False))
        print(f"      normalised spot totals ~ {totals.mean():.0f} (target 10,000)")
    return data


def check_gene_panel(data, verbose=True):
    """The panel must be refit per split and must not look at held-out spots."""
    splits = build_all_splits(data.metadata, seed=SEED)
    a = splits["random"][0]
    b = splits["cohort"][0]

    _, genes_a = build_expression_features(data, a.train, n_genes=50)
    _, genes_b = build_expression_features(data, b.train, n_genes=50)
    assert len(genes_a) == len(genes_b) == 50

    # Different training spots should pick a different panel. If these came out
    # identical, the selection is ignoring train_idx.
    overlap = len(set(genes_a) & set(genes_b))
    assert overlap < 50, "gene panel did not change with the training set"

    if verbose:
        print("[2/5] gene panel refit per split")
        print(f"      random vs leave-one-cohort-out panels share {overlap}/50 genes")
    return splits


def check_embedding_cache(data, verbose=True):
    """Write fake embeddings through the real cache path and read them back."""
    rng = np.random.default_rng(SEED)
    spot_ids = data.metadata["spot_id"].tolist()

    with tempfile.TemporaryDirectory() as tmp:
        for sample_id, group in data.metadata.groupby("sample_id"):
            features = rng.normal(size=(len(group), IMAGE_DIM)).astype(np.float32)
            save_embeddings(sample_id, group["spot_id"].tolist(), features,
                            encoder="dummy", root=tmp)

        # Deliberately request a shuffled order: the loader keys on spot id, so
        # it must return rows in the order asked for, not the order on disk.
        shuffled = list(rng.permutation(spot_ids))
        loaded = load_cached_embeddings(shuffled, encoder="dummy", root=tmp)
        assert loaded.shape == (len(data), IMAGE_DIM)

        in_order = load_cached_embeddings(spot_ids, encoder="dummy", root=tmp)
        remap = [shuffled.index(s) for s in spot_ids]
        assert np.allclose(in_order, loaded[remap]), "cache loader misaligned rows"

        try:
            load_cached_embeddings(spot_ids + ["not_a_spot"], encoder="dummy", root=tmp)
        except KeyError:
            pass
        else:
            raise AssertionError("cache loader should raise on a missing spot")

    if verbose:
        print("[3/5] embedding cache")
        print(f"      wrote/read {len(data)} x {IMAGE_DIM} vectors, order preserved, "
              "missing key raises")
    return in_order


def check_splits(data, splits, verbose=True):
    if verbose:
        print("[4/5] splits")
    for strategy in STRATEGIES:
        for split in splits[strategy]:
            sizes = split.sizes()
            assert sum(sizes.values()) == len(data)
            if strategy != "random":
                column = {"cohort": "cohort_id", "organ": "organ", "platform": "platform"}[strategy]
                test_groups = set(data.metadata.iloc[split.test][column])
                train_groups = set(data.metadata.iloc[split.train][column])
                assert not (test_groups & train_groups), "held-out group leaked into train"
            if verbose:
                print(f"      {split.name:<28} {sizes}")

    # The two-group fallback: exercise it explicitly, since the dummy set has
    # three organs and so never hits that branch above.
    two_organs = data.metadata[data.metadata["organ"].isin(["breast", "kidney"])].reset_index(drop=True)
    folds = leave_one_group_out(two_organs, "organ", seed=SEED)
    assert all(f.in_distribution_val for f in folds)
    if verbose:
        print(f"      two-group fallback: {len(folds)} folds, in-distribution val flagged")


def check_train_eval(data, image_features, verbose=True):
    cfg = TrainConfig(epochs=20, batch_size=16, patience=5, seed=SEED, device="cpu")
    if verbose:
        print("[5/5] train + evaluate")

    all_results = []
    for strategy in STRATEGIES:
        results = run_strategy(data, image_features, strategy, cfg, verbose=False)
        all_results.extend(results)
        for r in results:
            for key in ("i2e_recall@1", "e2i_recall@1", "i2e_mrr", "silhouette_organ"):
                assert key in r, key
            assert 0.0 <= r["i2e_recall@1"] <= 1.0
            assert r["epochs_run"] >= 1
        if verbose:
            r = results[0]
            print(f"      {strategy:<9} folds={len(results):<2} "
                  f"i2e R@1 {r['i2e_recall@1']:.3f} (chance {r['i2e_chance_recall@1']:.3f}) "
                  f"MRR {r['i2e_mrr']:.3f} "
                  f"sil organ {r['silhouette_organ']:+.3f} cohort {r['silhouette_cohort']:+.3f}")
    return all_results


def run(verbose=True):
    start = time.time()
    data = check_data(verbose)
    splits = check_gene_panel(data, verbose)
    image_features = check_embedding_cache(data, verbose)
    check_splits(data, splits, verbose)
    results = check_train_eval(data, image_features, verbose)
    elapsed = time.time() - start

    if verbose:
        print(f"\nOK — {len(results)} folds across {len(STRATEGIES)} strategies "
              f"in {elapsed:.1f}s on CPU")
        print("Scores are near chance as expected: the image features are random noise.")
    return elapsed


def test_smoke():
    """Pytest entry point. The time budget is part of the contract — this has to
    stay fast enough to run on every change."""
    assert run(verbose=False) < 30


if __name__ == "__main__":
    run()

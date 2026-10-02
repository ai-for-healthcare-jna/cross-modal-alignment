"""End-to-end check on synthetic data. CPU only, no downloads, a couple of seconds."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import fit_predict, pearson_per_gene  # noqa: E402
from embed_images import load_cached_embeddings, save_embeddings  # noqa: E402

SEED = 0
IMAGE_DIM = 1024
N_GENES = 50


def check_cache(verbose=True):
    rng = np.random.default_rng(SEED)
    spot_ids = [f"S1:bc{i}" for i in range(40)]

    with tempfile.TemporaryDirectory() as tmp:
        feats = rng.normal(size=(40, IMAGE_DIM)).astype(np.float32)
        save_embeddings("S1", spot_ids, feats, encoder="dummy", root=tmp)

        # Ask in a shuffled order: the loader keys on spot id, so it must
        # return rows as requested rather than as stored.
        shuffled = list(rng.permutation(spot_ids))
        loaded = load_cached_embeddings(shuffled, encoder="dummy", root=tmp)
        remap = [shuffled.index(s) for s in spot_ids]
        assert np.allclose(loaded[remap], feats), "cache loader misaligned rows"

        try:
            load_cached_embeddings(spot_ids + ["nope"], encoder="dummy", root=tmp)
        except KeyError:
            pass
        else:
            raise AssertionError("cache loader should raise on a missing spot")

    if verbose:
        print(f"[1/2] cache: {len(spot_ids)} x {IMAGE_DIM}, order preserved, missing key raises")


def check_regression(verbose=True):
    rng = np.random.default_rng(SEED)
    n_train, n_test = 600, 200

    # Expression built as a linear function of the features plus noise, so a
    # working PCA+ridge pipeline has to score well above zero.
    W = rng.normal(size=(IMAGE_DIM, N_GENES))
    X_train = rng.normal(size=(n_train, IMAGE_DIM))
    X_test = rng.normal(size=(n_test, IMAGE_DIM))
    with np.errstate(all="ignore"):  # Accelerate BLAS false positive
        y_train = X_train @ W + rng.normal(scale=2.0, size=(n_train, N_GENES))
        y_test = X_test @ W + rng.normal(scale=2.0, size=(n_test, N_GENES))

    preds, alpha = fit_predict(X_train, y_train, X_test)
    assert preds.shape == y_test.shape
    r = pearson_per_gene(y_test, preds)
    assert len(r) == N_GENES and np.isfinite(r).all()
    assert r.mean() > 0.1, r.mean()

    if verbose:
        print(f"[2/2] PCA+ridge: alpha {alpha:.6f}, pearson {r.mean():.3f} on synthetic signal")


def run(verbose=True):
    start = time.time()
    check_cache(verbose)
    check_regression(verbose)
    elapsed = time.time() - start
    if verbose:
        print(f"\nOK in {elapsed:.1f}s on CPU")
    return elapsed


def test_smoke():
    assert run(verbose=False) < 30


if __name__ == "__main__":
    run()

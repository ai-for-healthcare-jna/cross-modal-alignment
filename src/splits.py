"""Train/val/test indices for the within-cohort and cross-organ splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRATEGIES = ("within_cohort", "cross_organ")


@dataclass
class Split:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self):
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def within_cohort(metadata, organ="Breast", val_frac=0.1, test_frac=0.2, seed=0):
    # No grouping by slide, so adjacent spots land on both sides. That's the
    # point - this is the optimistic number.
    idx = np.where(metadata.organ == organ)[0]
    idx = np.random.default_rng(seed).permutation(idx)

    n_test = int(len(idx) * test_frac)
    n_val = int(len(idx) * val_frac)
    return Split(
        name=f"within_cohort:{organ}",
        test=np.sort(idx[:n_test]),
        val=np.sort(idx[n_test : n_test + n_val]),
        train=np.sort(idx[n_test + n_val :]),
    )


def cross_organ(metadata, train_organ="Breast", test_organ="Prostate", val_frac=0.1, seed=0):
    # Val comes out of the training organ: there's no third organ to spare and
    # validating on prostate would be selecting on the test set.
    train_idx = np.random.default_rng(seed).permutation(
        np.where(metadata.organ == train_organ)[0]
    )
    n_val = int(len(train_idx) * val_frac)
    return Split(
        name=f"cross_organ:{train_organ}->{test_organ}",
        val=np.sort(train_idx[:n_val]),
        train=np.sort(train_idx[n_val:]),
        test=np.where(metadata.organ == test_organ)[0],
    )


def build_split(metadata, strategy, seed=0):
    if strategy == "within_cohort":
        return within_cohort(metadata, seed=seed)
    if strategy == "cross_organ":
        return cross_organ(metadata, seed=seed)
    raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

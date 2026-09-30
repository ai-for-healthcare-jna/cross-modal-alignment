"""Train/val/test index generation under four increasingly hostile protocols.

The four strategies are the experiment, not a utility. Each one holds out a
different kind of structure:

  random      spots shuffled freely. Neighbouring spots from the same slide end
              up on both sides, so the model can win by memorising the slide.
              This is the optimistic baseline, not a result.
  cohort      a whole study is held out. Tests transfer across lab, protocol
              and patient population at once.
  organ       a whole tissue type is held out. Tests whether the alignment
              encodes anything about tissue biology in general.
  platform    a whole sequencing technology is held out. The most purely
              technical of the three, and the cleanest test of batch fitting.

The gap between `random` and the rest is the headline number. If random scores
well and leave-one-cohort-out collapses, the model was fitting batch structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRATEGIES = ("random", "cohort", "organ", "platform")

# Which metadata column defines the held-out group for each leave-one-out strategy.
_GROUP_COLUMN = {"cohort": "cohort_id", "organ": "organ", "platform": "platform"}


@dataclass
class Split:
    """One train/val/test partition, as positional indices into SpotData."""

    name: str
    strategy: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    held_out: str | None = None  # which group is in test, for leave-one-out
    # True when val had to be drawn from the training distribution rather than
    # a held-out group. Tracked so it can be reported alongside the metrics.
    in_distribution_val: bool = False

    def __post_init__(self) -> None:
        for part in ("train", "val", "test"):
            if len(getattr(self, part)) == 0:
                raise ValueError(f"split '{self.name}' has an empty {part} set")
        train, val, test = set(self.train), set(self.val), set(self.test)
        if (train & val) or (train & test) or (val & test):
            raise ValueError(f"split '{self.name}' has overlapping partitions")

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def random_split(
    metadata: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> list[Split]:
    """Shuffle spots and cut. The optimistic baseline.

    No grouping at all — this is intentionally leaky with respect to slide and
    cohort, because its job is to show how good the numbers look when you don't
    control for batch.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(metadata))
    n_test = max(1, int(round(len(idx) * test_frac)))
    n_val = max(1, int(round(len(idx) * val_frac)))
    return [
        Split(
            name="random",
            strategy="random",
            test=np.sort(idx[:n_test]),
            val=np.sort(idx[n_test : n_test + n_val]),
            train=np.sort(idx[n_test + n_val :]),
        )
    ]


def leave_one_group_out(
    metadata: pd.DataFrame, strategy: str, val_frac: float = 0.15, seed: int = 0
) -> list[Split]:
    """One Split per group: that group is test, and the rest is train/val.

    With three or more groups, val is a *different held-out group*. That is the
    protocol we want: selecting a checkpoint on an in-distribution val set picks
    whichever model best overfits the training cohorts, quietly reintroducing
    the leak this split exists to avoid.

    With exactly two groups that is impossible — holding one out for test leaves
    nothing to validate on but the training group. We fall back to a random spot
    subset of train and flag it on the Split, because the alternative (refusing
    to run) would make leave-one-organ-out unavailable for a two-organ starter
    set. The fallback weakens model selection, not the test measurement itself,
    so the headline number stays meaningful — but it should be reported.
    """
    column = _GROUP_COLUMN[strategy]
    groups = sorted(metadata[column].astype(str).unique())
    if len(groups) < 2:
        raise ValueError(
            f"leave-one-{strategy}-out needs >= 2 distinct '{column}' values, got {groups}"
        )

    values = metadata[column].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    splits = []

    for i, held in enumerate(groups):
        others = [g for g in groups if g != held]
        test_idx = np.where(values == held)[0]
        rest_idx = np.where(values != held)[0]

        if len(groups) >= 3:
            # Rotate the val group across folds so one unusual cohort doesn't
            # drive model selection everywhere.
            val_group = others[i % len(others)]
            val_idx = np.where(values == val_group)[0]
            train_idx = np.setdiff1d(rest_idx, val_idx)
            in_distribution_val = False
        else:
            shuffled = rng.permutation(rest_idx)
            n_val = max(1, int(round(len(shuffled) * val_frac)))
            val_idx = np.sort(shuffled[:n_val])
            train_idx = np.sort(shuffled[n_val:])
            in_distribution_val = True

        splits.append(
            Split(
                name=f"{strategy}:{held}",
                strategy=strategy,
                train=train_idx,
                val=val_idx,
                test=test_idx,
                held_out=held,
                in_distribution_val=in_distribution_val,
            )
        )
    return splits


def build_splits(metadata: pd.DataFrame, strategy: str, seed: int = 0, **kwargs) -> list[Split]:
    """Entry point. Returns a list because leave-one-out yields one fold per group."""
    if strategy == "random":
        return random_split(metadata, seed=seed, **kwargs)
    if strategy in _GROUP_COLUMN:
        return leave_one_group_out(metadata, strategy, seed=seed, **kwargs)
    raise ValueError(f"unknown strategy '{strategy}', expected one of {STRATEGIES}")


def build_all_splits(metadata: pd.DataFrame, seed: int = 0) -> dict[str, list[Split]]:
    """All four strategies from the same metadata, so results are comparable."""
    return {s: build_splits(metadata, s, seed=seed) for s in STRATEGIES}

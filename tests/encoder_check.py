"""Verify the real image encoder and the embedding code path, on CPU.

Separate from smoke_test.py because this downloads ~1.1 GB of weights and takes
a minute rather than a second. The smoke test has to stay cheap enough to run on
every change; this is the check you run once after touching embed_images.py, and
before paying for a GPU.

It exercises the actual `embed_patch_file` -> `save_embeddings` ->
`load_cached_embeddings` path against a synthetic .h5 laid out the way HEST
writes them ('img' uint8 (N,224,224,3) + 'barcode'). So everything the GPU run
will do is covered here except the contents of a real HEST file.

Run:  python tests/encoder_check.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embed_images import (  # noqa: E402
    DEFAULT_ENCODER,
    ENCODERS,
    embed_patch_file,
    load_cached_embeddings,
    load_encoder,
    save_embeddings,
)

N_PATCHES = 5
BATCH_SIZE = 2  # smaller than N_PATCHES so the chunking loop actually runs twice


def write_fake_patch_file(path: Path, n: int = N_PATCHES) -> None:
    """A .h5 shaped like HEST's patch dumps.

    Structured noise rather than uniform noise: each patch gets a different mean
    intensity, so if the encoder collapsed every input to the same vector we
    would see it in the pairwise distances below.
    """
    import h5py

    rng = np.random.default_rng(0)
    images = np.stack([
        np.clip(rng.normal(40 + 40 * i, 20, size=(224, 224, 3)), 0, 255).astype(np.uint8)
        for i in range(n)
    ])
    with h5py.File(path, "w") as f:
        f.create_dataset("img", data=images)
        f.create_dataset("barcode", data=np.array([f"BC{i}".encode() for i in range(n)]))


def main() -> None:
    encoder = DEFAULT_ENCODER
    expected_dim = ENCODERS[encoder]["embed_dim"]
    print(f"encoder={encoder} expected_dim={expected_dim} device=cpu")

    t0 = time.time()
    model, processor = load_encoder(encoder, "cpu")
    print(f"[1/5] loaded in {time.time() - t0:.1f}s  ({type(model).__name__}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    # The frozen-encoder assumption the whole repo rests on.
    assert not model.training, "model should be in eval mode"
    assert not any(p.requires_grad for p in model.parameters()), "params should be frozen"
    print("[2/5] frozen: eval mode, no params require grad")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        patch_file = tmp / "FAKE01.h5"
        write_fake_patch_file(patch_file)

        t0 = time.time()
        barcodes, features = embed_patch_file(
            patch_file, model, processor, "cpu", batch_size=BATCH_SIZE
        )
        elapsed = time.time() - t0

        assert features.shape == (N_PATCHES, expected_dim), features.shape
        assert np.isfinite(features).all(), "non-finite features"
        # Exact match, not just the right count. A loose check here is how the
        # bytes-repr bug ("b'BC0'") got through the first time.
        assert barcodes == [f"BC{i}" for i in range(N_PATCHES)], barcodes
        print(f"[3/5] forward pass: {features.shape} in {elapsed:.1f}s "
              f"({elapsed / N_PATCHES:.2f}s/patch on CPU)")
        print(f"      barcodes: {barcodes}")
        print(f"      feature range [{features.min():.2f}, {features.max():.2f}], "
              f"norm {np.linalg.norm(features, axis=1).mean():.1f}")

        # Distinct inputs must give distinct outputs. A collapsed encoder would
        # make every retrieval metric downstream meaningless while looking fine.
        normed = features / np.linalg.norm(features, axis=1, keepdims=True)
        sim = normed @ normed.T
        off_diagonal = sim[~np.eye(N_PATCHES, dtype=bool)]
        assert off_diagonal.max() < 0.999, f"encoder collapsed: max sim {off_diagonal.max()}"
        print(f"[4/5] not collapsed: off-diagonal cosine sim "
              f"[{off_diagonal.min():.3f}, {off_diagonal.max():.3f}]")

        # Round-trip through the cache, which is what train.py consumes.
        spot_ids = [f"FAKE01:{b}" for b in barcodes]
        save_embeddings("FAKE01", spot_ids, features, encoder=encoder, root=tmp)
        reloaded = load_cached_embeddings(spot_ids, encoder=encoder, root=tmp)
        assert reloaded.shape == features.shape
        assert np.allclose(reloaded, features, atol=1e-6), "cache round-trip changed values"
        print("[5/5] cache round-trip exact")

        # Determinism: same input, same output. Cheap guard against a stray
        # dropout or non-deterministic kernel creeping in.
        _, again = embed_patch_file(patch_file, model, processor, "cpu", batch_size=N_PATCHES)
        assert np.allclose(again, features, atol=1e-5), "encoder is not deterministic"
        print("      deterministic across runs and batch sizes")

    print(f"\nOK — {encoder} verified on CPU, {expected_dim}-d CLS embeddings")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

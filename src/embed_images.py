"""Run the frozen image encoder over every patch once and cache the result.

Why this is a separate script rather than part of the training loop:

The image encoder is frozen — it is never updated, so a given patch produces
the same vector on every epoch of every experiment. Re-running a 300M-parameter
ViT over the same patches for each of four split strategies would burn GPU time
to recompute a constant. Instead we pay for one pass, write the vectors to
disk, and every downstream script reads from that cache. The practical effect
is that this is the only step needing a GPU: after it finishes, training and
evaluation run on a laptop CPU in seconds, which is what makes it feasible to
iterate on the actual research question.

The cache is keyed by (encoder, sample_id) so switching encoders creates a new
cache directory instead of silently reusing vectors from a different model.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

# Encoder registry. Adding an entry here is all that is needed to compare
# against another encoder, which is worth having: "the result held with a second
# encoder" is a cheap and meaningful robustness check once the cache exists.
#
# Phikon-v2 is the default, chosen for three reasons:
#
#   Open access. It needs no gating request and no institutional email, so
#   anyone reading this repo can reproduce the numbers. The stronger
#   alternatives in this space (UNI, H-optimus) are both gated behind
#   institutional verification, which makes them unusable here and would make
#   the results unreproducible for most readers.
#
#   Magnification match. Trained on 20x histology tiles, which is the
#   magnification HEST patches are cut at (224 px at 0.5 um/px). An encoder
#   trained at a different magnification would see our patches out of
#   distribution, and any failure to align would be confounded with that.
#
#   Known quantity. Phikon-v2 appears in the HEST-Benchmark comparison table,
#   so its behaviour on this exact dataset family is already documented. That
#   gives a reference point for whether our frozen features are reasonable
#   before we start interpreting alignment results.
#
# It is a ViT-L/16 (DINOv2 architecture) pretrained on ~450M tiles from public
# cohorts including TCGA, CPTAC and GTEx. Non-commercial licence, which is fine
# for research use.
ENCODERS = {
    "phikon-v2": {"hf_id": "owkin/phikon-v2", "embed_dim": 1024},
    # ViT-B predecessor. Half the width, so it doubles as a quick check that
    # results are not an artefact of one particular feature dimensionality.
    "phikon": {"hf_id": "owkin/phikon", "embed_dim": 768},
}

DEFAULT_ENCODER = "phikon-v2"


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cache_dir_for(encoder: str, root: str | Path = "embeddings") -> Path:
    return Path(root) / encoder


def save_embeddings(
    sample_id: str, spot_ids: list[str], features: np.ndarray,
    encoder: str = DEFAULT_ENCODER, root: str | Path = "embeddings",
) -> Path:
    """One .npz per sample, holding features and the spot ids they belong to.

    Storing the ids alongside the array means a later reordering of the
    metadata can't silently misalign patches with expression vectors — the
    loader matches by id rather than trusting row order.
    """
    out_dir = cache_dir_for(encoder, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sample_id}.npz"
    np.savez_compressed(path, features=features.astype(np.float32),
                        spot_ids=np.array(spot_ids, dtype=object))
    return path


def load_cached_embeddings(
    spot_ids, encoder: str = DEFAULT_ENCODER, root: str | Path = "embeddings"
) -> np.ndarray:
    """Load features for the given spot ids, in that exact order.

    Raises on a cache miss rather than returning a partial array — a silently
    dropped cohort would be very hard to notice in a results table.
    """
    cache = cache_dir_for(encoder, root)
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cached embeddings in {cache}; run embed_images.py first")

    lookup: dict[str, np.ndarray] = {}
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            for sid, vec in zip(z["spot_ids"], z["features"]):
                lookup[str(sid)] = vec

    missing = [s for s in spot_ids if s not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} spots missing from the '{encoder}' cache "
            f"(first few: {missing[:3]})"
        )
    return np.stack([lookup[s] for s in spot_ids]).astype(np.float32)


def load_encoder(encoder: str, device: str):
    """Load the frozen encoder and its matching image processor.

    The processor is loaded from the same checkpoint rather than hand-rolled:
    it carries the resize and per-channel normalisation the model was trained
    with, and getting those wrong silently degrades features rather than
    raising.

    eval() and requires_grad_(False) are both set here rather than left to the
    caller. The encoder being frozen is an assumption baked into the whole
    design of this repo, so it is enforced at the point of construction.
    """
    from transformers import AutoImageProcessor, AutoModel

    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder '{encoder}', expected one of {list(ENCODERS)}")
    hf_id = ENCODERS[encoder]["hf_id"]

    processor = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    model.eval().requires_grad_(False).to(device)

    expected = ENCODERS[encoder]["embed_dim"]
    if model.config.hidden_size != expected:
        raise ValueError(
            f"{hf_id} reports hidden_size={model.config.hidden_size}, "
            f"registry says {expected} — update ENCODERS"
        )
    return model, processor


def _decode_barcode(raw) -> str:
    """h5py gives back bytes, not str.

    Worth a named helper because the naive `str(raw)` yields "b'AAACCT'" — the
    repr, complete with the b and the quotes. That still looks like a plausible
    id, so it survives every shape check and only fails much later when these
    keys are matched against the plain-string barcodes in HEST's AnnData. The
    symptom would be a total cache miss on the real run, after the GPU bill.
    """
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.bytes_):
        return raw.tobytes().decode("utf-8")
    return str(raw)


@torch.inference_mode()
def embed_patch_file(
    patch_file: Path, model, processor, device: str, batch_size: int = 64
) -> tuple[list[str], np.ndarray]:
    """Embed every patch in one HEST patch .h5.

    HEST writes patches as an 'img' dataset of uint8 (N, 224, 224, 3) with a
    matching 'barcode' dataset. We read in chunks rather than loading the whole
    file: a large Visium sample is several GB uncompressed.

    We take the CLS token rather than mean-pooling the patch tokens. For a
    DINOv2-style model the CLS token is the representation the pretraining
    objective actually shaped, and it is what Phikon-v2's own model card uses —
    so it is the comparable choice. Mean pooling is a defensible alternative and
    sometimes helps on dense tasks, but mixing conventions would make our
    numbers incomparable to published ones for no benefit.
    """
    import h5py
    from PIL import Image

    features, barcodes = [], []
    # fp16 on CUDA only. MPS autocast is unreliable and CPU fp16 is slower than
    # fp32, so neither is worth the risk on a step we run once.
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    with h5py.File(patch_file, "r") as f:
        images, codes = f["img"], f["barcode"]
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            inputs = processor(
                images=[Image.fromarray(img) for img in chunk], return_tensors="pt"
            ).to(device)

            with amp:
                outputs = model(**inputs)

            cls = outputs.last_hidden_state[:, 0, :]
            features.append(cls.float().cpu().numpy())
            barcodes.extend(
                _decode_barcode(c) for c in np.asarray(codes[start : start + batch_size]).ravel()
            )

    return barcodes, np.concatenate(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen image embeddings for HEST patches.")
    parser.add_argument("--samples", nargs="+", required=True, help="HEST sample ids")
    parser.add_argument("--data-dir", default="data", help="HEST download cache")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(ENCODERS))
    parser.add_argument("--out", default="embeddings")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"encoder={args.encoder} device={device}")

    model, processor = load_encoder(args.encoder, device)

    patch_dir = Path(args.data_dir) / "patches"
    for sample_id in args.samples:
        out_path = cache_dir_for(args.encoder, args.out) / f"{sample_id}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"  {sample_id}: cached, skipping")
            continue

        patch_file = patch_dir / f"{sample_id}.h5"
        if not patch_file.exists():
            raise FileNotFoundError(f"{patch_file} not found; run data_loading first")

        barcodes, features = embed_patch_file(
            patch_file, model, processor, device, batch_size=args.batch_size
        )
        # Prefix with the sample id so spot ids are globally unique once cohorts
        # are pooled — barcodes are only unique within a sample.
        spot_ids = [f"{sample_id}:{b}" for b in barcodes]
        save_embeddings(sample_id, spot_ids, features, encoder=args.encoder, root=args.out)
        print(f"  {sample_id}: {features.shape[0]} spots -> {features.shape[1]}-d")

    manifest = cache_dir_for(args.encoder, args.out) / "manifest.json"
    manifest.write_text(json.dumps(
        {"encoder": args.encoder, "embed_dim": ENCODERS[args.encoder]["embed_dim"],
         "samples": args.samples}, indent=2))
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()

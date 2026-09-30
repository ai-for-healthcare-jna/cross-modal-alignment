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
import json
from pathlib import Path

import numpy as np
import torch

# Encoder registry. UNI is the default: it comes from the same lab as HEST-1k,
# is trained at the 0.5 um/px resolution HEST patches are cut at, and is
# publicly downloadable, which matters for reproducibility. Adding an entry here
# is all that is needed to compare against another encoder.
ENCODERS = {
    "uni": {
        "hf_hub": "hf-hub:MahmoodLab/UNI",
        "embed_dim": 1024,
        "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": True},
    },
    "uni2": {
        "hf_hub": "hf-hub:MahmoodLab/UNI2-h",
        "embed_dim": 1536,
        "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": True},
    },
}

DEFAULT_ENCODER = "uni"


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
    """Load the frozen encoder and its matching preprocessing transform.

    eval() and requires_grad_(False) are both set here rather than left to the
    caller. The encoder being frozen is an assumption baked into the whole
    design of this repo, so it is enforced at the point of construction.
    """
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder '{encoder}', expected one of {list(ENCODERS)}")
    spec = ENCODERS[encoder]

    model = timm.create_model(spec["hf_hub"], pretrained=True, **spec["timm_kwargs"])
    model.eval().requires_grad_(False).to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


@torch.no_grad()
def embed_patch_file(
    patch_file: Path, model, transform, device: str, batch_size: int = 64
) -> tuple[list[str], np.ndarray]:
    """Embed every patch in one HEST patch .h5.

    HEST writes patches as an 'img' dataset of uint8 (N, 224, 224, 3) with a
    matching 'barcode' dataset. We read in chunks rather than loading the whole
    file: a large Visium sample is several GB uncompressed.
    """
    import h5py
    from PIL import Image

    features, barcodes = [], []
    autocast = torch.autocast(device_type=device, dtype=torch.float16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)

    with h5py.File(patch_file, "r") as f:
        images, codes = f["img"], f["barcode"]
        n = len(images)
        for start in range(0, n, batch_size):
            chunk = images[start : start + batch_size]
            batch = torch.stack([transform(Image.fromarray(img)) for img in chunk]).to(device)
            with autocast:
                out = model(batch)
            features.append(out.float().cpu().numpy())
            barcodes.extend(str(c) for c in np.asarray(codes[start : start + batch_size]).ravel())

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

    model, transform = load_encoder(args.encoder, device)

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
            patch_file, model, transform, device, batch_size=args.batch_size
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

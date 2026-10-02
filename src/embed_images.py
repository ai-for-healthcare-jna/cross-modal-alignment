"""Cache frozen image embeddings for HEST patches."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

# Phikon-v2 over UNI/H-optimus because those are gated on institutional email.
# Keyed registry so swapping encoders writes a separate cache dir rather than
# silently mixing vectors from two models.
ENCODERS = {
    "phikon-v2": {"hf_id": "owkin/phikon-v2", "embed_dim": 1024},
    "phikon": {"hf_id": "owkin/phikon", "embed_dim": 768},  # ViT-B, 768-d
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
    # Store ids next to the array so the loader can match on them instead of
    # trusting row order.
    out_dir = cache_dir_for(encoder, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sample_id}.npz"
    np.savez_compressed(path, features=features.astype(np.float32),
                        spot_ids=np.array(spot_ids, dtype=object))
    return path


def load_cached_embeddings(
    spot_ids, encoder: str = DEFAULT_ENCODER, root: str | Path = "embeddings"
) -> np.ndarray:
    """Features for the given spot ids, in that order."""
    cache = cache_dir_for(encoder, root)
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cached embeddings in {cache}; run embed_images.py first")

    lookup: dict[str, np.ndarray] = {}
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            for sid, vec in zip(z["spot_ids"], z["features"]):
                lookup[str(sid)] = vec

    # Raise rather than return a partial array - a quietly dropped cohort is
    # very hard to spot in a results table.
    missing = [s for s in spot_ids if s not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} spots missing from the '{encoder}' cache "
            f"(first few: {missing[:3]})"
        )
    return np.stack([lookup[s] for s in spot_ids]).astype(np.float32)


def load_encoder(encoder: str, device: str):
    """Frozen encoder plus its matching image processor."""
    from transformers import AutoImageProcessor, AutoModel

    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder '{encoder}', expected one of {list(ENCODERS)}")
    hf_id = ENCODERS[encoder]["hf_id"]

    # Processor from the same checkpoint - it carries the resize and channel
    # normalisation the model was trained with, and getting those wrong
    # degrades features silently.
    processor = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    # Freeze here, not at the call site. Everything downstream assumes it.
    model.eval().requires_grad_(False).to(device)

    expected = ENCODERS[encoder]["embed_dim"]
    if model.config.hidden_size != expected:
        raise ValueError(
            f"{hf_id} reports hidden_size={model.config.hidden_size}, "
            f"registry says {expected} — update ENCODERS"
        )
    return model, processor


def _decode_barcode(raw) -> str:
    # h5py returns bytes and str() on those gives the repr - "b\'AAACCT\'", not
    # "AAACCT". Looks like a valid id, passes every shape check, then misses the
    # cache completely when matched against the .h5ad barcodes.
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.bytes_):
        return raw.tobytes().decode("utf-8")
    return str(raw)


@torch.inference_mode()
def embed_patch_file(
    patch_file: Path, model, processor, device: str, batch_size: int = 64
) -> tuple[list[str], np.ndarray]:
    """Embed every patch in one HEST patch .h5."""
    import h5py
    from PIL import Image

    features, barcodes = [], []
    # fp16 on CUDA only - MPS autocast is unreliable and CPU fp16 is slower
    # than fp32.
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    # Chunked read: 'img' is uint8 (N, 224, 224, 3) and a big Visium sample is
    # several GB uncompressed.
    with h5py.File(patch_file, "r") as f:
        images, codes = f["img"], f["barcode"]
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            inputs = processor(
                images=[Image.fromarray(img) for img in chunk], return_tensors="pt"
            ).to(device)

            with amp:
                outputs = model(**inputs)

            # CLS token, matching Phikon-v2's model card. Mean pooling is
            # defensible but would break comparability with published numbers.
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
        # Prefix with sample id - barcodes are only unique within a sample.
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

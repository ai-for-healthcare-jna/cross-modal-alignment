# cross-modal-alignment

**Cross-organ generalization of image–expression alignment**

> **Status: pilot in progress.** The pipeline runs end to end on synthetic data.
> Real embeddings have not been computed yet and there are no results. This
> README gets rewritten once there are.

---

## Background

Spatial transcriptomics gives paired data: a histology image (a photo of tissue
under a microscope) and a gene expression profile (which genes are active at that
location), both measured from the same physical spot on a tissue slice. Foundation
models exist for each modality separately. This project asks whether their internal
representations can be aligned across modalities, and whether that alignment
survives when tested on tissue the model was never trained on.

## Research question

Train a contrastive alignment between frozen histopathology-image embeddings and
gene-expression embeddings on breast cancer tissue. Does it transfer to prostate
cancer tissue, or did it only learn something specific to the cohort it saw?

Both cohorts are Visium, so platform, assay chemistry and spot geometry are held
fixed. Organ is the only thing that changes.

### Three numbers

| | What it measures |
|---|---|
| Chance | 1/N for a gallery of N candidates. Reported alongside every score, because Recall@1 is meaningless without it. |
| Within-cohort | Breast spots split 80/20. Neighbouring spots from the same slide land on both sides, so this is the optimistic baseline. |
| Cross-organ | Train breast, test prostate. |

The gap between within-cohort and cross-organ is the result.

## Approach

Both encoders are frozen. The only trainable parameters are two small MLP heads
projecting into a shared 128-d space, trained with InfoNCE.

- **Image side:** [Phikon-v2](https://huggingface.co/owkin/phikon-v2) (ViT-L/16,
  DINOv2, 1024-d CLS token). Chosen over UNI and H-optimus because those are gated
  behind institutional-email verification, which would make these results
  irreproducible for most readers. Trained on ~450M 20x histology tiles from public
  cohorts, matching the magnification HEST patches are cut at.
- **Expression side:** the 50 highest-variance genes, library-size normalised to
  10,000 counts per spot and log1p transformed. No learned expression encoder — the
  normalised vector goes straight into its projection head.
- **Data:** [HEST-1k](https://github.com/mahmoodlab/hest)
  ([paper](https://arxiv.org/abs/2406.16192)). Two Visium cohorts:

| Cohort | Organ | Spots |
|---|---|---|
| High resolution mapping of the tumor microenvironment… (`NCBI776`) | Breast | 4,973 |
| Human Prostate Cancer, Adenocarcinoma with Invasive Carcinoma (`TENX40`) | Prostate | 4,371 |

### Note on the gene panel

HEST-Benchmark ships a 50-gene panel, but it is selected *per organ* — gene 7 in a
breast task is not gene 7 in a prostate task, which makes those panels useless for
a cross-organ question. This project builds its own panel instead: intersect the
genes present in both cohorts (17,368 of them), then take the 50 most variable,
**fit on training spots only and refit for each split**. Selecting genes with the
test organ visible would leak the test distribution into the input space.

## Layout

```
src/
  data_loading.py   HEST loading, normalisation, gene panel selection
  embed_images.py   frozen encoder -> cached embeddings (the only GPU-ish step)
  model.py          two projection heads + InfoNCE
  train.py          training loop over cached embeddings
  evaluate.py       retrieval metrics + silhouette
  splits.py         within-cohort and cross-organ splits
tests/
  smoke_test.py     end-to-end on synthetic data, ~1s
  encoder_check.py  loads the real encoder, needs a 1.1 GB download
```

`data/`, `embeddings/` and `results/` are gitignored.

### Why embeddings are cached

The image encoder is frozen, so a patch produces the same vector forever. Running
the ViT again for each split would be recomputing a constant. `embed_images.py`
does one pass and writes `embeddings/<encoder>/<sample_id>.npz`; everything
downstream reads from there, which keeps training and evaluation on CPU at a few
seconds per run.

Reading HEST needs only `h5py` — `.h5ad` is plain HDF5, so there is no `anndata`,
`scanpy` or `openslide` dependency, and the pre-extracted patches mean the
multi-GB whole-slide images are never downloaded.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

HEST-1k is a gated dataset: accept the terms at
https://huggingface.co/datasets/MahmoodLab/hest (automatic) and run `hf auth login`.

## Usage

```bash
# Check the pipeline on synthetic data. No downloads.
python tests/smoke_test.py

# Cache image embeddings (uses MPS on Apple Silicon, CUDA if present).
python src/embed_images.py --samples NCBI776 TENX40 --data-dir data/hest

# Train and evaluate both splits.
python src/train.py --samples NCBI776 TENX40
```

`train.py` without `--samples` runs on synthetic data.

## Limitations

- No learned expression encoder — 50 genes through one MLP is a thin
  representation of a transcriptome.
- One cohort per organ, one sample per cohort. Organ is confounded with everything
  else that differs between two studies, so a cross-organ drop is not proof of an
  organ effect specifically.
- No batch-effect correction. That is the natural follow-up.
- Two cohorts means the cohort and organ silhouettes are the same partition, and
  the platform silhouette is undefined.

## Licence

MIT.

# cross-modal-alignment

**Cross-cohort generalization of image–expression alignment**

> **Status: scaffold.** The pipeline is implemented and passes an end-to-end smoke
> test on synthetic data. No real data has been processed and there are no results
> yet. This README will be rewritten once there are.

---

## Background

Spatial transcriptomics gives paired data: a histology image (a photo of tissue
under a microscope) and a gene expression profile (which genes are active at that
location), both measured from the same physical spot on a tissue slice. Foundation
models exist for each modality separately. This project asks whether their internal
representations can be aligned across modalities, and whether that alignment
survives when tested on tissue samples from labs and platforms the model was never
trained on.

## Research question

When we train a contrastive alignment between frozen histopathology-image embeddings
and gene-expression embeddings on one set of tissue cohorts, does it generalize to
held-out cohorts — or was it fitting technical batch structure rather than biology?

The experiment is a comparison between four evaluation protocols:

| Split | What is held out | What it tests |
|---|---|---|
| `random` | nothing (spots shuffled) | optimistic baseline |
| `cohort` | a whole study | transfer across lab and protocol |
| `organ` | a whole tissue type | whether anything general about tissue was learned |
| `platform` | a whole sequencing technology | the purely technical case |

**The gap between `random` and the rest is the finding.** If random-split retrieval
looks strong and leave-one-cohort-out collapses, the model was matching pairs on
batch signature — stain colour, scanner profile, sequencing depth — rather than on
any relationship between morphology and expression. The silhouette scores in
`evaluate.py` are there to test that explanation directly.

## Approach

Both encoders are frozen; the only trainable parameters are two small MLP projection
heads into a shared 128-d space, trained with InfoNCE.

- **Image side:** [Phikon-v2](https://huggingface.co/owkin/phikon-v2) (ViT-L/16,
  DINOv2, 1024-d CLS token), frozen. Chosen over UNI and H-optimus because those
  are gated behind institutional-email verification — Phikon-v2 is openly
  downloadable, so these results are actually reproducible. It is trained on ~450M
  20x histology tiles from public cohorts (TCGA, CPTAC, GTEx), which matches the
  magnification HEST patches are cut at, and it appears in the HEST-Benchmark
  comparison table, so its behaviour on this dataset family is already documented.
  Licence is non-commercial, which is fine for research use.
- **Expression side:** the 50 highest-variance genes, library-size normalised to
  10,000 counts per spot and log1p transformed. No learned expression encoder in v1 —
  the normalised vector goes straight into its projection head.
- **Data:** [HEST-1k](https://github.com/mahmoodlab/hest)
  ([paper](https://arxiv.org/abs/2406.16192)) — 1,229 spatial transcriptomic profiles
  paired with H&E patches, across 153 cohorts, 26 organs and 3 platforms. v1 uses 3–4
  cohorts, deliberately chosen to span at least two organs and two platforms.

### Note on the gene panel

HEST-Benchmark ships a 50-gene panel, but it is selected *per organ/indication* — gene
7 in a breast task is not gene 7 in a kidney task. Those panels are therefore unusable
for a cross-organ question: the input vector's meaning would change between train and
test. Instead this project builds its own panel from the full transcriptome:
intersect the genes present in all selected cohorts, then take the 50 most variable,
**fit on training spots only and refit for every split**. Selecting genes using the
held-out cohort would leak the test distribution into the input space and inflate
exactly the number the project is trying to measure honestly.

## Layout

```
src/
  data_loading.py   HEST loading, expression normalisation, gene panel selection
  embed_images.py   frozen encoder -> cached embeddings (the only GPU step)
  model.py          two projection heads + InfoNCE
  train.py          training loop over cached embeddings (CPU)
  evaluate.py       retrieval metrics + batch-vs-biology silhouette
  splits.py         the four split strategies
tests/
  smoke_test.py     end-to-end run on synthetic data, <30s, CPU
notebooks/
  exploration.ipynb scratch
```

`data/`, `embeddings/` and `results/` are gitignored — all of it regenerates from
the scripts.

### Why embeddings are cached

The image encoder is frozen, so a given patch produces the same vector forever.
Re-running a 300M-parameter ViT for each of four split strategies would be paying
GPU time to recompute a constant. `embed_images.py` does one pass and writes to
`embeddings/<encoder>/<sample_id>.npz`; everything downstream reads from there. The
practical consequence is that only that one step needs a GPU — after it, a full
four-strategy experiment runs on a laptop CPU in seconds.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`hest` is installed separately (`requirements-data.txt`) because it pulls in a large,
fragile dependency stack that is only needed for the initial data pull.

## Usage

```bash
# 1. Verify the whole pipeline on synthetic data. No downloads, no GPU.
python tests/smoke_test.py

# 2. Cache image embeddings. Needs a GPU and HEST data on disk.
python src/embed_images.py --samples TENX95 TENX99 --data-dir data

# 3. Train and evaluate all four split strategies from the cache.
python src/train.py --samples TENX95 TENX99
```

`train.py` with no `--samples` runs on synthetic data, which is useful for checking
a change end to end without touching the real dataset.

## Current limitations

Worth stating plainly, since several are deliberate v1 scope cuts:

- No learned expression encoder — a 50-d vector through one MLP is a weak
  representation of a transcriptome.
- No batch-effect correction. That is the intended follow-up comparison: if the
  cross-cohort gap closes after correction, the gap was batch structure.
- Cohorts are not organ-balanced in HEST, so cohort identity partly proxies for
  organ. A high cohort silhouette alongside a high organ silhouette is ambiguous;
  the clear bad case is a high cohort silhouette with a low organ one.
- Small scale by design — a handful of cohorts, not all 1,229 samples.
- The HEST I/O path in `data_loading.py` is written against the documented API but
  has not yet been run against real data.

## Licence

MIT.

# Contrastive image–expression alignment on HEST-Benchmark (IDC)

A small study of contrastive alignment between histology images and spatial gene expression, evaluated on the breast cancer (IDC) task of HEST-Benchmark. I first reproduce the benchmark's published regression baseline, then train a contrastive alignment on the same patient-held-out splits and compare the two.

## Motivation

H&E-stained tissue images are taken routinely in pathology and are cheap to produce. Spatial transcriptomics measures which genes are active at each location in the tissue, but it is expensive and rare. If a model can learn the relationship between what tissue looks like and what its genes are doing, molecular information could in principle be inferred from routine images.

Two common ways to connect the modalities are direct regression (predict genes from image features) and contrastive alignment (map both modalities into a shared space where matching pairs sit close together). This project compares them under the same standard protocol.

## Setup

**Data.** The IDC task from [HEST-Benchmark](https://huggingface.co/datasets/MahmoodLab/hest-bench): 4 patients, 35,536 spots, each with a 224×224 H&E patch at 20× and an expression profile. I use the benchmark's official patient-stratified splits (4 folds, leave-one-patient-out) and its provided list of 50 highly variable genes. Expression is log1p-transformed, matching the benchmark code.

**Image encoder.** [Phikon-v2](https://huggingface.co/owkin/phikon-v2), kept frozen. 1024-dim CLS embeddings, extracted once and cached. I chose Phikon-v2 because it is openly available; several stronger pathology encoders require institutional access.

**Hardware.** All runs on a laptop (Apple M4 Max, MPS). Embedding all patches took about 13 minutes; alignment training takes seconds to minutes per fold.

## Methods

**1. Baseline reproduction.** Following the benchmark's own code: standardise image features, reduce to 256 dimensions with PCA, fit a ridge regression to the 50 genes. Scored by Pearson correlation per gene, averaged over genes per fold, then mean ± std over folds.

**2. Contrastive alignment.** Two small MLP heads map image features and expression vectors into a shared 128-dim space, trained with a symmetric InfoNCE loss. A positive pair is the image patch and expression profile from the same spot; every other pairing in the batch acts as a negative. A learned temperature controls how sharply the loss separates the true match from the rest. Both inputs are standardised using statistics from the training fold only. Early stopping uses a random 10% of training-fold spots; the test patient is never seen during training.

Hyperparameters, shared across folds: embedding dim 128, batch 256, lr 1e-3, weight decay 1e-4, dropout 0.1, patience 15, max 100 epochs, seed 0.

**3. Evaluation on the held-out patient.**
- *Gene prediction by nearest neighbours.* For each test spot, find its k nearest training spots and average their measured expression. Done in two spaces: raw Phikon-v2 features (no alignment), and the aligned space (test image embedding against training expression embeddings). This separates the effect of alignment from the simpler fact that similar-looking tissue tends to have similar expression. Scored exactly like the baseline. k = 10, 50, 100, all reported.
- *Cross-modal retrieval.* Recall@1/5/10 and MRR in both directions. Test patients range from 3,005 to 20,761 spots, so retrieval is computed on random 1,000-spot subsets (10 draws) to keep folds comparable.

## Results

**Baseline reproduction.** 0.5408 ± 0.0773, matching the [published Phikon-v2 IDC result](https://github.com/mahmoodlab/HEST) of 0.5408.

**Gene prediction (Pearson, mean over 50 genes).**

| Method | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean ± std |
|---|---|---|---|---|---|
| Ridge (benchmark protocol) | 0.4504 | 0.5926 | 0.4816 | 0.6385 | **0.5408 ± 0.0773** |
| kNN, raw features, k=10 | 0.3659 | 0.4691 | 0.4088 | 0.5170 | 0.4402 ± 0.0575 |
| kNN, raw features, k=50 | 0.3803 | 0.4987 | 0.4435 | 0.5541 | 0.4692 ± 0.0645 |
| kNN, raw features, k=100 | 0.3761 | 0.4994 | 0.4535 | 0.5635 | 0.4731 ± 0.0683 |
| kNN, aligned space, k=10 | 0.4189 | 0.5174 | 0.4191 | 0.4888 | 0.4610 ± 0.0432 |
| kNN, aligned space, k=50 | 0.4366 | 0.5373 | 0.4401 | 0.5156 | 0.4824 ± 0.0447 |
| kNN, aligned space, k=100 | 0.4412 | 0.5447 | 0.4460 | 0.5238 | 0.4889 ± 0.0459 |

Held-out patients: fold 0 TENX99 (20,761 spots), fold 1 TENX95 (7,760), fold 2 NCBI785 (4,010), fold 3 NCBI783 (3,005).

**Cross-modal retrieval (1,000-spot subsets, mean ± std over folds).**

| Direction | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Image → expression | 0.0166 ± 0.0074 | 0.0617 ± 0.0277 | 0.1088 ± 0.0497 | 0.0519 ± 0.0203 |
| Expression → image | 0.0162 ± 0.0092 | 0.0610 ± 0.0322 | 0.1046 ± 0.0551 | 0.0495 ± 0.0232 |

Chance R@1 is 0.001.

## What the results show

**Ridge is the best gene predictor here** (0.541 vs 0.489 for the best alignment-based method). That is not surprising: ridge is fitted directly to predict genes, while the alignment is trained to match pairs, and prediction is a by-product.

**Alignment gives a small, inconsistent improvement over raw-feature lookup.** On average, aligned kNN beats raw kNN at every k (by 0.013 to 0.021) and varies less across patients. Per patient, though, the picture is mixed: at k=100 it helps on folds 0 and 1, is roughly even on fold 2, and is worse on fold 3. With four patients, an average gain this small cannot be claimed with confidence.

**Retrieval carries a real but modest signal.** Recall@1 is about 16× chance on patients the model never saw, and both directions behave similarly. Exact-spot retrieval is inherently hard here: neighbouring spots look and express almost identically, so many "wrong" retrievals are near-duplicates of the right one.

## Limitations

- Four patients, so error bars are wide and small differences between methods are not meaningful.
- A single random seed for alignment training.
- kNN performance was still rising at k=100, so its ceiling is unknown.
- Early stopping uses spots from the training patients, which makes stopping slightly optimistic. It never touches the test patient.
- One task, one encoder.

## What I'd do next

1. **Run the same protocol on other HEST-Benchmark tasks.** Nothing changes in the code, and more patients and tissues would show whether the small alignment gain is real or noise.
2. **Multiple seeds and a wider range of k.** Cheap, and it would turn the current comparison into one with proper uncertainty.
3. **Ridge on the aligned image embeddings.** Tests whether the shared space is a better input for regression than raw features, combining the strengths of both approaches.

## A note on evaluation design

An earlier pilot on a single tissue slice used a random split of spots into train and test. Retrieval there looked far stronger, around 155× chance. With patient-held-out splits it drops to about 16×. The setups are not directly comparable (different sample, gene list and normalisation), but the size of the gap is consistent with spatial leakage: in a random split within one slice, most test spots have near-identical neighbours in the training set. The main takeaway for me was that the split design can matter more than the model.

## Reproducing

Python 3.11, run from the repository root.

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Hugging Face. Neither repo is gated at the time of writing, so this is
# optional; it avoids anonymous rate limits.
hf auth login

# Data: the 23 IDC files (patches, adata, splits, gene list), ~5.6 GB.
hf download MahmoodLab/hest-bench --repo-type dataset --include "IDC/*" --local-dir data/hest-bench

# Cache frozen Phikon-v2 CLS embeddings for all 35,536 patches (~13 min on MPS).
python src/embed_images.py --samples TENX95 TENX99 NCBI783 NCBI785 --data-dir data/hest-bench/IDC

# Step 1: PCA + ridge baseline -> results/benchmark_idc.json
python src/benchmark.py

# Steps 2-3: alignment training and evaluation -> results/alignment_idc.json
python src/alignment.py
```

`embed_images.py` picks MPS, then CUDA, then CPU unless `--device` says otherwise; the other two scripts run on CPU in a few minutes. `python tests/smoke_test.py` exercises the pipeline on synthetic data in a couple of seconds and needs no download.

## References

- Jaume et al. *HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis.* NeurIPS 2024. [arXiv:2406.16192](https://arxiv.org/abs/2406.16192)
- Filiot et al. *Phikon-v2, A large and public feature extractor for biomarker prediction.* 2024. [arXiv:2409.09173](https://arxiv.org/abs/2409.09173)
- Xie et al. *Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning (BLEEP).* NeurIPS 2023.
- Gindra et al. *A Large-Scale Benchmark of Cross-Modal Learning for Histology and Gene Expression in Spatial Transcriptomics (HESCAPE).* ICCV Workshops 2025. [arXiv:2508.01490](https://arxiv.org/abs/2508.01490)
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.

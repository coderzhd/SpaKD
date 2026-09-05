# SpaKD

SpaKD is a reference-guided framework for predicting spatial expression of genes that are available in a scRNA-seq reference but unmeasured in spatial transcriptomics data.

The public code is organized around the method described in the manuscript:

- Dual-branch cross-modal framework with an ST encoder branch and an scRNA-derived prediction branch.
- Multi-level knowledge distillation with sample-level contrastive distillation (SCD), global relational distillation (GRD), and expression-derived prototype distillation (EPD).
- Shared structured basis-residual decoder for recurrent spatial patterns and gene-specific variation.

## Repository Layout

```text
spakd/
  data.py           Dataset loading, scRNA reference aggregation, EPD group labels
  metrics.py        PCC, SSIM, RMSE, and JS divergence
  model.py          SpaKD model, decoder, and distillation losses
  train_kfold.py    K-fold training and ensemble evaluation
scripts/
  run_spakd_kfold.py
docs/
  method_mapping.md
tests/
  test_model_objective.py
```

## Installation

```bash
conda env create -f environment.yaml
conda activate SpaKD
pip install -e .
```

You can also install with pip:

```bash
pip install -r requirements.txt
pip install -e .
```

## Datasets

SpaKD was evaluated on the same 53 paired scRNA-seq and spatial
transcriptomics datasets used in SpaIM.

The processed datasets used in our experiments were obtained from
the public SpaIM data resource:

- SpaIM:
  https://github.com/QSong-github/SpaIM

SpaIM extends the 45-dataset transcript-distribution prediction
benchmark introduced by Li et al. (2022) with eight additional
high-resolution imaging-based spatial transcriptomics datasets.

- Original SpatialBenchmarking resource:
  https://github.com/QuKunLab/SpatialBenchmarking

For detailed platform, tissue, and dataset-size information of the
53 datasets used in SpaKD, please refer to Supplementary Table S1
of our manuscript.

## Data Layout

SpaKD expects each benchmark dataset to use the following structure:

```text
dataset/
  Dataset1/
    scRNA_count_cluster.h5ad
    Insitu_count.h5ad
    train_list.npy
    test_list.npy
  Dataset2/
    ...
```

SpaKD filters out scRNA-seq genes expressed in fewer than 10% of reference cells, then applies `log1p` to both the scRNA-seq reference and ST matrices when they are not already marked as log-transformed.

The scRNA-seq AnnData file may contain a precomputed Leiden annotation in `.obs`. By default SpaKD uses `leiden`; if this key is absent, it computes Leiden groups from the scRNA-seq reference:

```bash
--cluster_key leiden --leiden_resolution 1.0 --leiden_random_state 0
```

For each gene, SpaKD builds the EPD label by aggregating scRNA-seq expression across reference groups and assigning the gene to the group with maximal aggregated expression.

## Training Objective

The public implementation follows the manuscript objective directly:

```text
L = lambda_student_rec * L_rec^s
  + lambda_teacher_rec * L_rec^t
  + lambda_scd * L_SCD
  + lambda_grd * L_GRD
  + lambda_epd * L_EPD
```

`L_rec^s` and `L_rec^t` are computed as the mini-batch mean of per-gene squared L2 reconstruction errors. `L_GRD` and `L_EPD` use squared Frobenius differences between the corresponding similarity matrices. Earlier exploratory losses from the development code, such as direct latent matching, coarse-output matching, L1 reconstruction, residual penalties, basis-diversity regularization, and random feature-masking augmentation, are not part of this cleaned manuscript-aligned training objective.

## Training

Run one dataset with ten folds:

```bash
python scripts/run_spakd_kfold.py \
  --root ./dataset \
  --dataset_ids 1 \
  --folds 0 1 2 3 4 5 6 7 8 9 \
  --epochs 50 \
  --batch_size 64 \
  --lambda_student_rec 1.0 \
  --lambda_teacher_rec 0.5 \
  --save_root ./SpaKD_results
```

Run several datasets:

```bash
python scripts/run_spakd_kfold.py \
  --root ./dataset \
  --dataset_ids 1 2 3 \
  --folds 0 1 2 3 4 5 6 7 8 9
```

Evaluate existing checkpoints without retraining:

```bash
python scripts/run_spakd_kfold.py \
  --root ./dataset \
  --dataset_ids 1 \
  --folds 0 1 2 3 4 5 6 7 8 9 \
  --no_train
```

Outputs are written under `SpaKD_results/` by default. Raw datasets, trained checkpoints, and result pickle files are intentionally ignored by Git.

## Method Names

The development code used older names inherited from the SpaIM/SpaTS pipeline. The cleaned repository uses the manuscript terminology. See `docs/method_mapping.md`.

## Citation

The manuscript citation will be added after publication.

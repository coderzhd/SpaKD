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
  test_model_smoke.py
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

The scRNA-seq AnnData file should contain a cell-group annotation in `.obs`. By default SpaKD uses `merge_cell_type`:

```bash
--cluster_key merge_cell_type
```

For each gene, SpaKD builds the EPD label by aggregating scRNA-seq expression across reference groups and assigning the gene to the group with maximal aggregated expression.

## Training

Run one dataset with ten folds:

```bash
python scripts/run_spakd_kfold.py \
  --root ./dataset \
  --dataset_ids 1 \
  --folds 0 1 2 3 4 5 6 7 8 9 \
  --epochs 50 \
  --batch_size 64 \
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

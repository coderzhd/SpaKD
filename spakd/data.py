"""Data loading utilities for SpaKD."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset


def load_h5ad(path: str, min_cells: int = 3, min_genes: int = 3):
    """Load and lightly filter an AnnData file."""

    adata = sc.read(path)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if "log1p" not in adata.uns_keys():
        sc.pp.log1p(adata)
    return adata


def _ordered_intersection(candidates: Iterable[str], allowed: Sequence[str]) -> List[str]:
    allowed_set = set(allowed)
    return [str(item) for item in candidates if str(item) in allowed_set]


def _read_split(path: str, fold: int) -> List[str]:
    split = np.load(path, allow_pickle=True).tolist()
    return [str(gene) for gene in split[fold]]


def _row_to_vector(matrix, index: int) -> np.ndarray:
    row = matrix[index, ...]
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row, dtype=np.float32).reshape(-1)


def _matrix_to_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


class SpaKDDataset(Dataset):
    """Gene-wise dataset for reference-guided ST imputation."""

    def __init__(
        self,
        root: str,
        dataset_name: str,
        fold: int,
        split: str = "train",
        cluster_key: str = "merge_cell_type",
        train_list_name: str = "train_list.npy",
        test_list_name: str = "test_list.npy",
        min_cells: int = 3,
        min_genes: int = 3,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.root = root
        self.dataset_name = dataset_name
        self.fold = fold
        self.split = split
        self.cluster_key = cluster_key
        self.dataset_dir = os.path.join(root, dataset_name)

        self.sc_path = os.path.join(self.dataset_dir, "scRNA_count_cluster.h5ad")
        self.st_path = os.path.join(self.dataset_dir, "Insitu_count.h5ad")
        self.sc_adata = load_h5ad(self.sc_path, min_cells=min_cells, min_genes=min_genes)
        self.st_adata = load_h5ad(self.st_path, min_cells=min_cells, min_genes=min_genes)

        self.reference_group_codes = self._reference_group_codes()
        self.aggregated_reference = self._aggregate_reference_profiles()

        train_genes = _read_split(os.path.join(self.dataset_dir, train_list_name), fold)
        test_path = os.path.join(self.dataset_dir, test_list_name)
        if os.path.exists(test_path):
            heldout_genes = _read_split(test_path, fold)
        else:
            heldout_genes = [
                gene for gene in self.st_adata.var_names if gene not in set(train_genes)
            ]

        available_train = (
            set(self.aggregated_reference.var_names)
            & set(self.st_adata.var_names)
            & set(train_genes)
        )
        available_val = (
            set(self.aggregated_reference.var_names)
            & set(self.st_adata.var_names)
            & set(heldout_genes)
        )

        self.train_gene_names = _ordered_intersection(train_genes, available_train)
        self.val_gene_names = _ordered_intersection(heldout_genes, available_val)
        if not self.train_gene_names:
            raise ValueError(f"No training genes found for {dataset_name}, fold {fold}")
        if not self.val_gene_names:
            raise ValueError(f"No held-out genes found for {dataset_name}, fold {fold}")

        self.sc_train = self.aggregated_reference[:, self.train_gene_names].copy().T
        self.sc_val = self.aggregated_reference[:, self.val_gene_names].copy().T
        self.st_train = self.st_adata[:, self.train_gene_names].copy().T
        self.st_val = self.st_adata[:, self.val_gene_names].copy().T

        self.train_gene_group = self._expression_groups(self.sc_train.X)
        self.val_gene_group = self._expression_groups(self.sc_val.X)

    @property
    def sc_dim(self) -> int:
        return int(self.sc_train.shape[1])

    @property
    def st_dim(self) -> int:
        return int(self.st_train.shape[1])

    @property
    def st_location_names(self) -> List[str]:
        return [str(name) for name in self.st_adata.obs_names]

    def get_eval_names(self) -> Tuple[List[str], List[str]]:
        return [str(name) for name in self.st_val.obs_names], [
            str(name) for name in self.st_val.var_names
        ]

    def get_cluster_dim(self) -> int:
        return int(len(np.unique(self.reference_group_codes)))

    def _reference_group_codes(self) -> np.ndarray:
        if self.cluster_key not in self.sc_adata.obs:
            available = ", ".join(map(str, self.sc_adata.obs_keys()))
            raise KeyError(
                f"cluster_key '{self.cluster_key}' is not in scRNA obs. "
                f"Available obs keys: {available}"
            )
        labels = self.sc_adata.obs[self.cluster_key].astype("category")
        return labels.cat.codes.to_numpy(dtype=np.int64)

    def _aggregate_reference_profiles(self):
        matrix = self.sc_adata.X
        n_groups = len(np.unique(self.reference_group_codes))
        aggregated = np.zeros((n_groups, self.sc_adata.shape[1]), dtype=np.float32)
        for group in range(n_groups):
            mask = self.reference_group_codes == group
            if not np.any(mask):
                continue
            mean_expr = matrix[mask].mean(axis=0)
            aggregated[group] = np.asarray(mean_expr, dtype=np.float32).reshape(-1)

        adata = sc.AnnData(aggregated)
        adata.var_names = self.sc_adata.var_names.copy()
        adata.obs_names = [f"group_{i}" for i in range(n_groups)]
        return adata

    @staticmethod
    def _expression_groups(matrix) -> np.ndarray:
        values = _matrix_to_array(matrix)
        return np.argmax(values, axis=1).astype(np.int64)

    def __len__(self) -> int:
        if self.split == "train":
            return int(self.sc_train.shape[0])
        return int(self.sc_val.shape[0])

    def __getitem__(self, index: int):
        if self.split == "train":
            sc_profile = _row_to_vector(self.sc_train.X, index)
            st_profile = _row_to_vector(self.st_train.X, index)
            gene_group = int(self.train_gene_group[index])
        else:
            sc_profile = _row_to_vector(self.sc_val.X, index)
            st_profile = _row_to_vector(self.st_val.X, index)
            gene_group = int(self.val_gene_group[index])

        return (
            torch.from_numpy(sc_profile),
            torch.from_numpy(st_profile),
            torch.tensor(gene_group, dtype=torch.long),
        )

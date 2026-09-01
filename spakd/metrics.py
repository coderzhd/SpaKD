"""Evaluation metrics used by the SpaKD benchmark."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as st


def scale_max(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.copy()
    denom = scaled.max(axis=0).replace(0, np.nan)
    return scaled.divide(denom, axis=1).fillna(0)


def scale_z_score(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        st.zscore(df, axis=0, nan_policy="omit"),
        index=df.index,
        columns=df.columns,
    ).replace([np.inf, -np.inf], 0).fillna(0)


def scale_sum(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.copy()
    denom = scaled.sum(axis=0).replace(0, np.nan)
    return scaled.divide(denom, axis=1).fillna(0)


def ssim_1d(raw: np.ndarray, pred: np.ndarray, dynamic_range: float) -> float:
    raw = raw.reshape(-1, 1)
    pred = pred.reshape(-1, 1)
    mu_raw = raw.mean()
    mu_pred = pred.mean()
    sigma_raw = np.sqrt(((raw - mu_raw) ** 2).mean())
    sigma_pred = np.sqrt(((pred - mu_pred) ** 2).mean())
    sigma_cross = ((raw - mu_raw) * (pred - mu_pred)).mean()

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    c3 = c2 / 2
    luminance = (2 * mu_raw * mu_pred + c1) / (mu_raw**2 + mu_pred**2 + c1)
    contrast = (2 * sigma_raw * sigma_pred + c2) / (
        sigma_raw**2 + sigma_pred**2 + c2
    )
    structure = (sigma_cross + c3) / (sigma_raw * sigma_pred + c3)
    return float(luminance * contrast * structure)


def compute_all_metrics(raw: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Compute PCC, SSIM, RMSE and JS divergence for each gene.

    Both inputs are expected to be shaped as spatial locations/cells by genes.
    """

    common_genes = [gene for gene in raw.columns if gene in pred.columns]
    raw = raw.loc[:, common_genes].copy()
    pred = pred.loc[:, common_genes].copy()
    pred.index = raw.index
    pred[pred < 0] = 0

    raw_ssim = scale_max(raw)
    pred_ssim = scale_max(pred)
    raw_rmse = scale_z_score(raw)
    pred_rmse = scale_z_score(pred)
    raw_js = scale_sum(np.expm1(raw))
    pred_js = scale_sum(np.expm1(pred))

    rows = []
    for gene in common_genes:
        raw_col = raw[gene].fillna(1e-20)
        pred_col = pred[gene].fillna(1e-20)
        try:
            pcc = float(st.pearsonr(raw_col, pred_col)[0])
        except Exception:
            pcc = 0.0

        ssim_range = max(float(raw_ssim[gene].max()), float(pred_ssim[gene].max()))
        if ssim_range <= 0:
            ssim_range = 1.0
        ssim = ssim_1d(raw_ssim[gene].to_numpy(), pred_ssim[gene].to_numpy(), ssim_range)

        rmse = float(np.sqrt(((raw_rmse[gene] - pred_rmse[gene]) ** 2).mean()))
        mixture = (raw_js[gene].to_numpy() + pred_js[gene].to_numpy()) / 2
        js = float(
            (st.entropy(raw_js[gene].to_numpy(), mixture) + st.entropy(pred_js[gene].to_numpy(), mixture))
            / 2
        )
        rows.append({"gene": gene, "PCC": pcc, "SSIM": ssim, "RMSE": rmse, "JS": js})

    return pd.DataFrame(rows).set_index("gene")


def save_metrics(
    raw: pd.DataFrame,
    pred: pd.DataFrame,
    output_dir: str,
    prefix: str,
    fold: Optional[int | str] = None,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    metrics = compute_all_metrics(raw, pred)
    suffix = "none" if fold is None else str(fold)
    metrics.to_csv(os.path.join(output_dir, f"{prefix}_metrics_{suffix}.tsv"), sep="\t")
    return metrics

"""K-fold training and evaluation entry points for SpaKD."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import SpaKDDataset
from .metrics import save_metrics
from .model import SpaKDConfig, SpaKDModule


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def augment_training_batch(
    sc_profile: torch.Tensor,
    st_profile: torch.Tensor,
    gene_group: torch.Tensor,
    times: int = 4,
    zero_fraction: float = 0.25403000434604417,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Augment the scRNA-derived profile by random feature masking."""

    if times <= 1:
        return sc_profile, st_profile, gene_group

    sc_batches = [sc_profile]
    for _ in range(times - 1):
        mask = torch.rand(sc_profile.shape, device=sc_profile.device) < zero_fraction
        sc_batches.append(torch.where(mask, torch.zeros_like(sc_profile), sc_profile))

    return (
        torch.cat(sc_batches, dim=0),
        torch.cat([st_profile] * times, dim=0),
        torch.cat([gene_group] * times, dim=0),
    )


def build_loader(dataset: SpaKDDataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def config_from_args(args: argparse.Namespace, dataset: SpaKDDataset) -> SpaKDConfig:
    return SpaKDConfig(
        sc_dim=dataset.sc_dim,
        st_dim=dataset.st_dim,
        latent_dim=args.latent_dim,
        student_hidden_dims=tuple(args.hidden_dims),
        teacher_hidden_dims=tuple(args.hidden_dims),
        num_bases=args.num_bases,
        dropout=args.dropout,
        activation=args.activation,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        gpu=args.gpu,
        parallel=args.parallel,
        lambda_teacher_rec=args.lambda_teacher_rec,
        lambda_l1=args.lambda_l1,
        lambda_student_coarse=args.lambda_student_coarse,
        lambda_latent_mse=args.lambda_latent_mse,
        lambda_coarse_mse=args.lambda_coarse_mse,
        lambda_residual=args.lambda_residual,
        lambda_scd=args.lambda_scd,
        scd_margin=args.scd_margin,
        lambda_grd=args.lambda_grd,
        lambda_epd=args.lambda_epd,
    )


def train_one_fold(
    args: argparse.Namespace,
    dataset_name: str,
    fold: int,
    save_dir: str,
) -> SpaKDConfig:
    dataset = SpaKDDataset(
        root=args.root,
        dataset_name=dataset_name,
        fold=fold,
        split="train",
        cluster_key=args.cluster_key,
        train_list_name=args.train_list_name,
        test_list_name=args.test_list_name,
    )
    loader = build_loader(dataset, args.batch_size, shuffle=True, seed=args.seed)
    config = config_from_args(args, dataset)
    module = SpaKDModule(config)

    for epoch in range(args.epochs):
        totals: Dict[str, float] = {}
        n_batches = 0
        for sc_profile, st_profile, gene_group in loader:
            n_original = int(sc_profile.size(0))
            sc_profile, st_profile, gene_group = augment_training_batch(
                sc_profile,
                st_profile,
                gene_group,
                times=args.aug_times,
                zero_fraction=args.zero_fraction,
            )
            module.set_input(
                {
                    "scx": sc_profile,
                    "stx": st_profile,
                    "gene_group": gene_group,
                    "n_original": n_original,
                },
                train=True,
            )
            module.update_parameters()
            for key, value in module.get_current_loss().items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1

        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            denom = max(n_batches, 1)
            loss = totals.get("loss", 0.0) / denom
            scd = totals.get("scd", 0.0) / denom
            grd = totals.get("grd", 0.0) / denom
            epd = totals.get("epd", 0.0) / denom
            print(
                f"[{dataset_name} fold {fold}] epoch {epoch + 1}/{args.epochs} "
                f"loss={loss:.4f} scd={scd:.4f} grd={grd:.4f} epd={epd:.4f}",
                flush=True,
            )

    os.makedirs(save_dir, exist_ok=True)
    module.save(os.path.join(save_dir, f"last_{fold}.pth"))
    return config


def validate_one_fold(
    args: argparse.Namespace,
    dataset_name: str,
    fold: int,
    save_dir: str,
    config: SpaKDConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    dataset = SpaKDDataset(
        root=args.root,
        dataset_name=dataset_name,
        fold=fold,
        split="val",
        cluster_key=args.cluster_key,
        train_list_name=args.train_list_name,
        test_list_name=args.test_list_name,
    )
    loader = build_loader(dataset, args.batch_size, shuffle=False, seed=args.seed)
    if config is None:
        config = config_from_args(args, dataset)
    else:
        config = replace(config, sc_dim=dataset.sc_dim, st_dim=dataset.st_dim)

    module = SpaKDModule(config)
    module.load(os.path.join(save_dir, f"last_{fold}.pth"))

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for sc_profile, st_profile, gene_group in loader:
            module.set_input(
                {"scx": sc_profile, "gene_group": gene_group},
                train=False,
            )
            out = module.inference()
            pred = out["st_fake"].detach().cpu().numpy()
            pred[pred < 0] = 0
            predictions.append(pred)
            targets.append(st_profile.detach().cpu().numpy())

    pred_matrix = np.concatenate(predictions, axis=0)
    target_matrix = np.concatenate(targets, axis=0)
    gene_names, location_names = dataset.get_eval_names()

    df_pred = pd.DataFrame(pred_matrix.T, index=location_names, columns=gene_names)
    df_target = pd.DataFrame(target_matrix.T, index=location_names, columns=gene_names)

    df_pred.to_pickle(os.path.join(save_dir, f"impute_result_{fold}.pkl"))
    df_target.to_pickle(os.path.join(save_dir, f"input_result_{fold}.pkl"))
    metrics = save_metrics(df_target, df_pred, save_dir, prefix="spakd", fold=fold)
    summary = {
        "PCC": float(metrics["PCC"].mean()),
        "SSIM": float(metrics["SSIM"].mean()),
        "RMSE": float(metrics["RMSE"].mean()),
        "JS": float(metrics["JS"].mean()),
    }
    print(
        f"[{dataset_name} fold {fold}] "
        f"PCC={summary['PCC']:.6f} SSIM={summary['SSIM']:.6f} "
        f"RMSE={summary['RMSE']:.6f} JS={summary['JS']:.6f}",
        flush=True,
    )
    return df_pred, df_target, summary


def merge_duplicate_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    return df.T.groupby(level=0).mean().T


def run_dataset(args: argparse.Namespace, dataset_name: str) -> Dict[str, float]:
    dataset_root = os.path.join(args.save_root, dataset_name)
    fold_summaries = []
    pred_dfs = []
    target_dfs = []

    for fold in args.folds:
        seed_everything(args.seed)
        fold_dir = os.path.join(dataset_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        config = None
        if not args.no_train:
            config = train_one_fold(args, dataset_name, fold, fold_dir)
        df_pred, df_target, summary = validate_one_fold(
            args,
            dataset_name,
            fold,
            fold_dir,
            config=config,
        )
        fold_summaries.append({"fold": fold, **summary})
        pred_dfs.append(df_pred)
        target_dfs.append(df_target)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_summary_df = pd.DataFrame(fold_summaries)
    fold_summary_df.to_csv(os.path.join(dataset_root, "fold_summary.csv"), index=False)

    pred_all = merge_duplicate_gene_columns(pd.concat(pred_dfs, axis=1))
    target_all = merge_duplicate_gene_columns(pd.concat(target_dfs, axis=1))
    ensemble_dir = os.path.join(dataset_root, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    pred_all.to_pickle(os.path.join(ensemble_dir, "impute_result_ensemble.pkl"))
    target_all.to_pickle(os.path.join(ensemble_dir, "input_result_ensemble.pkl"))

    metrics = save_metrics(target_all, pred_all, ensemble_dir, prefix="spakd", fold=None)
    summary = {
        "dataset_name": dataset_name,
        "PCC": float(metrics["PCC"].mean()),
        "SSIM": float(metrics["SSIM"].mean()),
        "RMSE": float(metrics["RMSE"].mean()),
        "JS": float(metrics["JS"].mean()),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(dataset_root, "dataset_summary.csv"),
        index=False,
    )
    return summary


def run_all_datasets(args: argparse.Namespace) -> None:
    os.makedirs(args.save_root, exist_ok=True)
    summaries = []
    for dataset_id in args.dataset_ids:
        dataset_name = f"Dataset{dataset_id}"
        print(f"===== Running {dataset_name} =====", flush=True)
        summaries.append(run_dataset(args, dataset_name))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(args.save_root, "all_datasets_summary.csv"), index=False)
    mean_row = {
        "dataset_name": "ALL_DATASETS_MEAN",
        "PCC": float(summary_df["PCC"].mean()),
        "SSIM": float(summary_df["SSIM"].mean()),
        "RMSE": float(summary_df["RMSE"].mean()),
        "JS": float(summary_df["JS"].mean()),
    }
    pd.DataFrame([mean_row]).to_csv(
        os.path.join(args.save_root, "all_datasets_mean.csv"),
        index=False,
    )
    print(summary_df, flush=True)
    print(pd.DataFrame([mean_row]), flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate SpaKD.")

    parser.add_argument("--root", type=str, default="./dataset")
    parser.add_argument("--dataset_ids", type=int, nargs="+", default=[1])
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cluster_key", type=str, default="merge_cell_type")
    parser.add_argument("--train_list_name", type=str, default="train_list.npy")
    parser.add_argument("--test_list_name", type=str, default="test_list.npy")
    parser.add_argument("--save_root", type=str, default="./SpaKD_results")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--num_bases", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.026180677070756372)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.0003713287125770932)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--aug_times", type=int, default=4)
    parser.add_argument("--zero_fraction", type=float, default=0.25403000434604417)

    parser.add_argument("--lambda_teacher_rec", type=float, default=0.5)
    parser.add_argument("--lambda_l1", type=float, default=0.4151231486567266)
    parser.add_argument("--lambda_student_coarse", type=float, default=0.3)
    parser.add_argument("--lambda_latent_mse", type=float, default=0.5)
    parser.add_argument("--lambda_coarse_mse", type=float, default=0.3)
    parser.add_argument("--lambda_residual", type=float, default=0.001)
    parser.add_argument("--lambda_scd", type=float, default=0.3)
    parser.add_argument("--scd_margin", type=float, default=1.0)
    parser.add_argument("--lambda_grd", type=float, default=0.1)
    parser.add_argument("--lambda_epd", type=float, default=1.0)

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    run_all_datasets(args)


if __name__ == "__main__":
    main()

"""Core SpaKD model.

The implementation is organized around the method terminology used in the
manuscript:

* an ST encoder branch and an scRNA-derived prediction branch,
* sample-level contrastive distillation (SCD),
* global relational distillation (GRD),
* expression-derived prototype distillation (EPD), and
* a shared structured basis-residual decoder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


@dataclass
class SpaKDConfig:
    """Model and optimization configuration."""

    sc_dim: int
    st_dim: int
    latent_dim: int = 256
    student_hidden_dims: Tuple[int, ...] = (512, 256)
    teacher_hidden_dims: Tuple[int, ...] = (512, 256)
    num_bases: int = 64
    dropout: float = 0.03
    activation: str = "relu"
    basis_temperature: float = 1.0
    use_basis_norm: bool = True
    residual_scale: float = 1.0
    share_decoder: bool = True
    projection_dim: int = 128
    projection_hidden_dim: int = 256

    lr: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    gpu: int = 0
    parallel: bool = False

    lambda_student_rec: float = 1.0
    lambda_teacher_rec: float = 0.5

    lambda_scd: float = 0.3
    scd_margin: float = 1.0
    lambda_grd: float = 0.1
    lambda_epd: float = 1.0


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    return nn.ReLU()


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.backbone = nn.Sequential(
            *[
                MLPBlock(
                    dims[i],
                    dims[i + 1],
                    dropout=dropout,
                    activation=activation,
                    use_norm=True,
                )
                for i in range(len(dims) - 1)
            ]
        )
        final_dim = dims[-1]
        self.proj = None
        if out_dim is not None and out_dim != final_dim:
            self.proj = nn.Linear(final_dim, out_dim)
            final_dim = out_dim
        self.out_dim = final_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        if self.proj is not None:
            h = self.proj(h)
        return h


class ResidualHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(
                in_dim,
                hidden_dim,
                dropout=dropout,
                activation=activation,
                use_norm=True,
            ),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuredBasisResidualDecoder(nn.Module):
    """Shared decoder that separates basis-guided and residual components."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        num_bases: int = 32,
        dropout: float = 0.0,
        activation: str = "relu",
        temperature: float = 1.0,
        use_basis_norm: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_basis_norm = use_basis_norm
        self.residual_scale = residual_scale

        self.alpha_head = nn.Linear(latent_dim, num_bases)
        self.scale_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softplus())
        self.bases = nn.Parameter(torch.randn(num_bases, out_dim) * 0.02)
        self.residual_head = ResidualHead(
            in_dim=latent_dim + out_dim,
            hidden_dim=latent_dim,
            out_dim=out_dim,
            dropout=dropout,
            activation=activation,
        )

    def _basis_bank(self) -> torch.Tensor:
        if self.use_basis_norm:
            return F.normalize(self.bases, dim=1)
        return self.bases

    def forward(self, h: torch.Tensor) -> TensorDict:
        alpha_logits = self.alpha_head(h)
        alpha = torch.softmax(alpha_logits / self.temperature, dim=1)
        scale = self.scale_head(h)
        coarse = scale * torch.matmul(alpha, self._basis_bank())
        residual = self.residual_scale * self.residual_head(torch.cat([h, coarse], dim=1))
        return {
            "latent": h,
            "alpha_logits": alpha_logits,
            "alpha": alpha,
            "scale": scale,
            "coarse": coarse,
            "residual": residual,
            "pred": coarse + residual,
        }


class ProjectionHead(nn.Module):
    """Projection head used only for sample-level contrastive distillation."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SpaKDNetwork(nn.Module):
    """Dual-branch SpaKD network."""

    def __init__(self, config: SpaKDConfig) -> None:
        super().__init__()
        self.config = config

        self.student_encoder = MLPEncoder(
            in_dim=config.sc_dim,
            hidden_dims=config.student_hidden_dims,
            out_dim=config.latent_dim,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.teacher_encoder = MLPEncoder(
            in_dim=config.st_dim,
            hidden_dims=config.teacher_hidden_dims,
            out_dim=config.latent_dim,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.student_decoder = StructuredBasisResidualDecoder(
            latent_dim=config.latent_dim,
            out_dim=config.st_dim,
            num_bases=config.num_bases,
            dropout=config.dropout,
            activation=config.activation,
            temperature=config.basis_temperature,
            use_basis_norm=config.use_basis_norm,
            residual_scale=config.residual_scale,
        )
        if config.share_decoder:
            self.teacher_decoder = self.student_decoder
        else:
            self.teacher_decoder = StructuredBasisResidualDecoder(
                latent_dim=config.latent_dim,
                out_dim=config.st_dim,
                num_bases=config.num_bases,
                dropout=config.dropout,
                activation=config.activation,
                temperature=config.basis_temperature,
                use_basis_norm=config.use_basis_norm,
                residual_scale=config.residual_scale,
            )

        self.student_projection = ProjectionHead(
            config.latent_dim,
            config.projection_hidden_dim,
            config.projection_dim,
        )
        self.teacher_projection = ProjectionHead(
            config.latent_dim,
            config.projection_hidden_dim,
            config.projection_dim,
        )

    def forward(self, sc: torch.Tensor, st: Optional[torch.Tensor] = None) -> TensorDict:
        out: TensorDict = {}

        h_s = self.student_encoder(sc)
        student = self.student_decoder(h_s)
        out.update(
            {
                "student_latent": student["latent"],
                "student_alpha_logits": student["alpha_logits"],
                "student_alpha": student["alpha"],
                "student_scale": student["scale"],
                "student_coarse": student["coarse"],
                "student_residual": student["residual"],
                "student_pred": student["pred"],
                "student_z": self.student_projection(h_s),
            }
        )

        if st is not None:
            h_t = self.teacher_encoder(st)
            teacher = self.teacher_decoder(h_t)
            out.update(
                {
                    "teacher_latent": teacher["latent"],
                    "teacher_alpha_logits": teacher["alpha_logits"],
                    "teacher_alpha": teacher["alpha"],
                    "teacher_scale": teacher["scale"],
                    "teacher_coarse": teacher["coarse"],
                    "teacher_residual": teacher["residual"],
                    "teacher_pred": teacher["pred"],
                    "teacher_z": self.teacher_projection(h_t),
                }
            )

        return out


def sample_contrastive_distillation_loss(
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    margin: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample-level contrastive distillation loss."""

    n = z_s.size(0)
    zero = torch.zeros((), device=z_s.device, dtype=z_s.dtype)
    if n < 2:
        return zero, zero, zero

    z_t = z_t.detach()
    pos_distance = (z_s - z_t).norm(dim=1)
    pos_loss = pos_distance.pow(2).mean()

    distance = (z_s.unsqueeze(1) - z_t.unsqueeze(0)).norm(dim=2)
    eye = torch.eye(n, dtype=torch.bool, device=z_s.device)
    neg_distance = distance[~eye]
    neg_loss = torch.clamp(margin - neg_distance, min=0).pow(2).mean()
    return pos_loss + neg_loss, pos_loss, neg_loss


def global_relational_distillation_loss(
    h_s: torch.Tensor,
    h_t: torch.Tensor,
) -> torch.Tensor:
    """Global relational distillation via cosine-similarity matrix alignment."""

    h_s = F.normalize(h_s, dim=1)
    h_t = F.normalize(h_t.detach(), dim=1)
    sim_s = h_s @ h_s.T
    sim_t = h_t @ h_t.T
    return (sim_s - sim_t).pow(2).sum()


def expression_prototype_distillation_loss(
    h_s: torch.Tensor,
    h_t: torch.Tensor,
    gene_group: torch.Tensor,
) -> torch.Tensor:
    """Expression-derived prototype distillation loss.

    ``gene_group`` is the expression-derived group label for each gene, computed
    as the reference group with the highest aggregated scRNA-seq expression.
    """

    gene_group = gene_group.long().view(-1)
    valid = gene_group >= 0
    h_s = h_s[valid]
    h_t = h_t.detach()[valid]
    gene_group = gene_group[valid]

    if h_s.size(0) == 0:
        return torch.zeros((), device=h_s.device, dtype=h_s.dtype)

    n_groups = int(gene_group.max().item()) + 1
    proto_s = _build_group_prototypes(h_s, gene_group, n_groups)
    proto_t = _build_group_prototypes(h_t, gene_group, n_groups)

    h_s_norm = F.normalize(h_s, dim=1)
    h_t_norm = F.normalize(h_t, dim=1)
    proto_s_norm = F.normalize(proto_s, dim=1)
    proto_t_norm = F.normalize(proto_t, dim=1)

    sim_s = h_s_norm @ proto_s_norm.T
    sim_t = h_t_norm @ proto_t_norm.T
    return (sim_s - sim_t).pow(2).sum()


def _build_group_prototypes(
    h: torch.Tensor,
    labels: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    prototypes = torch.zeros(n_groups, h.size(1), device=h.device, dtype=h.dtype)
    for group in range(n_groups):
        mask = labels == group
        if mask.any():
            prototypes[group] = h[mask].mean(dim=0)
    return prototypes


def _batch_squared_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return 1/N sum_i ||pred_i - target_i||_2^2."""

    if pred.size(0) == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (pred - target).pow(2).sum(dim=1).mean()


class SpaKDModule(nn.Module):
    """Training wrapper around :class:`SpaKDNetwork`."""

    def __init__(self, config: SpaKDConfig) -> None:
        super().__init__()
        self.config = config
        self.loss_stat: Dict[str, float] = {}
        self.device = torch.device(
            f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"
        )

        model: nn.Module = SpaKDNetwork(config)
        if config.parallel and torch.cuda.is_available():
            model = nn.DataParallel(model)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        self.scx: Optional[torch.Tensor] = None
        self.stx: Optional[torch.Tensor] = None
        self.gene_group: Optional[torch.Tensor] = None
        self.n_original: Optional[int] = None
        self.out: TensorDict = {}
        self.loss: Optional[torch.Tensor] = None

    def _unwrap(self) -> SpaKDNetwork:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def set_input(self, inputs: Mapping[str, torch.Tensor], train: bool = True) -> None:
        self.scx = inputs["scx"].to(self.device)
        self.n_original = int(inputs.get("n_original", self.scx.size(0)))
        self.gene_group = inputs.get("gene_group")
        if self.gene_group is not None:
            self.gene_group = self.gene_group.to(self.device)
        self.stx = inputs["stx"].to(self.device) if train else None

    def forward(self) -> TensorDict:
        if self.scx is None:
            raise RuntimeError("set_input must be called before forward")
        self.out = self.model(self.scx, self.stx)
        return self.out

    @torch.no_grad()
    def inference(self) -> TensorDict:
        if self.scx is None:
            raise RuntimeError("set_input must be called before inference")
        self.model.eval()
        out = self.model(self.scx, st=None)
        return {
            "st_fake": out["student_pred"],
            "st_coarse": out["student_coarse"],
            "alpha": out["student_alpha"],
            "scale": out["student_scale"],
        }

    def compute_loss(self) -> torch.Tensor:
        if self.stx is None:
            raise RuntimeError("compute_loss requires spatial targets")
        if self.n_original is None:
            raise RuntimeError("n_original was not set")

        h_s = self.out["student_latent"]
        h_t = self.out["teacher_latent"]
        student_pred = self.out["student_pred"]
        teacher_pred = self.out["teacher_pred"]

        student_rec = _batch_squared_l2_loss(student_pred, self.stx)
        teacher_rec = _batch_squared_l2_loss(teacher_pred, self.stx)

        n0 = self.n_original
        scd_loss, scd_pos, scd_neg = sample_contrastive_distillation_loss(
            self.out["student_z"][:n0],
            self.out["teacher_z"][:n0],
            margin=self.config.scd_margin,
        )
        grd_loss = global_relational_distillation_loss(h_s[:n0], h_t[:n0])

        if self.gene_group is None:
            epd_loss = torch.zeros((), device=self.device)
        else:
            epd_loss = expression_prototype_distillation_loss(
                h_s[:n0],
                h_t[:n0],
                self.gene_group[:n0],
            )

        kd_loss = (
            self.config.lambda_scd * scd_loss
            + self.config.lambda_grd * grd_loss
            + self.config.lambda_epd * epd_loss
        )

        self.loss = (
            self.config.lambda_student_rec * student_rec
            + self.config.lambda_teacher_rec * teacher_rec
            + kd_loss
        )

        self.loss_stat = {
            "loss": float(self.loss.detach().cpu()),
            "student_rec": float(student_rec.detach().cpu()),
            "teacher_rec": float(teacher_rec.detach().cpu()),
            "kd": float(kd_loss.detach().cpu()),
            "scd": float(scd_loss.detach().cpu()),
            "scd_pos": float(scd_pos.detach().cpu()),
            "scd_neg": float(scd_neg.detach().cpu()),
            "grd": float(grd_loss.detach().cpu()),
            "epd": float(epd_loss.detach().cpu()),
        }
        return self.loss

    def backward(self) -> None:
        if self.loss is None:
            raise RuntimeError("compute_loss must be called before backward")
        self.optimizer.zero_grad(set_to_none=True)
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

    def update_parameters(self) -> None:
        self.model.train()
        self.forward()
        self.compute_loss()
        self.backward()

    def save(self, save_path: str) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_dict = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )
        torch.save(state_dict, save_path)

    def load(self, load_path: str, strict: bool = True) -> None:
        state_dict = torch.load(load_path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state_dict, strict=strict)
        else:
            self.model.load_state_dict(state_dict, strict=strict)

    def get_current_loss(self) -> Dict[str, float]:
        return dict(self.loss_stat)

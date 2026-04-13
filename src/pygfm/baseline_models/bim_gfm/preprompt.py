from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import ClassVar

from torch_geometric.utils import dropout_edge

from pygfm.public.model_bases import GFMPrePromptModelBase
from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import NodeNodeContrastiveLoss

GCN_INPUT_RESIDUAL = 0.12


def _split_modalities(x: torch.Tensor, modal_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return x[:, :modal_dim], x[:, modal_dim : modal_dim * 2]


def _covariance(feat: torch.Tensor) -> torch.Tensor:
    n = feat.size(0)
    x = feat - feat.mean(dim=0, keepdim=True)
    return x.t() @ x / max(n - 1, 1)


def _coral_loss_cross_domain(
    h: torch.Tensor,
    batch: torch.Tensor,
    num_domains: int,
    min_nodes: int = 6,
) -> torch.Tensor:
    device = h.device
    acc = torch.tensor(0.0, device=device)
    cnt = 0
    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            mi = batch == i
            mj = batch == j
            if int(mi.sum()) < min_nodes or int(mj.sum()) < min_nodes:
                continue
            acc = acc + F.mse_loss(_covariance(h[mi]), _covariance(h[mj]))
            cnt += 1
    if cnt == 0:
        return torch.tensor(0.0, device=device)
    return acc / cnt


class BimGFMPrePromptModel(GFMPrePromptModelBase):
    """
    BiM-GFM pretraining: shared GCN + per-domain Tdomain/Tdomain_out, masking and edge-drop aug,
    cosine + Smooth L1 alignment, domain classification + optional CORAL.
    """

    gfm_family: ClassVar[str] = "bim_gfm"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_domains: int,
        num_layers: int = 3,
        mask_ratio: float = 0.3,
        recon_weight: float = 1.0,
        align_weight: float = 1.0,
        align_robust_weight: float = 0.12,
        domain_weight: float = 1.0,
        coral_weight: float = 0.03,
        node_contrastive_weight: float = 0.0,
        node_contrastive_temperature: float = 0.5,
        domain_temperature: float = 0.5,
        domain_label_smoothing: float = 0.0,
        edge_drop_ratio: float = 0.0,
        feat_noise_std: float = 0.0,
        prompt_dropout: float = 0.0,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        if input_dim % 2 != 0:
            raise ValueError(f"input_dim must be even (img/txt concat), got {input_dim}")
        self.modal_dim = input_dim // 2
        self.num_domains = num_domains
        self.mask_ratio = mask_ratio
        self.recon_weight = recon_weight
        self.align_weight = align_weight
        self.align_robust_weight = float(align_robust_weight)
        self.domain_weight = domain_weight
        self.coral_weight = float(coral_weight)
        self.node_contrastive_weight = node_contrastive_weight
        self.domain_temperature = domain_temperature
        self.domain_label_smoothing = float(domain_label_smoothing)
        self.edge_drop_ratio = float(edge_drop_ratio)
        self.feat_noise_std = float(feat_noise_std)

        self.align_img = nn.Linear(self.modal_dim, hidden_dim)
        self.align_txt = nn.Linear(self.modal_dim, hidden_dim)
        self.Tmodal = nn.Parameter(torch.randn(hidden_dim))
        self.prompt_dropout = nn.Dropout(p=prompt_dropout)
        self.Tdomain = nn.Embedding(num_domains, hidden_dim)
        self.Tdomain_out = nn.Embedding(num_domains, hidden_dim)
        nn.init.zeros_(self.Tdomain_out.weight)

        self.recon_img = nn.Linear(hidden_dim, hidden_dim)
        self.recon_txt = nn.Linear(hidden_dim, hidden_dim)

        self.gcn = GCNEncoderSparse(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu",
            dropout=0.15,
            use_batch_norm=True,
        )

        self.node_contrastive = (
            NodeNodeContrastiveLoss(temperature=node_contrastive_temperature)
            if node_contrastive_weight > 0
            else None
        )

        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tuples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        x_img, x_txt = _split_modalities(x, self.modal_dim)
        if self.training and self.feat_noise_std > 0:
            x_img = x_img + self.feat_noise_std * torch.randn_like(x_img)
            x_txt = x_txt + self.feat_noise_std * torch.randn_like(x_txt)

        N = x.size(0)
        mask_nodes_img = torch.rand(N, device=self.device) < self.mask_ratio
        mask_nodes_txt = torch.rand(N, device=self.device) < self.mask_ratio

        x_img_masked = x_img.clone()
        x_txt_masked = x_txt.clone()
        if mask_nodes_img.any():
            x_img_masked[mask_nodes_img] = 0
        if mask_nodes_txt.any():
            x_txt_masked[mask_nodes_txt] = 0

        z_img = self.align_img(x_img_masked)
        z_txt = self.align_txt(x_txt_masked)

        z_img_drop = self.prompt_dropout(z_img)
        z_txt_drop = self.prompt_dropout(z_txt)
        s_img = (z_img_drop * self.Tmodal).sum(dim=-1)
        s_txt = (z_txt_drop * self.Tmodal).sum(dim=-1)
        alpha = F.softmax(torch.stack([s_img, s_txt], dim=1), dim=1)
        z_shared = alpha[:, 0:1] * z_img + alpha[:, 1:2] * z_txt
        x_in = z_shared + self.Tdomain(batch)

        ei = edge_index
        if self.training and self.edge_drop_ratio > 0:
            ei, _ = dropout_edge(ei, p=self.edge_drop_ratio, training=True)

        h_core = self.gcn(x_in, ei)
        h_core = h_core + GCN_INPUT_RESIDUAL * x_in
        h = h_core + self.Tdomain_out(batch)

        z_img_target = self.align_img(x_img)
        z_txt_target = self.align_txt(x_txt)
        recon_img = self.recon_img(h)
        recon_txt = self.recon_txt(h)
        loss_recon = torch.tensor(0.0, device=self.device)
        if mask_nodes_img.any():
            loss_recon = loss_recon + F.mse_loss(
                recon_img[mask_nodes_img], z_img_target[mask_nodes_img]
            )
        if mask_nodes_txt.any():
            loss_recon = loss_recon + F.mse_loss(
                recon_txt[mask_nodes_txt], z_txt_target[mask_nodes_txt]
            )

        z_img_norm = F.normalize(z_img_target, dim=-1)
        z_txt_norm = F.normalize(z_txt_target, dim=-1)
        cos = (z_img_norm * z_txt_norm).sum(dim=-1)
        loss_align = (1.0 - cos).mean()
        loss_align_robust = F.smooth_l1_loss(z_img_norm, z_txt_norm, beta=0.05)

        h_norm = F.normalize(h, dim=-1)
        td_norm = F.normalize(self.Tdomain.weight, dim=-1)
        logits_domain = (h_norm @ td_norm.t()) / self.domain_temperature
        loss_domain = F.cross_entropy(
            logits_domain,
            batch,
            label_smoothing=self.domain_label_smoothing,
        )

        loss_coral = _coral_loss_cross_domain(h, batch, self.num_domains)

        loss = (
            self.recon_weight * loss_recon
            + self.align_weight * loss_align
            + self.align_robust_weight * loss_align_robust
            + self.domain_weight * loss_domain
            + self.coral_weight * loss_coral
        )

        if self.node_contrastive is not None and tuples is not None:
            h_c = F.elu(h)
            loss = loss + self.node_contrastive_weight * self.node_contrastive(h_c, tuples)

        return loss

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        x_img, x_txt = _split_modalities(x, self.modal_dim)
        z_img = self.align_img(x_img)
        z_txt = self.align_txt(x_txt)
        s_img = (z_img * self.Tmodal).sum(dim=-1)
        s_txt = (z_txt * self.Tmodal).sum(dim=-1)
        alpha = F.softmax(torch.stack([s_img, s_txt], dim=1), dim=1)
        z_shared = alpha[:, 0:1] * z_img + alpha[:, 1:2] * z_txt
        dom0 = torch.zeros(x.size(0), device=self.device, dtype=torch.long)
        x_in = z_shared + self.Tdomain(dom0)
        h_core = self.gcn(x_in, edge_index)
        h_core = h_core + GCN_INPUT_RESIDUAL * x_in
        return h_core + self.Tdomain_out(dom0)

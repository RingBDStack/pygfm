from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.private.utlis.domain_alignment import TaskAdapter
from pygfm.public.model_bases import GFMPrePromptModelBase

from .preprompt import BimGFMPrePromptModel, _split_modalities


@dataclass(frozen=True)
class BimGFMAblationConfig:
    """
    Ablation config aligned to your BIM wording, but implemented on top of existing bim_gfm.

    Modes:
      - full: dual token + adapter + losses
      - no_attr: w/o Attribute Token (attribute view falls back to structure view)
      - no_struct: w/o Structure Token (structure view removed, falls back to attribute view)
      - no_adapter: w/o TaskAdapter (no bypass module, optionally allow full finetune outside)
    """

    ablation: Literal["full", "no_attr", "no_struct", "no_adapter"] = "full"

    # loss toggles
    enable_decoupling_loss: bool = True
    enable_consistency_loss: bool = True

    # weights
    decoupling_weight: float = 1.0
    consistency_weight: float = 1.0

    def resolved(self) -> "BimGFMAblationConfig":
        dec = self.enable_decoupling_loss
        cons = self.enable_consistency_loss
        if self.ablation == "no_struct":
            dec = False
        if self.ablation == "no_adapter":
            cons = False
        return BimGFMAblationConfig(
            ablation=self.ablation,
            enable_decoupling_loss=dec,
            enable_consistency_loss=cons,
            decoupling_weight=self.decoupling_weight,
            consistency_weight=self.consistency_weight,
        )


class BimGFMAblationPrePromptModel(GFMPrePromptModelBase):
    """
    Wrapper model for ablation experiments.

    It reuses the full bim_gfm preprompt (mask/recon/align/domain/node-contrastive),
    and optionally adds:
      - Decoupling_Loss: encourages img/txt branches to be different (cosine similarity penalty)
      - Consistency_Loss: TaskAdapter output close to backbone output (MSE)
    """

    gfm_family: ClassVar[str] = "bim_gfm"

    def __init__(
        self,
        base: BimGFMPrePromptModel,
        cfg: BimGFMAblationConfig,
        *,
        device: torch.device | None = None,
    ):
        super().__init__(device=device or base.device)
        self.base = base
        self.cfg = cfg.resolved()

        # TaskAdapter is a bypass module on top of backbone node embeddings
        self.task_adapter: Optional[TaskAdapter]
        if self.cfg.ablation == "no_adapter":
            self.task_adapter = None
        else:
            self.task_adapter = TaskAdapter(
                input_dim=base.gcn.layers[-1].out_channels if hasattr(base.gcn.layers[-1], "out_channels") else base.Tmodal.numel(),  # best-effort
                task_type="injection",
                output_dim=base.Tmodal.numel(),
            )

        self.to(self.device)

    def _ablate_modalities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attribute/structure token ablation at the *input* level:
        - no_attr: img view replaced by txt view
        - no_struct: txt view replaced by img view
        """
        modal_dim = self.base.modal_dim
        x_img, x_txt = _split_modalities(x, modal_dim)
        if self.cfg.ablation == "no_attr":
            x_img = x_txt
        elif self.cfg.ablation == "no_struct":
            x_txt = x_img
        return torch.cat([x_img, x_txt], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tuples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns scalar loss (compatible with existing pretrain scripts),
        adding ablation-specific losses when enabled.
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        x = self._ablate_modalities(x)

        # Re-implement the base forward but keep intermediate tensors for extra losses.
        # We reuse base modules directly (align layers, tokens, gcn, recon heads).
        x_img, x_txt = _split_modalities(x, self.base.modal_dim)

        N = x.size(0)
        mask_nodes_img = torch.rand(N, device=self.device) < self.base.mask_ratio
        mask_nodes_txt = torch.rand(N, device=self.device) < self.base.mask_ratio
        x_img_masked = x_img.clone()
        x_txt_masked = x_txt.clone()
        if mask_nodes_img.any():
            x_img_masked[mask_nodes_img] = 0
        if mask_nodes_txt.any():
            x_txt_masked[mask_nodes_txt] = 0

        z_img = self.base.align_img(x_img_masked)
        z_txt = self.base.align_txt(x_txt_masked)

        # Tmodal fuse
        z_img_drop = self.base.prompt_dropout(z_img)
        z_txt_drop = self.base.prompt_dropout(z_txt)
        s_img = (z_img_drop * self.base.Tmodal).sum(dim=-1)
        s_txt = (z_txt_drop * self.base.Tmodal).sum(dim=-1)
        alpha = F.softmax(torch.stack([s_img, s_txt], dim=1), dim=1)
        x_common = alpha[:, 0:1] * z_img + alpha[:, 1:2] * z_txt

        # Tdomain inject
        x_common = x_common + self.base.Tdomain(batch)

        # backbone gnn
        h = self.base.gcn(x_common, edge_index)
        if self.task_adapter is None:
            h_final = h
        else:
            h_final = self.task_adapter(h)

        # base losses
        z_img_target = self.base.align_img(x_img)
        z_txt_target = self.base.align_txt(x_txt)
        recon_img = self.base.recon_img(h_final)
        recon_txt = self.base.recon_txt(h_final)
        loss_recon = torch.tensor(0.0, device=self.device)
        if mask_nodes_img.any():
            loss_recon = loss_recon + F.mse_loss(recon_img[mask_nodes_img], z_img_target[mask_nodes_img])
        if mask_nodes_txt.any():
            loss_recon = loss_recon + F.mse_loss(recon_txt[mask_nodes_txt], z_txt_target[mask_nodes_txt])

        z_img_norm = F.normalize(z_img_target, dim=-1)
        z_txt_norm = F.normalize(z_txt_target, dim=-1)
        cos = (z_img_norm * z_txt_norm).sum(dim=-1)
        loss_align = (1.0 - cos).mean()

        h_norm = F.normalize(h_final, dim=-1)
        td_norm = F.normalize(self.base.Tdomain.weight, dim=-1)
        logits_domain = (h_norm @ td_norm.t()) / self.base.domain_temperature
        loss_domain = F.cross_entropy(logits_domain, batch)

        loss = (
            self.base.recon_weight * loss_recon
            + self.base.align_weight * loss_align
            + self.base.domain_weight * loss_domain
        )

        # optional base node contrastive
        if self.base.node_contrastive is not None and tuples is not None:
            loss = loss + self.base.node_contrastive_weight * self.base.node_contrastive(F.elu(h_final), tuples)

        # Decoupling_Loss (extra): penalize similarity between z_img_target and z_txt_target
        if self.cfg.enable_decoupling_loss and self.cfg.ablation not in ("no_struct",):
            loss_dec = F.cosine_similarity(z_img_target, z_txt_target, dim=-1).mean()
            loss = loss + self.cfg.decoupling_weight * loss_dec

        # Consistency_Loss (extra): adapter output close to backbone output
        if self.cfg.enable_consistency_loss and self.task_adapter is not None:
            loss_cons = F.mse_loss(h_final, h.detach())
            loss = loss + self.cfg.consistency_weight * loss_cons

        return loss


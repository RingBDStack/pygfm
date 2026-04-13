from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import ClassVar, Optional

from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.utils import compute_prototypes

from pygfm.baseline_models.bim_gfm.preprompt import GCN_INPUT_RESIDUAL


def _split_modalities(x: torch.Tensor, modal_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return x[:, :modal_dim], x[:, modal_dim : modal_dim * 2]


class BimGFMDownPromptModel(GFMDownPromptNodeModelBase):
    """Frozen shared GCN; same forward as PrePrompt: +Tdomain -> GCN -> residual -> +Tdomain_out -> prototype matching."""

    gfm_family: ClassVar[str] = "bim_gfm"

    def __init__(
        self,
        gcn: nn.Module,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_domains: int,
        domain_id: int = 0,
        prototype_temp: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        if input_dim % 2 != 0:
            raise ValueError(f"input_dim must be even (img/txt concat), got {input_dim}")
        self.modal_dim = input_dim // 2

        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False

        self.align_img = nn.Linear(self.modal_dim, hidden_dim)
        self.align_txt = nn.Linear(self.modal_dim, hidden_dim)
        self.Tmodal = nn.Parameter(torch.randn(hidden_dim))
        self.Tdomain = nn.Embedding(num_domains, hidden_dim)
        self.Tdomain_out = nn.Embedding(num_domains, hidden_dim)
        nn.init.zeros_(self.Tdomain_out.weight)
        self.domain_id = int(domain_id)
        self.prototype_temp = float(prototype_temp)

        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = int(num_classes)
        self.register_buffer("_prototypes", torch.zeros(self.num_classes, hidden_dim))

        self.to(self.device)

    def _modal_fuse(self, z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        s_img = (z_img * self.Tmodal).sum(dim=-1)
        s_txt = (z_txt * self.Tmodal).sum(dim=-1)
        alpha = F.softmax(torch.stack([s_img, s_txt], dim=1), dim=1)
        return alpha[:, 0:1] * z_img + alpha[:, 1:2] * z_txt

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_idx: torch.Tensor,
        support_labels: torch.Tensor,
        query_idx: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        support_idx = support_idx.to(self.device)
        support_labels = support_labels.to(self.device)

        x_img, x_txt = _split_modalities(x, self.modal_dim)
        z_img = self.align_img(x_img)
        z_txt = self.align_txt(x_txt)
        x_fused = self._modal_fuse(z_img, z_txt)

        domain = torch.full(
            (x_fused.size(0),),
            self.domain_id,
            dtype=torch.long,
            device=self.device,
        )
        x_in = x_fused + self.Tdomain(domain)

        h_core = self.gcn(x_in, edge_index)
        h = h_core + GCN_INPUT_RESIDUAL * x_in + self.Tdomain_out(domain)

        support_emb = h[support_idx]

        if train:
            self._prototypes.copy_(
                compute_prototypes(support_emb, support_labels, self.num_classes).detach()
            )

        query_emb = h[query_idx.to(self.device)] if query_idx is not None else h
        logits = self.head(query_emb, self._prototypes)
        return logits / max(self.prototype_temp, 1e-6)

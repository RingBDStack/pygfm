"""
GraphKeeper-style DownPrompt: frozen pretrained GCN + trainable NodeLevelPrompt + **embedding-space LoRA**
(low-rank correction like original `GCN_LoRA` on representations; PyG uses two Linear layers).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import ClassVar, Optional

from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.utils import compute_prototypes


class EmbeddingLoRA(nn.Module):
    """h' = h + scale * B(A(h)), low-rank adapter in the GraphKeeper expert-branch spirit."""

    def __init__(self, dim: int, rank: int, scale: float = 1.0):
        super().__init__()
        self.lora_a = nn.Linear(dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, dim, bias=False)
        self.scale = scale
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * self.lora_b(self.lora_a(h))


class GraphKeeperDownPromptModel(GFMDownPromptNodeModelBase):
    """
    prefeature(x) -> frozen GCN -> +LoRA(h) -> prototype matching (same interface as MDGPT DownPrompt scripts).
    """

    gfm_family: ClassVar[str] = "graphkeeper"

    def __init__(
        self,
        gcn: nn.Module,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        prompt_mode: str = "mul",
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False
        self.prefeature = NodeLevelPrompt(input_dim, mode=prompt_mode)
        self.lora = EmbeddingLoRA(hidden_dim, lora_rank, scale=lora_scale)
        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = num_classes
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_prompted = self.prefeature(x)
        h = self.gcn(x_prompted, edge_index)
        return h + self.lora(h)

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

        h = self._encode(x, edge_index)
        support_emb = h[support_idx]
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_emb, support_labels, self.num_classes).detach()
            )
        if query_idx is not None:
            query_idx = query_idx.to(self.device)
            query_emb = h[query_idx]
        else:
            query_emb = h
        return self.head(query_emb, self._prototypes)

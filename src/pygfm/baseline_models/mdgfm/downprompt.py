"""
MDGFM DownPrompt: composed pretext + open pretext -> sumtext -> frozen GCN -> prototype matching.
Reuses GFM: NodeLevelPrompt, ComposedNodeLevelPrompt, GCNEncoderSparse, TaskHead(matching), compute_prototypes.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

from typing import ClassVar, List, Literal, Optional

import torch
import torch.nn as nn

from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt, ComposedNodeLevelPrompt
from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.utils import compute_prototypes


class MDGFMDownPromptModel(GFMDownPromptNodeModelBase):
    """
    MDGFM DownPrompt: (composed_pretext + beta*open_pretext) -> ReLU -> sumtext -> frozen GCN -> matching.
    """

    gfm_family: ClassVar[str] = "mdgfm"

    def __init__(
        self,
        gcn: GCNEncoderSparse,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        pretext_weights: List[torch.Tensor],
        sumtext_weight: torch.Tensor,
        beta: float = 1.0,
        prompt_mode: Literal["add", "mul"] = "mul",
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False

        self.composed_pretext = ComposedNodeLevelPrompt(pretext_weights, mode=prompt_mode)
        self.open_pretext = NodeLevelPrompt(input_dim, mode=prompt_mode)
        self.beta = beta
        # sumtext: single weight (1, input_dim)
        self.sumtext = NodeLevelPrompt(input_dim, mode=prompt_mode)
        self.sumtext.weight.data.copy_(sumtext_weight)

        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = num_classes
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

    def _embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        seq_fea = self.composed_pretext(x) + self.beta * self.open_pretext(x)
        seq_fea = torch.relu(seq_fea)
        seq_fea = self.sumtext(seq_fea)
        return self.gcn(seq_fea, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_idx: torch.Tensor,
        support_labels: torch.Tensor,
        query_idx: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        h = self._embed(x, edge_index)
        support_emb = h[support_idx]
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_emb, support_labels, self.num_classes).detach()
            )
        query_emb = h[query_idx] if query_idx is not None else h
        return self.head(query_emb, self._prototypes)

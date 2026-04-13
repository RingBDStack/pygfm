"""
SAMGPT DownPrompt for few-shot graph classification.
Same as node DownPrompt but graph-level: scatter_mean over nodes -> prototypes -> matching.
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

from pygfm.public.model_bases import GFMDownPromptGraphModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt, ComposedNodeLevelPrompt
from pygfm.private.core.gnn_encoder import GCNEncoderSparseWithPrompts
from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.utils import compute_prototypes
from pygfm.baseline_models.samgpt.downprompt import _CombinedLayerPrompt


def _scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    try:
        import torch_scatter
        return torch_scatter.scatter_mean(src, index, dim=0, dim_size=dim_size)
    except ImportError:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        index_exp = index.unsqueeze(1).expand(-1, src.size(1))
        out.scatter_add_(0, index_exp, src)
        cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
        return out / cnt.clamp(min=1).unsqueeze(1)


class SAMGPTDownPromptGraphModel(GFMDownPromptGraphModelBase):
    """SAMGPT DownPrompt for few-shot graph classification; support/query are subgraphs."""

    gfm_family: ClassVar[str] = "samgpt"

    def __init__(
        self,
        gcn: GCNEncoderSparseWithPrompts,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        fea_pretext_weights: List[torch.Tensor],
        str_pretext_weights: List[List[torch.Tensor]],
        combines: List[float],
        prompt_mode: Literal["add", "mul"] = "mul",
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        alpha = combines[0]
        beta = combines[1] if len(combines) > 1 else 1.0
        self.alpha = alpha
        self.beta = beta

        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False

        self.composed_fea = ComposedNodeLevelPrompt(fea_pretext_weights, mode=prompt_mode)
        self.open_fea = NodeLevelPrompt(input_dim, mode=prompt_mode)
        self.composed_str = nn.ModuleList([
            ComposedNodeLevelPrompt([str_pretext_weights[d][i] for d in range(len(str_pretext_weights))], mode=prompt_mode)
            for i in range(num_layers)
        ])
        self.open_str = nn.ModuleList([
            NodeLevelPrompt(hidden_dim, mode=prompt_mode) for _ in range(num_layers)
        ])
        self.combined_str_layers = nn.ModuleList([
            _CombinedLayerPrompt(self.composed_str[i], self.open_str[i], beta)
            for i in range(num_layers)
        ])

        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = num_classes
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

    def _embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        seq_fea = self.composed_fea(x) + self.beta * self.open_fea(x)
        embed_fea = self.gcn(seq_fea, edge_index, batch=None, structure_prompt_layers_per_domain=None, structure_prompt_layers=None)
        embed_str = self.gcn(
            x,
            edge_index,
            batch=None,
            structure_prompt_layers_per_domain=None,
            structure_prompt_layers=self.combined_str_layers,
        )
        return embed_fea + self.alpha * embed_str

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_idx: torch.Tensor,
        support_batch: torch.Tensor,
        support_labels: torch.Tensor,
        query_idx: Optional[torch.Tensor] = None,
        query_batch: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        h = self._embed(x, edge_index)
        support_node_emb = h[support_idx]
        support_graph_emb = _scatter_mean(
            support_node_emb,
            support_batch,
            dim_size=int(support_batch.max().item()) + 1,
        )
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_graph_emb, support_labels, self.num_classes).detach()
            )
        if query_idx is not None and query_batch is not None:
            query_node_emb = h[query_idx]
            query_graph_emb = _scatter_mean(
                query_node_emb,
                query_batch,
                dim_size=int(query_batch.max().item()) + 1,
            )
        else:
            query_graph_emb = support_graph_emb
        logits = self.head(query_graph_emb, self._prototypes)
        return logits

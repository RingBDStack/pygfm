"""
GCoT DownPrompt for few-shot graph classification: same think loop, then scatter_mean -> prototype matching.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

from typing import ClassVar, Optional

import torch
import torch.nn as nn

from pygfm.public.model_bases import GFMDownPromptGraphModelBase
from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.utils import compute_prototypes
from pygfm.baseline_models.gcot.downprompt import ConditionNet


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> torch.Tensor:
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


class GCoTDownPromptGraphModel(GFMDownPromptGraphModelBase):
    """GCoT DownPrompt for few-shot graph classification."""

    gfm_family: ClassVar[str] = "gcot"

    def __init__(
        self,
        gcn: GCNEncoderSparse,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        think_layer_num: int = 1,
        condition_layer_num: int = 1,
        condition_hid_dim: int = 128,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False
        self.condition_layers = nn.ModuleList([
            ConditionNet(hidden_dim, condition_hid_dim, input_dim, num_layers=condition_layer_num)
            for _ in range(think_layer_num)
        ])
        self.res_weights = nn.Parameter(torch.ones(think_layer_num))
        self.gcn_weight1 = nn.Parameter(torch.tensor(1.0))
        self.gcn_weight2 = nn.Parameter(torch.tensor(0.0))
        self.gcn_weight3 = nn.Parameter(torch.tensor(0.0))
        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = num_classes
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

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
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        support_idx = support_idx.to(self.device)
        support_batch = support_batch.to(self.device)
        support_labels = support_labels.to(self.device)

        origin_x = x
        num_layers = len(self.gcn.layers)
        for i, condition_net in enumerate(self.condition_layers):
            hs = []
            h = self.gcn.layers[0](x, edge_index)
            hs.append(h)
            for j in range(1, num_layers):
                h = self.gcn.layers[j](h, edge_index) + h
                hs.append(h)
            w1, w2, w3 = self.gcn_weight1, self.gcn_weight2, self.gcn_weight3
            if num_layers == 1:
                embed = w1 * hs[0]
            elif num_layers == 2:
                embed = w1 * hs[0] + w2 * hs[1]
            else:
                embed = w1 * hs[0] + w2 * hs[1] + w3 * hs[2]
            prompt = condition_net(embed)
            x = origin_x + self.res_weights[i] * (prompt * origin_x)

        h = self.gcn(x, edge_index)
        support_node_emb = h[support_idx]
        support_graph_emb = _scatter_mean(
            support_node_emb, support_batch, dim_size=int(support_batch.max().item()) + 1
        )
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_graph_emb, support_labels, self.num_classes).detach()
            )
        if query_idx is not None and query_batch is not None:
            query_graph_emb = _scatter_mean(
                h[query_idx], query_batch, dim_size=int(query_batch.max().item()) + 1
            )
        else:
            query_graph_emb = support_graph_emb
        return self.head(query_graph_emb, self._prototypes)

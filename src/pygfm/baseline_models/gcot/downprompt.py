"""
GCoT DownPrompt: Chain-of-Thought conditioning — ConditionNet on GCN layer outputs, prompt modulates input, then GCN -> prototype matching.
Reuses GFM: GCNEncoderSparse (.layers), TaskHead(matching), compute_prototypes.
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
import torch.nn.functional as F

from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.utils import compute_prototypes


class ConditionNet(nn.Module):
    """MLP: hidden_dim -> condition_hid -> input_dim (prompt to modulate features)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.input_fc(x))
        for layer in self.hidden_fc:
            x = F.elu(layer(x))
            x = self.dropout(x)
        return self.output_fc(x)


class GCoTDownPromptModel(GFMDownPromptNodeModelBase):
    """
    GCoT DownPrompt: for each think layer, get GCN layer-wise outputs (with residual),
    ConditionNet(weighted embed) -> prompt, then x = origin_x + res_weight * (prompt * origin_x);
    finally embed = gcn(x), prototype matching.
    """

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
        self.think_layer_num = think_layer_num
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

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

        origin_x = x
        num_layers = len(self.gcn.layers)
        for i, condition_net in enumerate(self.condition_layers):
            # GCN layer-wise with residual (match original GCoT)
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
        support_emb = h[support_idx]
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_emb, support_labels, self.num_classes).detach()
            )
        query_emb = h[query_idx] if query_idx is not None else h
        return self.head(query_emb, self._prototypes)

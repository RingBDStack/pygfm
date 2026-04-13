"""GraphKeeper graph-level DownPrompt: subgraph scatter + LoRA on node embeddings (MDGPT DownPromptGraph API)."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

import torch
import torch.nn as nn
from typing import ClassVar, Optional

from pygfm.public.model_bases import GFMDownPromptGraphModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.utils import compute_prototypes

from pygfm.baseline_models.graphkeeper.downprompt import EmbeddingLoRA


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


class GraphKeeperDownPromptGraphModel(GFMDownPromptGraphModelBase):
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
        h = self.gcn(self.prefeature(x), edge_index)
        return h + self.lora(h)

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

        h = self._encode(x, edge_index)
        support_node_emb = h[support_idx]
        support_graph_emb = _scatter_mean(
            support_node_emb,
            support_batch,
            dim_size=int(support_batch.max().item()) + 1,
        )
        if train:
            self._prototypes.copy_(
                compute_prototypes(
                    support_graph_emb, support_labels, self.num_classes
                ).detach()
            )
        if query_idx is not None and query_batch is not None:
            query_idx = query_idx.to(self.device)
            query_batch = query_batch.to(self.device)
            query_graph_emb = _scatter_mean(
                h[query_idx],
                query_batch,
                dim_size=int(query_batch.max().item()) + 1,
            )
        else:
            query_graph_emb = support_graph_emb
        return self.head(query_graph_emb, self._prototypes)

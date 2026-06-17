"""
MDGFM DownPrompt for few-shot graph classification: same node-level pipeline +
graph-level scatter_mean for prototypes and query aggregation.

Faithful port matching MDGFM-main/downprompt.py + graph scatter pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

from typing import ClassVar, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.public.model_bases import GFMDownPromptGraphModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from .preprompt import ATTLearner, CombinePrompt
from .downprompt import (
    WeightedPrompt,
    ComposedToken,
    DownstreamPrompt,
    PreFeaturePrompt,
    averageemb,
)


def _scatter_mean(
    src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None
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


class MDGFMDownPromptGraphModel(GFMDownPromptGraphModelBase):
    """MDGFM DownPrompt for few-shot graph classification."""

    gfm_family: ClassVar[str] = "mdgfm"

    def __init__(
        self,
        gcn: nn.Module,
        all_weights: Dict[str, List[torch.Tensor]],
        ft_in: int,
        nb_classes: int,
        feature_dim: int,
        prompt_mode: Literal["add", "mul"] = "mul",
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.gcn = gcn
        self.nb_classes = nb_classes
        self.feature_dim = feature_dim

        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False

        token_weights = all_weights["texttoken"]
        pretext_weights = all_weights["pretext"]
        sumtext_w = all_weights["sumtext"]
        balance_weights = all_weights["balancetoken"]

        self.downprompt = DownstreamPrompt(ft_in)
        self.composedprompt = ComposedToken(token_weights, prompt_mode=prompt_mode)
        self.prefeature = PreFeaturePrompt(
            pretext_weights, sumtext_w, dim=feature_dim, prompt_mode=prompt_mode
        )
        self.combineprompt1 = CombinePrompt()
        self.combineprompt2 = CombinePrompt()
        self.balancedprompt = NodeLevelPrompt(2 * feature_dim, mode=prompt_mode)

        self.learner = ATTLearner(
            nlayers=2, isize=2 * feature_dim, i=6,
            dropedge_rate=0.5, sparse=True, act="relu",
        )

        self.leakyrelu = nn.ELU()
        self.register_buffer("one", torch.ones(1, ft_in))
        self.register_buffer("ave", torch.zeros(nb_classes, ft_in))

        self.to(self.device)

    def forward(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        sparse: bool,
        support_idx: torch.Tensor,
        support_batch: torch.Tensor,
        support_labels: torch.Tensor,
        downk: int = 30,
        query_idx: Optional[torch.Tensor] = None,
        query_batch: Optional[torch.Tensor] = None,
        train: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            features: [N, feature_dim]
            adj: sparse COO adjacency
            sparse: always True
            support_idx: support node indices
            support_batch: graph batch IDs for support nodes
            support_labels: per-graph support labels
            downk: k for k-NN
            query_idx: query node indices
            query_batch: graph batch IDs for query nodes
            train: 1 to update prototypes
        Returns:
            classification logits [num_query_graphs, nb_classes]
        """
        features = features.to(self.device)
        adj = adj.to(self.device)

        # Step 1: prefeatureprompt
        features1 = self.prefeature(features)

        # Step 2: feature-neighbor concat + balance token
        reseq1 = torch.sparse.mm(adj, features1)
        reseq1 = torch.cat((features1, reseq1), dim=1)
        reseq111 = self.balancedprompt(reseq1)

        # Step 3: ATT_learner k-NN refined adj
        adj1 = self.learner.graph_process(downk, reseq111)

        # Step 4: alpha blending
        alpha = nn.Parameter(torch.tensor(0.5, device=self.device))
        adj_dense = adj.to_dense() if adj.is_sparse else adj
        adjtot = alpha * adj_dense + (1 - alpha) * adj1

        # Step 5: GCN on blended adjacency
        embeds1 = self.gcn(features1, adjtot, sparse=False, LP=False).squeeze()

        # Step 6: support graph embeddings (scatter_mean over nodes)
        support_node_emb = embeds1[support_idx]
        support_graph_emb = _scatter_mean(
            support_node_emb, support_batch,
            dim_size=int(support_batch.max().item()) + 1,
        )

        # Step 7: prototypes
        if train == 1:
            self.ave = averageemb(support_labels, support_graph_emb, self.nb_classes)

        # Step 8: query graph embeddings
        if query_idx is not None and query_batch is not None:
            query_node_emb = embeds1[query_idx]
            query_graph_emb = _scatter_mean(
                query_node_emb, query_batch,
                dim_size=int(query_batch.max().item()) + 1,
            )
        else:
            query_graph_emb = support_graph_emb

        # Step 9: cosine similarity + softmax
        rawret = torch.cat((query_graph_emb, self.ave), dim=0)
        rawret = torch.cosine_similarity(
            rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1
        )
        ret = rawret[: query_graph_emb.shape[0], query_graph_emb.shape[0]:]
        ret = F.softmax(ret, dim=1)

        return ret


__all__ = ["MDGFMDownPromptGraphModel", "_scatter_mean"]

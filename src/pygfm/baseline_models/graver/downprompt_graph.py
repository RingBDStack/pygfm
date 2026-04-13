"""
GRAVER graph-level DownPrompt: extends node-level DownPrompt with scatter_mean
pooling of node embeddings per graph, then prototype cosine classification.
For few-shot graph classification (see scripts/graver/finetune_graph.py).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .downprompt import (
    GRAVERDownPromptModel,
    GraphonGenerator,
    inject_graphs_to_target,
)
from ...public.utils import compute_prototypes


def _scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Group-mean pool src by index (fallback when torch_scatter is missing)."""
    try:
        import torch_scatter
        return torch_scatter.scatter_mean(src, index, dim=0, dim_size=dim_size)
    except ImportError:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        idx_exp = index.unsqueeze(1).expand(-1, src.size(1))
        out.scatter_add_(0, idx_exp, src)
        cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
        return out / cnt.clamp(min=1).unsqueeze(1)


class GRAVERDownPromptGraphModel(GRAVERDownPromptModel):
    """
    GRAVER graph-level few-shot classifier.

    Inherits node-level DownPrompt params and prompting; forward_graph adds scatter_mean graph pooling.
    """

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        idx: torch.Tensor,
        batch: torch.Tensor,
        graphon_list: List[List[torch.Tensor]],
        labels: torch.Tensor | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [N, input_dim] full-graph node features
        :param edge_index: [2, E] full-graph edges
        :param idx: [M] node indices (one per subgraph / graph instance)
        :param batch: [M] graph id for each idx node
        :param graphon_list: [num_sources][num_labels_i] graphon tensors
        :param labels: [G] graph-level labels when train=True
        :param train: if True, refresh graph-level prototypes
        :return: (probs [G, C], entropy [G])
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        idx = idx.to(self.device)
        batch = batch.to(self.device)

        x_prompted, graphon, token = self._prompt_features(x, graphon_list)

        gen = GraphonGenerator(graphon, self.gen_num_nodes, token)
        idx_list = idx.tolist()
        with torch.no_grad():
            graphs = [gen.generate() for _ in range(len(idx_list))]
        gen_x = [g[0] for g in graphs]
        gen_ei = [g[1] for g in graphs]
        x_exp, ei_exp = inject_graphs_to_target(gen_x, gen_ei, x_prompted, edge_index, idx_list)

        embeds = self.disen_gcn(x_exp, ei_exp)
        emb_at_idx = embeds[idx]

        num_graphs = int(batch.max().item()) + 1
        graph_emb = _scatter_mean(emb_at_idx, batch, dim_size=num_graphs)

        if train and labels is not None:
            labels = labels.to(self.device)
            self.prototypes.copy_(
                compute_prototypes(graph_emb.detach(), labels, self.num_classes)
            )

        all_emb = torch.cat([graph_emb, self.prototypes], dim=0)
        cos_sim = F.cosine_similarity(all_emb.unsqueeze(1), all_emb.unsqueeze(0), dim=-1)
        G = graph_emb.size(0)
        logits = cos_sim[:G, G:]
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return probs, entropy

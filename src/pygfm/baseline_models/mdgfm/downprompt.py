"""
MDGFM DownPrompt: composed pretext + open pretext → ReLU → sumtext →
ATT_learner refined adj → alpha-blended adjacency → GCN → prototype matching.

Faithful port of the original MDGFM-main/downprompt.py architecture.
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

from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from .preprompt import ATTLearner, CombinePrompt


# ---------------------------------------------------------------------------
# weighted_prompt: softmax-weighted combination of concatenated tokens
# ---------------------------------------------------------------------------

class WeightedPrompt(nn.Module):
    """Softmax-weighted sum of N input tokens. Original: weighted_prompt."""

    def __init__(self, weightednum: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return torch.mm(self.weight, graph_embedding)


# ---------------------------------------------------------------------------
# composedtoken: weighted mix of pretrained text tokens, then add/mul on input
# ---------------------------------------------------------------------------

class ComposedToken(nn.Module):
    """Original: composedtoken — weighted mix of text tokens + add/mul on x."""

    def __init__(
        self,
        texttoken_weights: List[torch.Tensor],
        prompt_mode: Literal["add", "mul"] = "mul",
    ):
        super().__init__()
        self.texttoken = torch.cat(
            [w.detach().clone() for w in texttoken_weights], dim=0
        )  # [K, dim]
        self.prompt = WeightedPrompt(len(texttoken_weights))
        self.mode = prompt_mode

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        combined = self.prompt(self.texttoken)  # [1, dim]
        if self.mode == "add":
            return combined.repeat(seq.shape[0], 1) + seq
        return combined * seq


# ---------------------------------------------------------------------------
# downstreamprompt: simple element-wise scaling
# ---------------------------------------------------------------------------

class DownstreamPrompt(nn.Module):
    """Original: downstreamprompt — element-wise weight * x."""

    def __init__(self, hid_units: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.weight * graph_embedding


# ---------------------------------------------------------------------------
# prefeatureprompt: composed token + open token → combineprompt
# ---------------------------------------------------------------------------

class PreFeaturePrompt(nn.Module):
    """
    Original: prefeatureprompt.

    composedtoken(pretext1-5) → ReLU → sumtext * result
    + downstreamprompt(open)
    → combineprompt → output
    """

    def __init__(
        self,
        pretext_weights: List[torch.Tensor],
        sumtext_weight: torch.Tensor,
        dim: int,
        prompt_mode: Literal["add", "mul"] = "mul",
    ):
        super().__init__()
        self.precomposedfeature = ComposedToken(pretext_weights, prompt_mode=prompt_mode)
        self.preopenfeature = DownstreamPrompt(dim)
        self.sumtext_weight = nn.Parameter(
            sumtext_weight.detach().clone(), requires_grad=False
        )
        self.combineprompt = CombinePrompt()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq1 = self.precomposedfeature(seq)
        seq1 = F.relu(seq1)
        seq3 = self.sumtext_weight * seq1
        seq2 = self.preopenfeature(seq)
        ret = self.combineprompt(seq3, seq2)
        return ret


# ---------------------------------------------------------------------------
# averageemb: scatter-mean for prototype computation
# ---------------------------------------------------------------------------

def averageemb(
    labels: torch.Tensor, rawret: torch.Tensor, nb_class: int
) -> torch.Tensor:
    """Per-class mean embeddings (prototypes)."""
    try:
        import torch_scatter
        return torch_scatter.scatter(
            src=rawret, index=labels, dim=0, reduce="mean"
        )
    except ImportError:
        out = torch.zeros(nb_class, rawret.size(1), device=rawret.device, dtype=rawret.dtype)
        ones = torch.ones_like(labels, dtype=rawret.dtype)
        index_exp = labels.unsqueeze(1).expand(-1, rawret.size(1))
        out.scatter_add_(0, index_exp, rawret)
        cnt = torch.zeros(nb_class, device=rawret.device, dtype=rawret.dtype)
        cnt.scatter_add_(0, labels, ones)
        return out / cnt.clamp(min=1).unsqueeze(1)


# ---------------------------------------------------------------------------
# MDGFMDownPromptModel: few-shot node classification
# ---------------------------------------------------------------------------

class MDGFMDownPromptModel(GFMDownPromptNodeModelBase):
    """
    MDGFM DownPrompt for few-shot node classification.

    Forward: prefeatureprompt → balance token → ATT_learner k-NN →
    alpha-blended adjacency → GCN → prototype cosine similarity → softmax.
    """

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
        """
        Args:
            gcn: frozen GcnLayers from PrePrompt
            all_weights: dict with keys 'texttoken', 'pretext', 'sumtext', 'balancetoken'
            ft_in: hidden dim (256)
            nb_classes: number of classes
            feature_dim: input feature dim (50)
            prompt_mode: 'add' or 'mul'
        """
        super().__init__(device=device)
        self.gcn = gcn
        self.nb_classes = nb_classes
        self.feature_dim = feature_dim

        # Freeze pretrained GCN
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

        # isize=2*feature_dim to match balance token output
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
        gcn: Optional[nn.Module] = None,
        idx: Optional[torch.Tensor] = None,
        seq: Optional[torch.Tensor] = None,
        downk: int = 30,
        labels: Optional[torch.Tensor] = None,
        train: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            features: [N, feature_dim] raw PCA features
            adj: sparse COO adjacency
            sparse: always True
            gcn: unused (kept for GFM compatibility), uses self.gcn
            idx: support node indices
            seq: pretrained embeddings at support nodes [K, hidden_dim]
            downk: k for k-NN (30 for Planetoid, 15 for others)
            labels: support labels (for prototype update when train=1)
            train: 1 to update prototypes, 0 to use existing
        Returns:
            classification logits [K, nb_classes]
        """
        features = features.to(self.device)
        adj = adj.to(self.device)

        # Step 1: prefeatureprompt
        features1 = self.prefeature(features)

        # Step 2: feature-neighbor concatenation + balance token
        reseq1 = torch.sparse.mm(adj, features1)
        reseq1 = torch.cat((features1, reseq1), dim=1)
        reseq111 = self.balancedprompt(reseq1)

        # Step 3: ATT_learner k-NN refined adjacency
        adj1 = self.learner.graph_process(downk, reseq111)

        # Step 4: alpha blending (trainable fusion)
        alpha = nn.Parameter(torch.tensor(0.5, device=self.device))
        adj_dense = adj.to_dense() if adj.is_sparse else adj
        adjtot = alpha * adj_dense + (1 - alpha) * adj1

        # Step 5: GCN on blended adjacency
        embeds1 = self.gcn(features1, adjtot, sparse=False, LP=False).squeeze()

        # Step 6: support embeddings
        pretrain_embs1 = embeds1[idx]
        rawret = pretrain_embs1

        # Step 7: prototype update (if training)
        if train == 1 and labels is not None:
            self.ave = averageemb(labels, rawret, self.nb_classes)

        # Step 8: cosine similarity + softmax classification
        rawret = torch.cat((rawret, self.ave), dim=0)
        rawret = torch.cosine_similarity(
            rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1
        )
        ret = rawret[: seq.shape[0], seq.shape[0]:]
        ret = F.softmax(ret, dim=1)

        return ret


__all__ = [
    "WeightedPrompt",
    "ComposedToken",
    "DownstreamPrompt",
    "PreFeaturePrompt",
    "averageemb",
    "MDGFMDownPromptModel",
]

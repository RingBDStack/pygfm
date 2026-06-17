"""
MDGFM PrePrompt: per-domain pretext + shared sumtext → balance token +
ATT_learner refined adjacency → GCN (LP mode) → Calbound LP loss.

Faithful port of the original MDGFM-main/preprompt.py architecture.

Reuses GFM NodeLevelPrompt (identical to original textprompt).
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

from pygfm.public.model_bases import GFMPrePromptModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from ._gcn import GcnLayers
from ._lp import Lp
from ._attentive import Attentive
from ._tools import knn_fast, apply_non_linearity, calc_lower_bound


# ---------------------------------------------------------------------------
# ATT_learner: attention-based graph structure learner
# ---------------------------------------------------------------------------

class ATTLearner(nn.Module):
    """
    Attention-based k-NN graph learner.

    Original name: ATT_learner (preprompt.py:15).

    graph_process() builds a learned adjacency via knn_fast +
    non-linearity + dropout + symmetrization.

    Note: internal_forward / forward are provided for completeness
    but graph_process() is the method actually used in MDGFM.
    """

    def __init__(
        self,
        nlayers: int,
        isize: int,
        i: float,
        dropedge_rate: float,
        sparse: bool = True,
        act: str = "relu",
    ):
        super().__init__()
        self.non_linearity = "relu"
        self.i = i
        self.sparse = sparse
        self.act = act
        self.dropedge_rate = dropedge_rate

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentive(isize))

    def internal_forward(self, h: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.internal_forward(features)

    def graph_process(self, k: int, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Build a k-NN adjacency from embeddings (sparse path).

        Args:
            k: number of neighbors
            embeddings: [N, D] node embeddings
        Returns:
            learned_adj: [N, N] dense adjacency matrix
        """
        if self.sparse:
            rows, cols, values = knn_fast(embeddings, k, 1000)
            values[torch.isnan(values)] = 0
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            values_ = F.dropout(values_, p=self.dropedge_rate, training=self.training)

            num_nodes = embeddings.shape[0]
            learned_adj = torch.zeros((num_nodes, num_nodes), device=embeddings.device)
            learned_adj[rows_, cols_] = values_

            return learned_adj
        else:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = torch.mm(embeddings, embeddings.t())
            similarities = _top_k(similarities, k + 1)
            similarities = (similarities + similarities.t()) / 2
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            learned_adj = _normalize(similarities, "sym")
            learned_adj = F.dropout(learned_adj, p=self.dropedge_rate, training=self.training)
            return learned_adj


def _top_k(raw_graph: torch.Tensor, K: int) -> torch.Tensor:
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    mask = torch.zeros_like(raw_graph)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.0
    mask.requires_grad = False
    return raw_graph * mask


def _normalize(adj: torch.Tensor, mode: str) -> torch.Tensor:
    EOS_VAL = 1e-10
    if mode == "sym":
        inv_sqrt_degree = 1.0 / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS_VAL)
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1.0 / (adj.sum(dim=1, keepdim=False) + EOS_VAL)
        return inv_degree[:, None] * adj
    else:
        raise ValueError("wrong norm mode")


# ---------------------------------------------------------------------------
# combineprompt: learnable weighted combination of two embeddings
# ---------------------------------------------------------------------------

class CombinePrompt(nn.Module):
    """
    Learnable weighted combination: w[0]*g1 + w[1]*g2 → ELU.

    Initialized with w=[0, 1] so it starts as identity on g2.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.weight.data[0, 0] = 0.0
        self.weight.data[0, 1] = 1.0

    def forward(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        return self.act(self.weight[0, 0] * g1 + self.weight[0, 1] * g2)


# ---------------------------------------------------------------------------
# prompt_pretrain_sample: negative sampling for combined adjacency
# ---------------------------------------------------------------------------

def prompt_pretrain_sample(adj, n: int):
    """
    Sample negative pairs from adjacency for contrastive pretraining.

    Args:
        adj: scipy sparse csr_matrix
        n: number of negative samples per node
    Returns:
        [nodenum, 1+n] indices (first column = positive neighbor, rest = negatives)
    """
    import numpy as np

    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n), dtype=np.int64)
    whole = np.array(range(nodenum))

    for i in range(nodenum):
        nonzero_index_i_row = indices[indptr[i]:indptr[i + 1]]
        zero_index_i_row = np.setdiff1d(whole, nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row) == 0:
            res[i][0] = i
        else:
            res[i][0] = nonzero_index_i_row[0]
        res[i][1:1 + n] = zero_index_i_row[0:n]

    return res.astype(int)


# ---------------------------------------------------------------------------
# MDGFMPrePromptModel
# ---------------------------------------------------------------------------

class MDGFMPrePromptModel(GFMPrePromptModelBase):
    """
    MDGFM PrePrompt: per-domain pretext + shared sumtext + balance token +
    ATT_learner k-NN refined adjacency + GCN LP head + Calbound loss.

    Matches the original PrePrompt class exactly.
    """

    gfm_family: ClassVar[str] = "mdgfm"

    def __init__(
        self,
        n_in: int,
        n_h: int,
        num_domains: int = 5,
        num_layers_num: int = 3,
        dropout: float = 0.1,
        prompt_mode: Literal["add", "mul"] = "mul",
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.n_in = n_in
        self.n_h = n_h
        self.num_domains = num_domains
        self.prompt_mode = prompt_mode

        # LP head (pretraining objective)
        self.lp = Lp(n_in, n_h)

        # Custom GCN stack (with residuals, BN+Dropout only in LP mode)
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)

        # Per-domain pretext tokens (on input features, dim=n_in)
        self.pretexts = nn.ModuleList(
            [NodeLevelPrompt(n_in, mode=prompt_mode) for _ in range(num_domains)]
        )

        # Shared sumtext token
        self.sumtext = NodeLevelPrompt(n_in, mode=prompt_mode)

        # Per-domain text tokens (on hidden dim, saved for downstream)
        self.texttokens = nn.ModuleList(
            [NodeLevelPrompt(n_h, mode=prompt_mode) for _ in range(num_domains)]
        )

        # Per-domain balance tokens (on 2*n_in dim after feature-neighbor concat)
        self.balancetokens = nn.ModuleList(
            [NodeLevelPrompt(2 * n_in, mode=prompt_mode) for _ in range(num_domains)]
        )

        # ATT_learner: shared graph structure learner
        # isize=2*n_in to match balance token output dimension
        self.learner = ATTLearner(
            nlayers=2, isize=2 * n_in, i=6,
            dropedge_rate=0.5, sparse=True, act="relu",
        )

        self.to(self.device)

    def forward(
        self,
        feats: List[torch.Tensor],
        adjs: List[torch.Tensor],
        sparse: bool = True,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Calbound LP loss across all domains.

        Args:
            feats: list of [N_k, n_in] per-domain PCA features
            adjs: list of sparse COO adjacency tensors per domain
            sparse: always True
            target_idx: index of target domain (gets k=15 for k-NN, others k=30)
        Returns:
            scalar LP loss = lploss1 + lploss2
        """
        # Pre-k values: match original MDGFM [30,30,30,15,15,15]
        # Large graphs (Cora/CiteSeer/Pubmed) → k=30, small graphs (Chameleon/Squirrel/Cornell) → k=15
        # The target domain always gets k=15
        nd = len(feats)
        pre_k = [30] * 3 + [15] * max(0, nd - 3)  # first 3 domains k=30, rest k=15
        pre_k = pre_k[:nd]  # truncate to actual domain count
        if target_idx is not None and 0 <= target_idx < nd:
            pre_k[target_idx] = 15

        lp_loss_1 = torch.tensor(0.0, device=self.device)
        lp_loss_2 = torch.tensor(0.0, device=self.device)

        for d in range(len(feats)):
            seq = feats[d].to(self.device)
            adj = adjs[d].to(self.device)
            k_val = pre_k[d]

            # Step 1: per-domain pretext → ReLU
            preseq = self.pretexts[d](seq)
            preseq = F.relu(preseq)

            # Step 2: shared sumtext (no activation)
            preseq = self.sumtext(preseq)

            # Step 3: feature-neighbor concatenation
            neighbor = torch.sparse.mm(adj, preseq)
            reseq = torch.cat((preseq, neighbor), dim=1)  # [N, 2*n_in]

            # Step 4: per-domain balance token
            reseq = self.balancetokens[d](reseq)

            # Step 5: ATT_learner k-NN refined adjacency
            refined_adj = self.learner.graph_process(k_val, reseq)

            # Step 6: LP prelogits (GCN on refined adj) + logits (GCN on original adj)
            prelogits = self.lp(self.gcn, preseq, refined_adj, sparse)
            logits = self.lp(self.gcn, preseq, adj, sparse)

            # Step 7: Calbound lower bound loss
            num_nodes = preseq.shape[0]
            pos_eye = torch.eye(num_nodes, device=self.device)

            lp_loss_1 = lp_loss_1 + calc_lower_bound(prelogits, logits, pos_eye)
            lp_loss_2 = lp_loss_2 + calc_lower_bound(
                prelogits, logits, refined_adj.detach()
            )

        lp_loss = lp_loss_1 + lp_loss_2
        lp_loss.requires_grad_(True)
        return lp_loss

    def embedding(
        self,
        feats: List[torch.Tensor],
        adjs: List[torch.Tensor],
        sparse: bool = True,
    ) -> List[torch.Tensor]:
        """
        Get per-domain embeddings for downstream use.
        Uses pretext + Lp(GCN, feat, orig_adj) — no balance token, no ATT_learner.
        """
        out = []
        for d in range(len(feats)):
            seq = feats[d].to(self.device)
            adj = adjs[d].to(self.device)

            preseq = self.pretexts[d](seq)
            prelogits = self.lp(self.gcn, preseq, adj, sparse)
            out.append(prelogits.detach())

        return out

    def embed(
        self,
        seq: torch.Tensor,
        adj: torch.Tensor,
        sparse: bool = True,
        LP: bool = False,
    ) -> torch.Tensor:
        """
        Single-domain embed for downstream finetuning.
        Uses GCN directly (no pretext, no LP head).
        """
        seq = seq.to(self.device)
        adj = adj.to(self.device)
        h = self.gcn(seq, adj, sparse, LP)
        return h.detach()

    def get_weights(self) -> Dict[str, List[torch.Tensor]]:
        """
        Return all prompt weights needed by DownPrompt.
        """
        return {
            "texttoken": [t.weight.detach().clone() for t in self.texttokens],
            "pretext": [t.weight.detach().clone() for t in self.pretexts],
            "sumtext": self.sumtext.weight.detach().clone(),
            "balancetoken": [t.weight.detach().clone() for t in self.balancetokens],
        }

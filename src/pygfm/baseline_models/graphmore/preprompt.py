"""
GraphMoRE PrePrompt: diverse Riemannian experts + topology-aware gating + embedding
distortion minimization + link prediction pretraining.

GraphMoRE trains multiple Riemannian GNN experts in κ-stereographic spaces at different
curvatures, then uses topology-aware gating from multi-resolution ego-graph sampling to
assign each node an adaptive best embedding space.

Core components:
1. Diverse Riemannian Experts: κ-GCN experts with initial curvatures {-3,-1,0,1,3}
2. Topology-aware Gating: multi-resolution ego-graph sampling → GCN encode → softmax weights
3. Distortion Loss: align embedding distances with shortest-path distances on the graph
4. Fermi-Dirac Decoder: distance-based link prediction decoder

Reference: GraphMoRE: Mitigating Topological Heterogeneity via Mixture of Riemannian Experts (AAAI 2025)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import geoopt
except ImportError:
    raise ImportError(
        "GraphMoRE requires the 'geoopt' library for Riemannian geometry operations. "
        "Install it with: pip install geoopt"
    )

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "GraphMoRE requires 'networkx' for ego-graph sampling and shortest path "
        "computation. Install it with: pip install networkx"
    )

from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


# ---------------------------------------------------------------------------
# κ-stereographic Riemannian GCN building blocks
# ---------------------------------------------------------------------------

class KappaLinear(nn.Module):
    """Möbius linear layer in κ-stereographic space."""

    def __init__(
        self,
        manifold: geoopt.Stereographic,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, project=True)
        if self.use_bias:
            bias = self.manifold.proju(
                self.manifold.origin(self.bias.shape, device=x.device, dtype=x.dtype),
                self.bias,
            )
            kappa_bias = self.manifold.expmap0(bias, project=True)
            res = self.manifold.mobius_add(res, kappa_bias, project=True)
        return res


class KappaGCNConv(MessagePassing):
    """GCN conv in κ-stereographic space: log → tangent message passing → exp to manifold."""

    def __init__(self, k: float, in_dim: int, out_dim: int, learnable: bool = True):
        super().__init__(aggr="add")
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.lin = KappaLinear(manifold=self.manifold, in_dim=in_dim, out_dim=out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        x_tan0 = self.manifold.logmap0(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x_tan0, norm=norm)
        out = self.manifold.expmap0(out, project=True)
        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class RiemannianEncoder(nn.Module):
    """One Riemannian expert: two-layer κ-GCN encoder in a fixed curvature space."""

    def __init__(
        self,
        k: float,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        learnable: bool = True,
    ):
        super().__init__()
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.conv1 = KappaGCNConv(k, in_dim, hidden_dim, learnable=learnable)
        self.conv2 = KappaGCNConv(k, hidden_dim, out_dim, learnable=learnable)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode and stay on the manifold (link prediction)."""
        x = self.manifold.proju(
            self.manifold.origin(x.shape, device=x.device, dtype=x.dtype), x
        )
        x = self.manifold.expmap0(x, project=True)
        h = self.conv1(x, edge_index)
        z = self.conv2(h, edge_index)
        return z

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode then map to tangent space (node classification; near-Euclidean output)."""
        z = self.forward(x, edge_index)
        return self.manifold.logmap0(z)


# ---------------------------------------------------------------------------
# Diverse Riemannian expert ensemble
# ---------------------------------------------------------------------------

class RiemannianExperts(nn.Module):
    """
    K Riemannian experts at different curvatures; outputs are concatenated then LayerNorm.

    By initial curvature:
    - Hyperbolic (κ < 0): tree-like / hierarchical structure
    - Euclidean (κ = 0): general structure
    - Spherical (κ > 0): cyclic / periodic structure
    """

    def __init__(
        self,
        init_curvs: list[float],
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        learnable: bool = True,
    ):
        super().__init__()
        self.experts = nn.ModuleList()
        for curv in init_curvs:
            learn = (curv != 0) and learnable
            self.experts.append(
                RiemannianEncoder(curv, in_dim, hidden_dim, out_dim, learnable=learn)
            )
        self.norm = nn.LayerNorm(len(init_curvs) * out_dim)
        self.num_experts = len(init_curvs)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode with all experts → concat → LayerNorm (link prediction)."""
        embeds = [expert(x, edge_index) for expert in self.experts]
        return self.norm(torch.cat(embeds, dim=-1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Tangent-space encoding from all experts → concat (node classification; no LayerNorm)."""
        embeds = [expert.encode(x, edge_index) for expert in self.experts]
        return torch.cat(embeds, dim=-1)


# ---------------------------------------------------------------------------
# Multi-resolution ego-graph sampler
# ---------------------------------------------------------------------------

class EgoGraphSampler:
    """
    Multi-resolution ego-graph sampling: for each node, extract ego subgraphs at several hop radii
    for local topology encoding in the topology-aware gating network.

    Ego-graph: induced subgraph of k-hop neighbors around a center node.
    Multi-resolution: several k values to capture geometry at different local scales.
    """

    def __init__(self, sample_hops: list[int] | None = None):
        self.sample_hops = sample_hops or [2, 3]

    def sample(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        :return: (features_per_hop, edge_indices_per_hop, batches_per_hop)
            Each list has length len(sample_hops); entries concatenate ego subgraphs over all nodes.
        """
        # Keep the sampled ego-subgraphs on CPU. Concatenating all ego-graphs on GPU can
        # easily blow up memory for medium / large graphs during GraphMoRE finetuning.
        features_cpu = features.detach().cpu()
        G = nx.Graph()
        G.add_nodes_from(range(features_cpu.shape[0]))
        edge_np = edge_index.detach().cpu().numpy()
        G.add_edges_from(zip(edge_np[0].tolist(), edge_np[1].tolist()))

        all_features, all_edges, all_batches = [], [], []
        for hop in self.sample_hops:
            feat, ei, batch = self._sample_ego(G, features_cpu, hop)
            all_features.append(feat)
            all_edges.append(ei)
            all_batches.append(batch)
        return all_features, all_edges, all_batches

    @staticmethod
    def _sample_ego(
        G: nx.Graph, features: torch.Tensor, k_hop: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_features: list[torch.Tensor] = []
        new_edges: list[torch.Tensor] = []
        batches: list[torch.Tensor] = []
        offset = 0

        for node in G.nodes():
            sub = nx.ego_graph(G, node, radius=k_hop)
            sub_nodes = list(sub.nodes())
            new_features.append(features[sub_nodes])

            node_map = {n: idx + offset for idx, n in enumerate(sub_nodes)}
            if sub.number_of_edges() > 0:
                sub_ei = torch.tensor(
                    [[node_map[u], node_map[v]] for u, v in sub.edges()],
                    dtype=torch.long,
                ).t()
                new_edges.append(sub_ei)
            else:
                new_edges.append(torch.zeros(2, 0, dtype=torch.long))

            offset += len(sub_nodes)
            batches.append(torch.full((len(sub_nodes),), node, dtype=torch.long))

        return (
            torch.cat(new_features, dim=0),
            torch.cat(new_edges, dim=1),
            torch.cat(batches, dim=0),
        )


# ---------------------------------------------------------------------------
# Topology-aware gating network
# ---------------------------------------------------------------------------

class TopologyAwareGating(nn.Module):
    """
    Topology-aware gating: GCN over multi-resolution ego subgraphs → pool & concat → softmax expert weights.

    The gate consumes multi-resolution local topology features per node and outputs weights over
    Riemannian experts. Training minimizes embedding distortion so nodes route to the best-matching curvature.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_experts: int,
        num_hops: int = 2,
        chunk_centers: int = 256,
    ):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.pooling = global_mean_pool
        self.classifier = nn.Linear(out_dim * num_hops, num_experts, bias=True)
        self.num_experts = num_experts
        self.chunk_centers = chunk_centers
        self._cached_edge: Optional[torch.Tensor] = None
        self._cached_dis: Optional[torch.Tensor] = None

    def _encode_scale_chunked(
        self,
        feat: torch.Tensor,
        ei: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a concatenated ego-subgraph tensor in small chunks of center nodes.
        This avoids materializing the whole multi-resolution ego-graph collection on GPU.
        """
        device = next(self.parameters()).device
        feat = feat.cpu()
        ei = ei.cpu()
        batch = batch.cpu()

        num_centers = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        out_dim = self.gcn2.out_channels
        scale_out = torch.zeros(num_centers, out_dim, device=device)
        if num_centers == 0:
            return scale_out

        # batch is monotonic because EgoGraphSampler appends ego-graphs node by node.
        start_ptrs = []
        end_ptrs = []
        node_start = 0
        for center in range(num_centers):
            node_end = node_start
            while node_end < batch.numel() and int(batch[node_end].item()) == center:
                node_end += 1
            start_ptrs.append(node_start)
            end_ptrs.append(node_end)
            node_start = node_end

        for center_start in range(0, num_centers, self.chunk_centers):
            center_end = min(num_centers, center_start + self.chunk_centers)
            node_lo = start_ptrs[center_start]
            node_hi = end_ptrs[center_end - 1]
            feat_chunk = feat[node_lo:node_hi].to(device, non_blocking=True)
            batch_chunk = (batch[node_lo:node_hi] - center_start).to(device, non_blocking=True)

            if ei.numel() == 0:
                ei_chunk = torch.zeros(2, 0, dtype=torch.long, device=device)
            else:
                edge_mask = (
                    (ei[0] >= node_lo)
                    & (ei[0] < node_hi)
                    & (ei[1] >= node_lo)
                    & (ei[1] < node_hi)
                )
                ei_chunk = (ei[:, edge_mask] - node_lo).to(device, non_blocking=True)

            h = self.gcn1(feat_chunk, ei_chunk)
            h = self.gcn2(h, ei_chunk)
            pooled = self.pooling(h, batch_chunk)
            scale_out[center_start:center_end] = pooled

            del feat_chunk, batch_chunk, ei_chunk, h, pooled

        return scale_out

    def forward(
        self,
        subgraph_features: list[torch.Tensor],
        subgraph_edge_indices: list[torch.Tensor],
        subgraph_batches: list[torch.Tensor],
        embeddings: Optional[torch.Tensor] = None,
        dis_shortest: Optional[dict] = None,
        embed_dim: Optional[int] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        :return: expert_weights [N, K]; if embeddings is not None, also return distortion loss.
        """
        scale_outputs = []
        for feat, ei, batch in zip(
            subgraph_features, subgraph_edge_indices, subgraph_batches
        ):
            scale_outputs.append(self._encode_scale_chunked(feat, ei, batch))

        x = torch.cat(scale_outputs, dim=-1)
        out = F.softmax(self.classifier(x), dim=-1)

        if embeddings is None:
            return out

        loss_dis = self._compute_distortion(
            out, embeddings, dis_shortest, embed_dim, edge_index
        )
        return out, loss_dis

    def _compute_distortion(
        self,
        expert_weights: torch.Tensor,
        embeddings: torch.Tensor,
        dis_shortest: dict,
        embed_dim: int,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Distortion loss: |d_embed(u,v) / d_graph(u,v) - 1|."""
        needs_update = (
            self._cached_edge is None
            or self._cached_edge.shape != edge_index.shape
            or not torch.equal(self._cached_edge, edge_index)
        )
        if needs_update:
            self._cached_edge = edge_index.detach()
            edges = [
                (edge_index[0][i].item(), edge_index[1][i].item())
                for i in range(edge_index.size(1))
            ]
            self._cached_dis = torch.tensor(
                [dis_shortest.get(e, 1.0) for e in edges],
                device=edge_index.device,
                dtype=torch.float,
            )

        diff = (embeddings[edge_index[0]] - embeddings[edge_index[1]]) ** 2
        num_experts = diff.shape[1] // embed_dim
        diff = diff.reshape(diff.shape[0], num_experts, embed_dim).sum(dim=2)
        weights = F.softmax(
            expert_weights[edge_index[0]] * expert_weights[edge_index[1]], dim=1
        )
        dis = torch.sum(diff * weights, dim=-1)
        distortion = torch.abs(dis / self._cached_dis.clamp(min=1e-8) - 1)
        return distortion.mean()


# ---------------------------------------------------------------------------
# Fermi-Dirac decoder
# ---------------------------------------------------------------------------

class FermiDiracDecoder(nn.Module):
    """Distance-based link decoder: σ((r - dist) / t)."""

    def __init__(self, r: float = 2.0, t: float = 1.0):
        super().__init__()
        self.r = r
        self.t = t

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((self.r - dist) / self.t)


# ---------------------------------------------------------------------------
# Shortest-path distance helpers
# ---------------------------------------------------------------------------

def compute_shortest_path_distances(edge_index: torch.Tensor) -> dict:
    """Shortest-path distances between edge endpoints (for distortion loss)."""
    ei = edge_index.cpu().numpy().astype(int)
    G = nx.Graph()
    for i in range(ei.shape[1]):
        G.add_edge(int(ei[0, i]), int(ei[1, i]))

    d = dict(nx.shortest_path_length(G))
    dis_shortest: dict[tuple[int, int], float] = {}
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        sp = d.get(u, {}).get(v, 1)
        if sp == 0:
            sp = float("inf")
        dis_shortest[(u, v)] = sp
        dis_shortest[(v, u)] = sp
    return dis_shortest


# ---------------------------------------------------------------------------
# GraphMoRE PrePrompt model (link prediction pretraining)
# ---------------------------------------------------------------------------

class GraphMoREPrePromptModel(nn.Module):
    """
    GraphMoRE pretraining: diverse Riemannian experts + topology-aware gating + link prediction.

    Training:
    1. All experts encode nodes → concat + LayerNorm
    2. Gating estimates expert weights from ego topology + distortion loss
    3. Alignment-weighted pairwise distances
    4. Fermi-Dirac decoder → link probs → BCE + λ × distortion

    After training, expert + gating weights initialize downstream tasks.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        init_curvs: list[float] | None = None,
        learnable_curv: bool = True,
        sample_hops: list[int] | None = None,
        r: float = 2.0,
        t: float = 1.0,
        coef_dis: float = 0.1,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.init_curvs = init_curvs or [-3.0, -1.0, 0.0, 1.0, 3.0]
        self.sample_hops = sample_hops or [2, 3]
        self.embed_dim = embed_dim
        self.coef_dis = coef_dis
        self.num_experts = len(self.init_curvs)

        self.experts = RiemannianExperts(
            self.init_curvs, in_dim, hidden_dim, embed_dim, learnable_curv
        )
        self.gating = TopologyAwareGating(
            in_dim, hidden_dim, embed_dim, self.num_experts, len(self.sample_hops)
        )
        self.decoder = FermiDiracDecoder(r, t)
        self.to(self.device)

    def forward(
        self,
        features: torch.Tensor,
        train_edge_index: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        subgraph_features: list[torch.Tensor],
        subgraph_edge_indices: list[torch.Tensor],
        subgraph_batches: list[torch.Tensor],
        dis_shortest: dict,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Link prediction forward.

        :return: (total_loss, auc, ap)
        """
        embeddings = self.experts(features, train_edge_index)
        expert_weights, loss_dis = self.gating(
            subgraph_features,
            subgraph_edge_indices,
            subgraph_batches,
            embeddings,
            dis_shortest,
            self.embed_dim,
            train_edge_index,
        )
        loss_lp, auc, ap = self._compute_lp_loss(
            embeddings, expert_weights, pos_edges, neg_edges
        )
        total_loss = loss_lp + self.coef_dis * loss_dis
        return total_loss, auc, ap

    def _compute_lp_loss(
        self,
        embeddings: torch.Tensor,
        expert_weights: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """BCE link prediction + AUC/AP metrics."""
        pos_scores = self._decode_edges(embeddings, expert_weights, pos_edges)
        neg_scores = self._decode_edges(embeddings, expert_weights, neg_edges)

        loss = F.binary_cross_entropy(
            pos_scores.clamp(0.01, 0.99), torch.ones_like(pos_scores)
        ) + F.binary_cross_entropy(
            neg_scores.clamp(0.01, 0.99), torch.zeros_like(neg_scores)
        )

        with torch.no_grad():
            labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
            preds = (
                pos_scores.detach().cpu().tolist()
                + neg_scores.detach().cpu().tolist()
            )
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score

                auc = roc_auc_score(labels, preds)
                ap_val = average_precision_score(labels, preds)
            except (ImportError, ValueError):
                auc, ap_val = 0.0, 0.0

        return loss, auc, ap_val

    def _decode_edges(
        self,
        embeddings: torch.Tensor,
        expert_weights: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        """Decode edges using alignment-weighted distances."""
        diff = (embeddings[edges[0]] - embeddings[edges[1]]) ** 2
        diff = diff.reshape(diff.shape[0], self.num_experts, self.embed_dim).sum(dim=2)
        weights = F.softmax(
            expert_weights[edges[0]] * expert_weights[edges[1]], dim=1
        )
        dist = torch.sum(diff * weights, dim=-1)
        return self.decoder(dist)

    def encode(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Tangent-space embeddings for downstream."""
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.experts.encode(features, edge_index)

    def get_expert_weights(
        self,
        subgraph_features: list[torch.Tensor],
        subgraph_edge_indices: list[torch.Tensor],
        subgraph_batches: list[torch.Tensor],
    ) -> torch.Tensor:
        """Expert weights from the gating network for downstream."""
        return self.gating(subgraph_features, subgraph_edge_indices, subgraph_batches)

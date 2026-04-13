"""
Shared GNN backbone (sparse PyG / dense adjacency).

Canonical module: ``pygfm.private.core.gnn_encoder``; data symbols in ``pygfm.private.utlis``.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class GNNBackboneEncoder:
    """
    Main entry point for the GNN Backbone.
    Supports GCN, GAT, and GIN architectures with domain-specific prompting.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        gnn_type: Literal["gcn", "gat", "gin"] = "gcn",
        activation: Literal["relu", "tanh", "gelu"] = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        device: Optional[str] = None,
        trainable: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.activation_type = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.prompt_embedding = nn.Embedding(10, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.model = self._build_model()
        self.model.to(self.device)
        self.input_proj.to(self.device)
        self.prompt_embedding.to(self.device)

        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool):
        """Toggle gradients and training/evaluation modes for all modules."""
        self.trainable = trainable
        main_modules = [self.model, self.input_proj, self.prompt_embedding]
        for module in main_modules:
            for param in module.parameters():
                param.requires_grad = trainable
            if trainable:
                module.train()
            else:
                module.eval()

    def _build_model(self) -> nn.Module:
        """Instantiate the internal sparse GNN encoder based on gnn_type."""
        params = {
            "input_dim": self.hidden_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "activation": self.activation_type,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
        }
        if self.gnn_type == "gcn":
            return GCNEncoderSparse(**params)
        if self.gnn_type == "gat":
            return GATEncoderSparse(**params)
        if self.gnn_type == "gin":
            return GINEncoderSparse(**params)
        raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def forward(self, aligned_features: torch.Tensor, edge_index: torch.Tensor, domain_id: int = 0) -> torch.Tensor:
        if not self.trainable:
            with torch.no_grad():
                return self._process(aligned_features, edge_index, domain_id)
        return self._process(aligned_features, edge_index, domain_id)

    def _process(self, x, edge_index, domain_id):
        """Unified processing pipeline: Projection -> Prompt Injection -> GNN Encoding."""
        x = self.input_proj(x)
        prompt = self.prompt_embedding(torch.tensor(domain_id).to(self.device))
        x = x + prompt
        h = self.model(x, edge_index)
        return h


# --- Sparse GNN Encoder Implementations ---


class GCNEncoderSparse(nn.Module):
    """GCN implementation using PyG Sparse message passing."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None

        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(GCNConv(in_d, hidden_dim))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.activation = {"relu": F.relu, "tanh": torch.tanh, "gelu": F.gelu}[activation]
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.bns:
                x = self.bns[i](x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEEncoderSparse(nn.Module):
    """GraphSAGE (mean aggregation) using PyG SAGEConv."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(SAGEConv(in_d, hidden_dim, aggr="mean"))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activation = {"relu": F.relu, "tanh": torch.tanh, "gelu": F.gelu}[activation]
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.bns:
                x = self.bns[i](x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNEncoderSparseWithPrompts(nn.Module):
    """
    GCN with optional per-layer structure prompts (e.g. SAMGPT).
    """

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(GCNConv(in_d, hidden_dim))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activation = {"relu": F.relu, "tanh": torch.tanh, "gelu": F.gelu}[activation]
        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
        batch: Optional[torch.Tensor] = None,
        structure_prompt_layers_per_domain: Optional[List[List[nn.Module]]] = None,
        structure_prompt_layers: Optional[List[nn.Module]] = None,
    ):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.bns:
                x = self.bns[i](x)
            if structure_prompt_layers_per_domain is not None and batch is not None:
                x_out = x.clone()
                for d, prompt_list in enumerate(structure_prompt_layers_per_domain):
                    if i < len(prompt_list):
                        mask = batch == d
                        if mask.any():
                            x_out[mask] = prompt_list[i](x[mask])
                x = x_out
            elif structure_prompt_layers is not None and i < len(structure_prompt_layers):
                x = structure_prompt_layers[i](x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoderSparse(nn.Module):
    """GAT implementation using PyG Sparse attention mechanisms."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim * heads
            concat = True if i < num_layers - 1 else False
            self.layers.append(GATConv(in_d, hidden_dim, heads=heads, concat=concat, dropout=dropout))

        self.activation = {"relu": F.relu, "tanh": torch.tanh, "gelu": F.gelu}[activation]

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class GINEncoderSparse(nn.Module):
    """GIN implementation using PyG Sparse MLP-based aggregation."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


# --- Dense/Standard Matrix-based GNN Implementations ---


class GCNEncoder(nn.Module):
    """Dense matrix GCN implementation with normalized adjacency processing."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.activation = self._get_act(activation)
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_batch_norm
        if use_batch_norm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def _get_act(self, name):
        return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[name]

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1)
        deg_inv = torch.pow(deg, -0.5)
        deg_inv[torch.isinf(deg_inv)] = 0
        deg_mat = torch.diag(deg_inv)
        adj_norm = deg_mat @ adj @ deg_mat

        for i, layer in enumerate(self.layers):
            res = x
            x = adj_norm @ layer(x)
            if self.use_bn:
                x = self.bns[i](x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
            if i > 0 and x.shape == res.shape:
                x = x + res
        return x


class GATEncoder(nn.Module):
    """Dense matrix GAT implementation."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout))
        for _ in range(num_layers - 1):
            self.layers.append(GraphAttentionLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout))

        self.activation = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[activation]
        self.use_bn = use_batch_norm
        if use_batch_norm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers)])

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if self.use_bn:
                x = self.bns[i](x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class GraphAttentionLayer(nn.Module):
    """Dense GAT Layer using einsum for efficient multi-head attention calculation."""

    def __init__(self, in_f, out_f, heads, dropout):
        super().__init__()
        self.out_f, self.heads = out_f, heads
        self.W = nn.Parameter(torch.empty(in_f, out_f * heads))
        self.a = nn.Parameter(torch.empty(2 * out_f, heads))
        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):
        N = x.size(0)
        h = torch.mm(x, self.W).view(N, self.heads, self.out_f)

        e_src = torch.einsum("nhf,fh->nh", h, self.a[: self.out_f, :])
        e_dst = torch.einsum("nhf,fh->nh", h, self.a[self.out_f :, :])

        e = self.leaky(e_src.unsqueeze(1) + e_dst.unsqueeze(0))
        mask = -9e15 * torch.ones_like(e)
        attn = torch.where(adj.unsqueeze(-1) > 0, e, mask)
        attn = F.softmax(attn, dim=1)
        attn = self.drop(attn)

        h_prime = torch.einsum("ijh,jhf->ihf", attn, h)
        return h_prime.reshape(N, self.heads * self.out_f)


class GINEncoder(nn.Module):
    """Dense matrix GIN implementation."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, use_batch_norm):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINLayer(mlp))

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class GINLayer(nn.Module):
    """Dense GIN Layer applying (1 + epsilon) * central_node + sum(neighbors)."""

    def __init__(self, mlp, eps=0.0):
        super().__init__()
        self.mlp, self.eps = mlp, eps

    def forward(self, x, adj):
        return self.mlp((1 + self.eps) * x + torch.mm(adj, x))


__all__ = [
    "GNNBackboneEncoder",
    "GCNEncoderSparse",
    "GATEncoderSparse",
    "GINEncoderSparse",
    "GCNEncoder",
    "GATEncoder",
    "GraphAttentionLayer",
    "GINEncoder",
    "GINLayer",
]

"""
Two-layer GCN surrogate compatible with DeepRobust Nettack (expects transposed conv weights).
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.gc1 = GCNConv(in_channels, hidden_channels)
        self.gc2 = GCNConv(hidden_channels, out_channels)
        self.gc1.weight = self.gc1.lin.weight.T
        self.gc2.weight = self.gc2.lin.weight.T
        self.nfeat = in_channels
        self.nclass = out_channels
        self.hidden_sizes = [hidden_channels]

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = self.gc2(x, edge_index)
        return x

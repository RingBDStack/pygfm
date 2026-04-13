import numpy as np
import scipy.sparse as sp
import torch

class GraphConstruction:
    def __init__(self, directed: bool = False, self_loop: bool = True):
        self.directed = directed
        self.self_loop = self_loop

    def forward(self, edge_list: np.ndarray, num_nodes: int = None) -> sp.coo_matrix:
        if edge_list.size == 0:
            return sp.coo_matrix((0, 0))
        
        if num_nodes is None:
            num_nodes = int(np.max(edge_list)) + 1
        
        row, col = edge_list[0], edge_list[1]
        data = np.ones(row.shape[0])
        adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        
        if not self.directed:
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        if self.self_loop:
            adj = adj + sp.eye(adj.shape[0])
            adj = adj.sign()
        return adj.tocoo()
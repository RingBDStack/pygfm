# down_model.py (final architecture variant)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from .moe import MoEModel
import numpy as np
import time
from torch_geometric.utils import subgraph, remove_self_loops

# create_community_bins and MultiHeadSimilarity unchanged from prior version.
def create_community_bins(communities, bin_edges):
    sorted_edges = sorted(bin_edges)
    bins = [[] for _ in range(len(sorted_edges) + 1)]
    for comm in communities:
        size = len(comm)
        for i, edge in enumerate(sorted_edges):
            if size <= edge:
                bins[i].append(comm)
                break
        else:
            bins[-1].append(comm)
    return [b for b in bins if b]

class MultiHeadSimilarity(nn.Module):
    def __init__(self, input_dim, num_heads=4, head_dim=16):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        total_dim = num_heads * head_dim
        self.q_proj = nn.Linear(input_dim, total_dim)
        self.k_proj = nn.Linear(input_dim, total_dim)
    def forward(self, x, mask):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim_heads = torch.matmul(q, k.transpose(-1, -2))
        sim_mean, sim_var = sim_heads.mean(dim=1), sim_heads.var(dim=1)
        valid_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        diag_mask = ~torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        final_mask = valid_mask & diag_mask
        sim_mean[~final_mask] = -1e9
        sim_var[~final_mask] = 0.0
        return sim_mean, sim_var

class IntraClusterOptimizer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads, head_dim, gamma, tau, lambda_var, alpha, bucket_boundaries):
        super().__init__()
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.sim_net = MultiHeadSimilarity(hidden_dim, num_heads, head_dim)
        self.gamma, self.tau, self.lambda_var, self.alpha = gamma, tau, lambda_var, alpha
        self.bucket_boundaries = bucket_boundaries
    def forward(self, features, edge_index, num_nodes, valid_communities):
        x = F.relu(self.projection(features))
        all_fused_edge_indices, all_fused_edge_weights = [], []
        total_variance, total_valid_pairs = 0.0, 0.0
        if valid_communities:
            community_bins = create_community_bins(valid_communities, self.bucket_boundaries)
            for comm_bin in community_bins:
                max_len_in_bin = max(len(c) for c in comm_bin)
                x_cluster_batch = torch.zeros(len(comm_bin), max_len_in_bin, x.shape[1], device=x.device)
                mask = torch.zeros(len(comm_bin), max_len_in_bin, dtype=torch.bool, device=x.device)
                for i, c in enumerate(comm_bin):
                    x_cluster_batch[i, :len(c), :] = x[c]
                    mask[i, :len(c)] = True
                sim_mean, sim_var = self.sim_net(x_cluster_batch, mask)
                A_optimized = torch.sigmoid(self.gamma * (sim_mean - self.tau)) * torch.exp(-self.lambda_var * sim_var)
                diag_mask = ~torch.eye(max_len_in_bin, device=x.device, dtype=torch.bool).unsqueeze(0)
                valid_mask = mask.unsqueeze(1) & mask.unsqueeze(2) & diag_mask
                total_variance += (sim_var * valid_mask).sum()
                total_valid_pairs += valid_mask.sum()
                for b, cluster in enumerate(comm_bin):
                    cluster_nodes = torch.tensor(cluster, dtype=torch.long, device=x.device)
                    sub_edge_index, _ = subgraph(cluster_nodes, edge_index, relabel_nodes=True, num_nodes=num_nodes)
                    A_ori_cluster = torch.zeros((len(cluster), len(cluster)), device=x.device)
                    if sub_edge_index.numel() > 0:
                        A_ori_cluster[sub_edge_index[0], sub_edge_index[1]] = 1.0
                    A_opt_cluster = A_optimized[b, :len(cluster), :len(cluster)]
                    fused_cluster_dense = self.alpha * A_ori_cluster + (1 - self.alpha) * A_opt_cluster
                    local_rows, local_cols = fused_cluster_dense.nonzero(as_tuple=True)
                    fused_cluster_sparse_weights = fused_cluster_dense[local_rows, local_cols]
                    global_rows, global_cols = cluster_nodes[local_rows], cluster_nodes[local_cols]
                    global_indices = torch.stack([global_rows, global_cols])
                    all_fused_edge_indices.append(global_indices)
                    all_fused_edge_weights.append(fused_cluster_sparse_weights)
        uncertainty_loss = total_variance / total_valid_pairs.clamp(min=1.0) if total_valid_pairs > 0 else torch.tensor(0.0, device=x.device)
        if not all_fused_edge_indices:
             return torch.empty((2, 0), dtype=torch.long, device=x.device), torch.empty(0, device=x.device), uncertainty_loss
        final_edge_index = torch.cat(all_fused_edge_indices, dim=1)
        final_edge_weight = torch.cat(all_fused_edge_weights, dim=0)
        return final_edge_index, final_edge_weight, uncertainty_loss

# Fast lookup helper for sparse similarity scores on edges.
class SparseLookup:
    def __init__(self, S_sparse, num_nodes):
        print("    - Building fast lookup (SparseLookup)...")
        S_indices = S_sparse.indices()
        S_values = S_sparse.values()
        
        # 1. Hash keys and sort once
        S_keys = S_indices[0] * num_nodes + S_indices[1]
        self.sorted_S_keys, p = S_keys.sort()
        self.sorted_S_values = S_values[p]
        self.num_nodes = num_nodes
        self.device = S_sparse.device
        print("    - SparseLookup ready.")

    def lookup(self, query_src, query_dst):
        # 2. Hash keys for query edges
        query_keys = query_src * self.num_nodes + query_dst

        # 3. Binary search via searchsorted
        insertion_indices = torch.searchsorted(self.sorted_S_keys, query_keys)

        # 4. Fill score tensor from matches
        scores = torch.zeros(query_keys.numel(), device=self.device)
        
        # 5. Keys that actually match
        found_mask = (insertion_indices < len(self.sorted_S_keys))
        found_mask &= (self.sorted_S_keys[insertion_indices.clamp(max=len(self.sorted_S_keys)-1)] == query_keys)

        # 6. Assign scores only for matched edges
        scores[found_mask] = self.sorted_S_values[insertion_indices[found_mask]]
        return scores

# InterClusterOptimizer takes a SparseLookup for O(log n) score queries.
class InterClusterOptimizer(nn.Module):
    def __init__(self, initial_threshold, temperature):
        super().__init__()
        self.inter_cluster_threshold = nn.Parameter(torch.tensor(initial_threshold))
        self.temperature = temperature

    def forward(self, edge_index, num_nodes, S_lookup, node_to_cluster_id):
        device = edge_index.device
        row, col = edge_index
        
        cluster_map = torch.empty(num_nodes, dtype=torch.long, device=device)
        for node_id, cluster_id in node_to_cluster_id.items():
            cluster_map[node_id] = cluster_id
            
        is_inter = cluster_map[row] != cluster_map[col]
        inter_src, inter_dst = row[is_inter], col[is_inter]
        
        if inter_src.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)

        # Use lookup() instead of per-batch sorting.
        scores = S_lookup.lookup(inter_src, inter_dst)

        retention_probs = torch.sigmoid((scores - self.inter_cluster_threshold) * self.temperature)
        
        retained_edge_mask = retention_probs > 0
        final_inter_src = inter_src[retained_edge_mask]
        final_inter_dst = inter_dst[retained_edge_mask]
        final_inter_weights = retention_probs[retained_edge_mask]
        
        final_inter_edge_index = torch.stack([final_inter_src, final_inter_dst])
        return final_inter_edge_index, final_inter_weights

# downprompt.forward accepts S_lookup when inter-cluster optimizer is on.
class downprompt(nn.Module):
    def __init__(self, args, pretrained_models, pretrain_model_multi, init_weights=None):
        super().__init__()
        self.moe = MoEModel(pretrained_models, init_weights)
        self.intra_optimizer = IntraClusterOptimizer(
            args.unify_dim, args.hidden_dim, args.num_heads, args.head_dim, args.gamma, args.tau, 
            args.lambda_var, args.alpha, args.bucket_boundaries
        )
        self.use_inter_optimizer = args.inter_cluster_optimizer
        if self.use_inter_optimizer:
            self.inter_optimizer = InterClusterOptimizer(args.inter_cluster_threshold, args.inter_cluster_temperature)
        
        self.num_classes = None
        self.ave_embeddings = None
        self.final_graph = None
        self.multi_model = pretrain_model_multi
        self.moe_weight = args.moe_embedding_weight
        self.multi_weight = args.multi_embedding_weight
        assert abs(self.moe_weight + self.multi_weight - 1.0) < 1e-6

    def forward(self, x, edge_index, num_nodes, idx, labels, is_train, valid_communities, size_one_communities, S_lookup=None, node_to_cluster_id=None):
        if isinstance(labels, np.ndarray): labels = torch.from_numpy(labels)
        if is_train: self.num_classes = len(torch.unique(labels))
        
        struct_loss = torch.tensor(0.0, device=x.device)
        if is_train:
            intra_start_time = time.time()
            intra_edge_index, intra_edge_weight, struct_loss = self.intra_optimizer(x, edge_index, num_nodes, valid_communities)
            intra_end_time = time.time()
            print(f"Intra-cluster optimization time: {intra_end_time - intra_start_time:.2f} s")

            if self.use_inter_optimizer:
                inter_start_time = time.time()
                if S_lookup is None or node_to_cluster_id is None:
                    raise ValueError("S_lookup and node_to_cluster_id must be provided.")
                inter_edge_index, inter_edge_weight = self.inter_optimizer(edge_index, num_nodes, S_lookup, node_to_cluster_id)
                final_edge_index = torch.cat([intra_edge_index, inter_edge_index], dim=1)
                final_edge_weight = torch.cat([intra_edge_weight, inter_edge_weight], dim=0)
                inter_end_time = time.time()
                print(f"Inter-cluster optimization time: {inter_end_time - inter_start_time:.2f} s")
            else:
                final_edge_index, final_edge_weight = intra_edge_index, intra_edge_weight

            if size_one_communities:
                size_one_start_time = time.time()
                nodes_in_size_one = torch.tensor([c[0] for c in size_one_communities], device=x.device)
                mask_src = torch.isin(edge_index[0], nodes_in_size_one)
                mask_dst = torch.isin(edge_index[1], nodes_in_size_one)
                mask_size_one = mask_src | mask_dst
                size_one_edges = edge_index[:, mask_size_one]
                size_one_weights = torch.ones(size_one_edges.shape[1], device=x.device)
                final_edge_index = torch.cat([final_edge_index, size_one_edges], dim=1)
                final_edge_weight = torch.cat([final_edge_weight, size_one_weights], dim=0)
                size_one_end_time = time.time()
                print(f"size_one communities time: {size_one_end_time - size_one_start_time:.2f} s")

            remove_self_loops_start_time = time.time()
            final_edge_index, final_edge_weight = remove_self_loops(final_edge_index, final_edge_weight)
            remove_self_loops_end_time = time.time()
            print(f"remove_self_loops time: {remove_self_loops_end_time - remove_self_loops_start_time:.2f} s")

            final_adj_sparse = torch.sparse_coo_tensor(final_edge_index, final_edge_weight, (num_nodes, num_nodes)).coalesce()
            self.final_graph = final_adj_sparse

        current_adj = self.final_graph if self.final_graph is not None else torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.shape[1], device=x.device), (num_nodes, num_nodes)
        ).coalesce()
        if self.multi_model is not None:
            node_embeddings = self.moe(x, current_adj)
            node_embeddings_multi = self.multi_model.get_embeddings([x], [current_adj])
            node_embeddings = node_embeddings * self.moe_weight + node_embeddings_multi * self.multi_weight
        else:
            node_embeddings = self.moe(x, current_adj)
        selected_embeddings = node_embeddings[idx]

        if is_train:
            self.ave_embeddings = torch_scatter.scatter(src=selected_embeddings, index=labels.to(x.device), dim=0, reduce='mean')
        
        if self.ave_embeddings is None:
            raise RuntimeError("ave_embeddings has not been computed. Train model once before validation.")
        rawret_concat = torch.cat((selected_embeddings, self.ave_embeddings), dim=0)
        sim_matrix = torch.cosine_similarity(rawret_concat.unsqueeze(1), rawret_concat.unsqueeze(0), dim=-1)
        ret = F.softmax(sim_matrix[:len(idx), len(idx):], dim=1)
        moe_loss = self.moe.get_uncertainty_loss() if is_train else torch.tensor(0.0, device=x.device)
        return ret, moe_loss, struct_loss
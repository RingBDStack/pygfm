import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class MoEModel(nn.Module):
    def __init__(self, pretrained_models, init_weights=None):
        """
        Mixture-of-experts over frozen pretrained GNNs.

        Args:
            pretrained_models: List of pretrained GNN models.
            init_weights: Optional initial routing logits [num_experts] (already softmaxed).
        """
        super(MoEModel, self).__init__()
        
        self.pretrained_models = pretrained_models
        self.num_experts = len(pretrained_models)
        
        # Learnable routing weights (one row, softmaxed in forward).
        if init_weights is not None:
            self.routing_weights = nn.Parameter(init_weights.unsqueeze(0))
        else:
            self.routing_weights = nn.Parameter(torch.ones(1, self.num_experts) / self.num_experts)
    
    def get_uncertainty_loss(self):
        """Entropy of softmax routing weights (uncertainty regularizer)."""
        weights = F.softmax(self.routing_weights, dim=1)
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=1).mean()
        return entropy
    
    def forward(self, x, adj_sparse):
        """
        Weighted sum of expert node embeddings.

        Args:
            x: Node features [num_nodes, in_channels].
            adj_sparse: Sparse adjacency.

        Returns:
            weighted_output: Fused node representations [num_nodes, out_channels].
        """
        expert_outputs = []
        for model in self.pretrained_models:
            with torch.no_grad():
                output = model.get_embeddings([x], [adj_sparse])
                expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs)
        routing_weights = F.softmax(self.routing_weights, dim=1)  # [1, num_experts]
        weighted_output = (routing_weights.unsqueeze(-1) * expert_outputs.permute(1, 0, 2)).sum(dim=1)
        return weighted_output
    
    def get_weights(self):
        return self.routing_weights
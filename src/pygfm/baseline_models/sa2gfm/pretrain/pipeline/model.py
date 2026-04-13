# JointContrastiveModel (single-graph pretrain only in this bundle; no joint_data merge).

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.public.utils.loss_func import gather_rows


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=True):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.matmul(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GaussianEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


def compareloss(feature, tuples, temperature=1.0):
    h_tuples = gather_rows(feature, tuples)
    anchor_indices = torch.arange(0, len(tuples), device=feature.device).view(-1, 1).expand(-1, tuples.size(1))
    h_i = gather_rows(feature, anchor_indices)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    sim = torch.clamp(sim, min=-10, max=10)
    exp_sim = torch.exp(sim / temperature)
    numerator = exp_sim[:, 0]
    denominator = torch.sum(exp_sim, dim=1)
    loss = -torch.log(numerator / (denominator + 1e-9))
    return loss.mean()


class JointContrastiveModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(JointContrastiveModel, self).__init__()
        self.dropout_p = dropout
        self.num_layers = num_layers
        self.gcn_encoder = nn.ModuleList()
        self.gcn_encoder.append(GCN(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_encoder.append(GCN(hidden_channels, hidden_channels))
        self.gcn_encoder.append(GCN(hidden_channels, out_channels))
        self.gaussian_encoder = GaussianEncoder(out_channels, hidden_channels, out_channels)
        self.temperature = 1.0

    def _encode(self, features, adj):
        h = features
        for i, layer in enumerate(self.gcn_encoder):
            h = layer(h, adj, sparse=True)
            if i < self.num_layers - 1:
                h = F.dropout(h, self.dropout_p, training=self.training)
        return self.gaussian_encoder(h)

    def get_embeddings(self, features_list, adj_list):
        mu_list = []
        with torch.no_grad():
            for features, adj in zip(features_list, adj_list):
                _, mu, _ = self._encode(features, adj)
                mu_list.append(mu)
        return torch.cat(mu_list, dim=0)

    def forward(self, features_list, adj_list, neg_samples):
        z_list, mu_list, logvar_list = [], [], []
        for features, adj in zip(features_list, adj_list):
            z, mu, logvar = self._encode(features, adj)
            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)
        z_all = torch.cat(z_list, dim=0)
        mu_all = torch.cat(mu_list, dim=0)
        logvar_all = torch.cat(logvar_list, dim=0)
        contrastive_loss = compareloss(z_all, neg_samples, temperature=self.temperature)
        kl_loss = -0.5 * torch.mean(1 + logvar_all - mu_all.pow(2) - logvar_all.exp())
        return contrastive_loss, kl_loss

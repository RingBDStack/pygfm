"""
Canonical CCA-style loss and projections (legacy OFA / GP stack).

**Canonical** implementation here; ``pygfm.baseline_models.oneforall.gp.nn.loss`` re-exports only.
``forward`` returns ``(corr, U, V)``; training code may optimize ``corr`` only—use what your task needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CCALoss(nn.Module):
    def __init__(self, outdim_size: int = 20) -> None:
        super().__init__()
        self.outdim_size = outdim_size

    def forward(
        self, h1: torch.Tensor, h2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r1 = 1e-5
        r2 = 1e-5
        eps = 1e-7

        h1, h2 = h1.t(), h2.t()

        o1 = o2 = h1.size(0)
        m = h1.size(1)

        h1bar = h1 - h1.mean(dim=1).unsqueeze(dim=1)
        h2bar = h2 - h2.mean(dim=1).unsqueeze(dim=1)

        sigma_hat12 = (1.0 / (m - 1)) * torch.matmul(h1bar, h2bar.t())
        sigma_hat11 = (1.0 / (m - 1)) * torch.matmul(
            h1bar, h1bar.t()
        ) + r1 * torch.eye(o1, device=h1.device)
        sigma_hat22 = (1.0 / (m - 1)) * torch.matmul(
            h2bar, h2bar.t()
        ) + r2 * torch.eye(o2, device=h2.device)

        assert torch.isnan(sigma_hat11).sum().item() == 0
        assert torch.isnan(sigma_hat12).sum().item() == 0
        assert torch.isnan(sigma_hat22).sum().item() == 0

        d1, v1 = torch.linalg.eigh(sigma_hat11)
        d2, v2 = torch.linalg.eigh(sigma_hat22)
        uv1, c1 = torch.unique(d1, return_counts=True)
        uv2, c2 = torch.unique(d2, return_counts=True)
        if len(uv1[c1 > 1]) > 0 or len(uv2[c2 > 1]) > 0:
            return (
                torch.tensor(0.0, requires_grad=True, dtype=torch.float, device=h1.device),
                torch.eye(o1, device=h1.device),
                torch.eye(o1, device=h1.device),
            )

        assert torch.isnan(d1).sum().item() == 0
        assert torch.isnan(d2).sum().item() == 0
        assert torch.isnan(v1).sum().item() == 0
        assert torch.isnan(v2).sum().item() == 0

        pos_ind1 = torch.gt(d1, eps).nonzero()[:, 0]
        d1 = d1[pos_ind1]
        v1 = v1[:, pos_ind1]
        pos_ind2 = torch.gt(d2, eps).nonzero()[:, 0]
        d2 = d2[pos_ind2]
        v2 = v2[:, pos_ind2]

        sigma_hat11_root_inv = torch.matmul(
            torch.matmul(v1, torch.diag(d1**-0.5)), v1.t()
        )
        sigma_hat22_root_inv = torch.matmul(
            torch.matmul(v2, torch.diag(d2**-0.5)), v2.t()
        )

        t_val = torch.matmul(
            torch.matmul(sigma_hat11_root_inv, sigma_hat12), sigma_hat22_root_inv
        )

        trace_tt = torch.matmul(t_val.t(), t_val)
        trace_tt = torch.add(
            trace_tt, (torch.eye(trace_tt.shape[0], device=h1.device) * r1)
        )
        u_eig, _ = torch.linalg.eigh(trace_tt)
        u_eig = torch.where(u_eig > eps, u_eig, torch.ones_like(u_eig, device=h1.device) * eps)
        u_eig = u_eig.topk(self.outdim_size)[0]
        corr = torch.sum(torch.sqrt(u_eig))
        u_svd, _, vh = torch.linalg.svd(t_val, full_matrices=False)
        u_svd = u_svd[:, : self.outdim_size]
        v_svd = vh.mT[:, : self.outdim_size]
        u_svd = torch.matmul(sigma_hat11_root_inv, u_svd)
        v_svd = torch.matmul(sigma_hat22_root_inv, v_svd)
        return corr, u_svd, v_svd

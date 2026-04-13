"""Gradient Reversal Layer (GRL) for adversarial domain alignment."""
from __future__ import annotations

import torch


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


__all__ = ["GradientReversal"]

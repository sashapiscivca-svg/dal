"""Leaky integrate-and-fire (LIF) neuron with a surrogate gradient, about 40 lines, no SNN framework.

The membrane potential is an explicit input and output of every call, so the module unrolls in time into a static graph for
ONNX and the membrane is carried between frames as the motion memory. Ideas: LIF / PLIF (Fang et al., 2021), surrogate
gradients (Neftci et al., 2019).
"""
from __future__ import annotations

import torch
from torch import nn


class SpikeFn(torch.autograd.Function):
    """Heaviside step forward; a fast sigmoid surrogate with slope 4 backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        a = 4.0
        sg = a * torch.sigmoid(a * x) * (1 - torch.sigmoid(a * x))
        return g * sg


class LIF(nn.Module):
    """mem' = beta·mem + x;  s = [mem' ≥ thr];  mem'' = mem' − s·thr  (soft reset)."""

    def __init__(self, beta: float = 0.8, thr: float = 1.0, learn_beta: bool = False):
        super().__init__()
        self.thr = thr
        self.learn_beta = learn_beta
        raw = torch.logit(torch.tensor(float(beta)).clamp(1e-3, 1 - 1e-3))
        if learn_beta:
            self.raw_beta = nn.Parameter(raw)
        else:
            self.register_buffer("raw_beta", raw)

    def beta_value(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta)

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.beta_value() * mem + x
        s = SpikeFn.apply(mem - self.thr)
        return s, mem - s.detach() * self.thr

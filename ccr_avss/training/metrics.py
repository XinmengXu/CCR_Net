"""Evaluation metrics used by the command-line evaluator."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def si_snr(signal: Tensor, target: Tensor, epsilon: float = 1.0e-8) -> Tensor:
    signal = signal - signal.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (signal * target).sum(dim=-1, keepdim=True) * target
    projection = projection / (target.square().sum(dim=-1, keepdim=True) + epsilon)
    noise = signal - projection
    return 10.0 * torch.log10(
        projection.square().sum(dim=-1) / (noise.square().sum(dim=-1) + epsilon) + epsilon
    )


def sdr(signal: Tensor, target: Tensor, epsilon: float = 1.0e-8) -> Tensor:
    error = target - signal
    return 10.0 * torch.log10(
        target.square().sum(dim=-1) / (error.square().sum(dim=-1) + epsilon) + epsilon
    )

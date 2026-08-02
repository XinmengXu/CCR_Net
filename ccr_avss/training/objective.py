"""Multi-term training objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ccr_avss.audio.spectral import CompressedSpectralTransform

Tensor = torch.Tensor


@dataclass(frozen=True)
class ObjectiveWeights:
    si_snr: float = 0.025
    magnitude: float = 0.9
    complex: float = 0.1
    phase: float = 0.3


def negative_si_snr(estimate: Tensor, target: Tensor, epsilon: float = 1.0e-8) -> Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target
    projection = projection / (target.square().sum(dim=-1, keepdim=True) + epsilon)
    residual = estimate - projection
    ratio = projection.square().sum(dim=-1) / (residual.square().sum(dim=-1) + epsilon)
    return -(10.0 * torch.log10(ratio + epsilon)).mean()


class SeparationObjective(nn.Module):
    def __init__(
        self,
        fft_size: int = 512,
        hop_size: int = 256,
        window_size: int = 512,
        compression: float = 0.3,
        weights: ObjectiveWeights | None = None,
    ) -> None:
        super().__init__()
        self.transform = CompressedSpectralTransform(fft_size, hop_size, window_size, compression)
        self.weights = weights or ObjectiveWeights()

    def forward(self, prediction: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        if target.ndim == 3:
            target = target[:, 0]
        target_packet = self.transform.analyze(target)
        waveform_loss = negative_si_snr(prediction["waveform"][:, 0], target)
        magnitude_loss = F.l1_loss(prediction["magnitude"], target_packet.compressed_magnitude)
        complex_loss = F.l1_loss(prediction["compressed_cartesian"], target_packet.compressed_cartesian)
        phase_loss = (1.0 - torch.cos(prediction["phase"] - target_packet.phase)).mean()
        total = (
            self.weights.si_snr * waveform_loss
            + self.weights.magnitude * magnitude_loss
            + self.weights.complex * complex_loss
            + self.weights.phase * phase_loss
        )
        return {
            "total": total,
            "si_snr": waveform_loss,
            "magnitude": magnitude_loss,
            "complex": complex_loss,
            "phase": phase_loss,
        }

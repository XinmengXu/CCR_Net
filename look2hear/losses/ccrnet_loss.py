"""Training objective for the complete CCRNet AVSS model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


Tensor = torch.Tensor


def negative_si_snr(estimate: Tensor, target: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Negative scale-invariant SNR averaged over the batch."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        (estimate * target).sum(dim=-1, keepdim=True)
        * target
        / (target.square().sum(dim=-1, keepdim=True) + eps)
    )
    noise = estimate - projection
    ratio = projection.square().sum(dim=-1) / (noise.square().sum(dim=-1) + eps)
    return -(10.0 * torch.log10(ratio + eps)).mean()


@dataclass(frozen=True)
class CCRNetLossWeights:
    si_snr: float = 0.025
    magnitude: float = 0.9
    complex: float = 0.1
    phase: float = 0.3


class CCRNetLoss(nn.Module):
    """Combined waveform, magnitude, complex, and phase objective."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        compression: float = 0.3,
        weights: CCRNetLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.compression = compression
        self.weights = weights or CCRNetLossWeights()
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
            persistent=False,
        )

    def _target_features(self, target: Tensor) -> Dict[str, Tensor]:
        spectrum = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(target),
            center=True,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.abs(spectrum).clamp_min(1.0e-12).pow(self.compression)
        phase = torch.angle(spectrum)
        compressed_complex = torch.stack(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)],
            dim=1,
        )
        return {
            "magnitude": magnitude,
            "phase": phase,
            "compressed_complex": compressed_complex,
        }

    def forward(self, output: Dict[str, object], target: Tensor) -> Dict[str, Tensor]:
        if target.ndim == 3:
            if target.size(1) != 1:
                raise ValueError("CCRNetLoss expects one target source per sample.")
            target = target[:, 0]
        if target.ndim != 2:
            raise ValueError("target must have shape [B, T] or [B, 1, T].")

        waveform = output["waveform"]
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("output['waveform'] must be a tensor.")
        waveform = waveform[:, 0]
        target_features = self._target_features(target)

        estimated_magnitude = output["magnitude"]
        estimated_phase = output["phase"]
        estimated_complex = output["compressed_complex"]
        if not all(
            isinstance(item, torch.Tensor)
            for item in (estimated_magnitude, estimated_phase, estimated_complex)
        ):
            raise TypeError("CCRNet spectral outputs must be tensors.")

        si_snr_loss = negative_si_snr(waveform, target)
        magnitude_loss = F.l1_loss(estimated_magnitude, target_features["magnitude"])
        complex_loss = F.l1_loss(
            estimated_complex,
            target_features["compressed_complex"],
        )
        phase_loss = (1.0 - torch.cos(estimated_phase - target_features["phase"])).mean()

        total = (
            self.weights.si_snr * si_snr_loss
            + self.weights.magnitude * magnitude_loss
            + self.weights.complex * complex_loss
            + self.weights.phase * phase_loss
        )
        return {
            "total": total,
            "si_snr": si_snr_loss,
            "magnitude": magnitude_loss,
            "complex": complex_loss,
            "phase": phase_loss,
        }

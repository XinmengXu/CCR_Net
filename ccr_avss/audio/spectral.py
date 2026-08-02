"""Spectral analysis and waveform reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

Tensor = torch.Tensor


@dataclass(frozen=True)
class SpectralPacket:
    compressed_magnitude: Tensor
    phase: Tensor
    compressed_cartesian: Tensor
    complex_spectrum: Tensor


class CompressedSpectralTransform(nn.Module):
    """STFT analysis with power compression and exact-length synthesis."""

    def __init__(
        self,
        fft_size: int = 512,
        hop_size: int = 256,
        window_size: int = 512,
        compression: float = 0.3,
    ) -> None:
        super().__init__()
        if fft_size < window_size:
            raise ValueError("fft_size must be at least window_size")
        if not 0.0 < compression <= 1.0:
            raise ValueError("compression must lie in (0, 1]")
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.window_size = int(window_size)
        self.compression = float(compression)
        self.register_buffer("window", torch.hann_window(window_size), persistent=False)

    @property
    def frequency_bins(self) -> int:
        return self.fft_size // 2 + 1

    def analyze(self, waveform: Tensor) -> SpectralPacket:
        spectrum = torch.stft(
            waveform,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            onesided=True,
            return_complex=True,
        )
        magnitude = spectrum.abs().clamp_min(1.0e-12).pow(self.compression)
        phase = torch.angle(spectrum)
        cartesian = torch.stack(
            (magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1
        )
        return SpectralPacket(magnitude, phase, cartesian, spectrum)

    def synthesize(self, compressed_magnitude: Tensor, phase: Tensor, length: int) -> Tensor:
        magnitude = compressed_magnitude.clamp_min(1.0e-12).pow(1.0 / self.compression)
        spectrum = torch.polar(magnitude, phase)
        return torch.istft(
            spectrum,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.window.to(device=phase.device, dtype=phase.dtype),
            center=True,
            onesided=True,
            length=length,
        )

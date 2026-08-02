"""Reusable convolutional and sequence modeling blocks."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


class ChannelNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class GatedConvUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.filter = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation)
        self.gate = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation)
        self.mix = nn.Conv2d(channels, channels, 1)
        self.norm = ChannelNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        update = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        return self.norm(x + self.mix(update))


class MultiScaleFeatureStack(nn.Module):
    def __init__(self, channels: int, layers: int = 3) -> None:
        super().__init__()
        self.units = nn.ModuleList(
            GatedConvUnit(channels, dilation=2**index) for index in range(layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for unit in self.units:
            x = unit(x)
        return x


class FrequencyReducer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2)),
            ChannelNorm(channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class AxisAttention(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.post_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        normalized = self.pre_norm(sequence)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        sequence = sequence + attended
        return sequence + self.feed_forward(self.post_norm(sequence))


class TimeFrequencyContext(nn.Module):
    """Alternating temporal and frequency attention without external libraries."""

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.time = AxisAttention(channels, heads, dropout)
        self.frequency = AxisAttention(channels, heads, dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frames, bins = x.shape
        temporal = x.permute(0, 3, 2, 1).reshape(batch * bins, frames, channels)
        temporal = self.time(temporal)
        spectral = temporal.reshape(batch, bins, frames, channels).permute(0, 2, 1, 3)
        spectral = spectral.reshape(batch * frames, bins, channels)
        spectral = self.frequency(spectral)
        return spectral.reshape(batch, frames, bins, channels).permute(0, 3, 1, 2).contiguous()

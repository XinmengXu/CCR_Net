"""Visual feature extraction and audio-grid alignment."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


class SpatialResidualUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.path(x))


class MouthMotionEncoder(nn.Module):
    """Compact frame encoder that preserves the video time axis."""

    def __init__(self, output_channels: int = 512) -> None:
        super().__init__()
        self.temporal_stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            SpatialResidualUnit(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            SpatialResidualUnit(128),
        )
        self.temporal_projection = nn.Sequential(
            nn.Conv1d(128, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(),
            nn.Conv1d(output_channels, output_channels, 3, padding=1),
        )

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        if frames.ndim != 5:
            raise ValueError("mouth frames must have shape [B,T,H,W] or [B,1,T,H,W]")
        features = self.temporal_stem(frames)
        batch, channels, time, height, width = features.shape
        features = features.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        features = self.spatial(features).mean(dim=(-2, -1))
        features = features.reshape(batch, time, -1).transpose(1, 2)
        return self.temporal_projection(features)


class VisualGridProjector(nn.Module):
    """Map frame-level visual descriptors to the spectro-temporal grid."""

    def __init__(self, input_channels: int, frequency_bins: int) -> None:
        super().__init__()
        self.frequency_projection = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, 3, padding=1, groups=max(1, input_channels // 32)),
            nn.SiLU(),
            nn.Conv1d(input_channels, frequency_bins, 1),
        )

    def forward(self, embeddings: Tensor, audio_frames: int) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError("visual embeddings must have shape [B,C,T]")
        embeddings = F.interpolate(embeddings, size=audio_frames, mode="linear", align_corners=False)
        grid = self.frequency_projection(embeddings)
        return grid.transpose(1, 2).unsqueeze(1).contiguous()

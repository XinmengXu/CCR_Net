"""Stage-wise audio-visual encoder with controllable repair."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .blocks import ChannelNorm, MultiScaleFeatureStack
from .repair import RepairSettings, RepairTrace, VerifiedCrossModalRepair

Tensor = torch.Tensor


class CrossModalStage(nn.Module):
    def __init__(self, channels: int, settings: RepairSettings) -> None:
        super().__init__()
        self.audio_context = MultiScaleFeatureStack(channels, layers=3)
        self.visual_context = MultiScaleFeatureStack(channels, layers=3)
        self.audio_exchange = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), ChannelNorm(channels), nn.SiLU()
        )
        self.visual_exchange = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), ChannelNorm(channels), nn.SiLU()
        )
        self.repair = VerifiedCrossModalRepair(channels, settings)

    def forward(
        self,
        audio: Tensor,
        visual: Tensor,
        reference: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, RepairTrace]:
        audio_local = self.audio_context(audio)
        visual_local = self.visual_context(visual)
        audio = audio_local + self.audio_exchange(torch.cat((audio_local, visual_local), dim=1))
        visual = visual_local + self.visual_exchange(torch.cat((visual_local, audio_local), dim=1))
        return self.repair(audio, visual, reference)


class RepairControlledAVEncoder(nn.Module):
    def __init__(self, channels: int, stages: int, settings: RepairSettings) -> None:
        super().__init__()
        self.audio_entry = nn.Sequential(
            nn.Conv2d(1, channels, 1), ChannelNorm(channels), nn.SiLU()
        )
        self.visual_entry = nn.Sequential(
            nn.Conv2d(1, channels, 1), ChannelNorm(channels), nn.SiLU()
        )
        self.stages = nn.ModuleList(CrossModalStage(channels, settings) for _ in range(stages))
        self.final_audio = MultiScaleFeatureStack(channels, layers=2)
        self.final_visual = MultiScaleFeatureStack(channels, layers=2)

    def forward(self, magnitude_grid: Tensor, visual_grid: Tensor) -> tuple[Tensor, Tensor, list[RepairTrace]]:
        audio = self.audio_entry(magnitude_grid)
        visual = self.visual_entry(visual_grid)
        reference: Optional[Tensor] = None
        traces: list[RepairTrace] = []
        for stage in self.stages:
            audio, visual, reference, trace = stage(audio, visual, reference)
            traces.append(trace)
        return self.final_audio(audio), self.final_visual(visual), traces

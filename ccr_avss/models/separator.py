"""End-to-end audio-visual target speech separator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ccr_avss.audio.spectral import CompressedSpectralTransform
from ccr_avss.modules.av_encoder import RepairControlledAVEncoder
from ccr_avss.modules.blocks import ChannelNorm, FrequencyReducer, MultiScaleFeatureStack, TimeFrequencyContext
from ccr_avss.modules.repair import RepairSettings
from ccr_avss.vision.frontend import MouthMotionEncoder, VisualGridProjector

Tensor = torch.Tensor


@dataclass(frozen=True)
class SeparatorConfiguration:
    sample_rate: int = 16000
    visual_channels: int = 512
    feature_channels: int = 64
    av_stages: int = 4
    context_blocks: int = 4
    attention_heads: int = 4
    fft_size: int = 512
    hop_size: int = 256
    window_size: int = 512
    compression: float = 0.3
    shift_radius: int = 5
    leading_rank: int = 8
    reference_momentum: float = 0.95
    passthrough: float = 0.25
    trust_ratio: float = 0.25
    proposal_scale: float = 1.0


class MagnitudeHead(nn.Module):
    def __init__(self, channels: int, frequency_bins: int) -> None:
        super().__init__()
        self.context = MultiScaleFeatureStack(channels, layers=3)
        self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=(1, 3), stride=(1, 2))
        self.mask = nn.Conv2d(channels, 1, 1)
        self.slope = nn.Parameter(torch.ones(frequency_bins, 1))

    def forward(self, features: Tensor) -> Tensor:
        mask = self.mask(self.upsample(self.context(features)))
        mask = mask.squeeze(1).transpose(1, 2)
        return 2.0 * torch.sigmoid(self.slope * mask)


class PhaseHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.context = MultiScaleFeatureStack(channels, layers=3)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=(1, 3), stride=(1, 2)),
            ChannelNorm(channels),
            nn.SiLU(),
        )
        self.real = nn.Conv2d(channels, 1, 1)
        self.imaginary = nn.Conv2d(channels, 1, 1)

    def forward(self, features: Tensor) -> Tensor:
        features = self.upsample(self.context(features))
        phase = torch.atan2(self.imaginary(features), self.real(features))
        return phase.squeeze(1).transpose(1, 2)


class ConsistencyRepairSeparator(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.configuration = SeparatorConfiguration(**kwargs)
        cfg = self.configuration
        self.sample_rate = cfg.sample_rate
        self.spectral = CompressedSpectralTransform(
            cfg.fft_size, cfg.hop_size, cfg.window_size, cfg.compression
        )
        self.visual_encoder = MouthMotionEncoder(cfg.visual_channels)
        self.visual_projector = VisualGridProjector(cfg.visual_channels, self.spectral.frequency_bins)
        settings = RepairSettings(
            shift_radius=cfg.shift_radius,
            leading_rank=cfg.leading_rank,
            reference_momentum=cfg.reference_momentum,
            passthrough=cfg.passthrough,
            trust_ratio=cfg.trust_ratio,
            proposal_scale=cfg.proposal_scale,
        )
        self.av_encoder = RepairControlledAVEncoder(cfg.feature_channels, cfg.av_stages, settings)
        self.phase_magnitude_encoder = nn.Sequential(
            nn.Conv2d(2, cfg.feature_channels, 1),
            ChannelNorm(cfg.feature_channels),
            nn.SiLU(),
            MultiScaleFeatureStack(cfg.feature_channels, layers=4),
            FrequencyReducer(cfg.feature_channels),
        )
        self.av_fusion = nn.Sequential(
            nn.Conv2d(cfg.feature_channels * 3, cfg.feature_channels, 1),
            ChannelNorm(cfg.feature_channels),
            nn.SiLU(),
            FrequencyReducer(cfg.feature_channels),
        )
        self.spectral_fusion = nn.Sequential(
            nn.Conv2d(cfg.feature_channels * 2, cfg.feature_channels, 1),
            ChannelNorm(cfg.feature_channels),
            nn.SiLU(),
        )
        self.context_model = nn.ModuleList(
            TimeFrequencyContext(cfg.feature_channels, cfg.attention_heads)
            for _ in range(cfg.context_blocks)
        )
        self.magnitude_head = MagnitudeHead(cfg.feature_channels, self.spectral.frequency_bins)
        self.phase_head = PhaseHead(cfg.feature_channels)

    def _normalize_waveform(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError("mixture must have shape [T], [B,T], or [B,1,T]")
        return waveform

    def forward(
        self,
        mixture: Tensor,
        visual: Tensor,
        visual_kind: str = "embedding",
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        mixture = self._normalize_waveform(mixture)
        length = mixture.size(-1)
        packet = self.spectral.analyze(mixture)
        frames = packet.compressed_magnitude.size(-1)
        if visual_kind == "frames":
            visual = self.visual_encoder(visual)
        elif visual_kind != "embedding":
            raise ValueError("visual_kind must be 'frames' or 'embedding'")
        visual_grid = self.visual_projector(visual, frames)
        magnitude_grid = packet.compressed_magnitude.transpose(1, 2).unsqueeze(1)
        audio_features, visual_features, traces = self.av_encoder(magnitude_grid, visual_grid)
        av_features = self.av_fusion(
            torch.cat((audio_features, visual_features, audio_features * visual_features), dim=1)
        )
        phase_magnitude = self.phase_magnitude_encoder(
            torch.stack((packet.compressed_magnitude, packet.phase), dim=1).permute(0, 1, 3, 2)
        )
        encoded = self.spectral_fusion(torch.cat((av_features, phase_magnitude), dim=1))
        for block in self.context_model:
            encoded = block(encoded)
        mask = self.magnitude_head(encoded)
        estimated_magnitude = packet.compressed_magnitude * mask
        estimated_phase = self.phase_head(encoded)
        waveform = self.spectral.synthesize(estimated_magnitude, estimated_phase, length).unsqueeze(1)
        if not return_details:
            return waveform
        compressed_cartesian = torch.stack(
            (
                estimated_magnitude * torch.cos(estimated_phase),
                estimated_magnitude * torch.sin(estimated_phase),
            ),
            dim=1,
        )
        return {
            "waveform": waveform,
            "magnitude": estimated_magnitude,
            "phase": estimated_phase,
            "compressed_cartesian": compressed_cartesian,
            "repair_traces": traces,
        }

    def export_state(self) -> dict[str, Any]:
        return {"configuration": asdict(self.configuration), "state_dict": self.state_dict()}

    @classmethod
    def from_state(cls, payload: dict[str, Any], map_location: str | torch.device = "cpu") -> "ConsistencyRepairSeparator":
        configuration = payload["configuration"]
        model = cls(**configuration)
        model.load_state_dict(payload["state_dict"])
        return model.to(map_location)

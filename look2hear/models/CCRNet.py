"""Complete CCRNet audio-visual speech separation backbone.

The network contains the full waveform-to-waveform AVSS path used by CCRNet:
STFT analysis, magnitude/phase encoders, visual alignment, a stage-wise
 audio-visual encoder with Controllable Consistency Repair (CCR), dual-path
 time-frequency Transformer blocks, magnitude/phase decoders, and iSTFT
 synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from .base_av_model import BaseAVModel


Tensor = torch.Tensor


def safe_nan_to_num(
    x: Tensor,
    nan: float = 0.0,
    pos: float = 1.0e4,
    neg: float = -1.0e4,
) -> Tensor:
    """Apply ``nan_to_num`` to real or complex tensors."""
    if torch.is_complex(x):
        real = torch.nan_to_num(x.real, nan=nan, posinf=pos, neginf=neg)
        imag = torch.nan_to_num(x.imag, nan=nan, posinf=pos, neginf=neg)
        return torch.complex(real, imag)
    return torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)


class FeedForwardGRU(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(
            channels,
            channels * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.projection = nn.Linear(channels * 4, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        return self.projection(self.dropout(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = FeedForwardGRU(channels, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.norm1(x)
        residual, _ = self.attention(residual, residual, residual, need_weights=False)
        x = x + self.dropout1(residual)
        residual = self.feed_forward(self.norm2(x))
        return self.norm3(x + self.dropout2(residual))


class TimeFrequencyTransformerBlock(nn.Module):
    """Alternating temporal and frequency sequence modeling."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.time_transformer = TransformerBlock(channels, num_heads=num_heads)
        self.frequency_transformer = TransformerBlock(channels, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frames, bins = x.shape

        time_sequence = x.permute(0, 3, 2, 1).reshape(batch * bins, frames, channels)
        time_sequence = self.time_transformer(time_sequence) + time_sequence

        frequency_sequence = (
            time_sequence.reshape(batch, bins, frames, channels)
            .permute(0, 2, 1, 3)
            .reshape(batch * frames, bins, channels)
        )
        frequency_sequence = (
            self.frequency_transformer(frequency_sequence) + frequency_sequence
        )

        return (
            frequency_sequence.reshape(batch, frames, bins, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


def _same_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(
        int((kernel_size[i] * dilation[i] - dilation[i]) / 2) for i in range(2)
    )


class DenseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int = 4,
        kernel_size: Tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()
        layers = []
        for index in range(depth):
            dilation = (2**index, 1)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels * (index + 1),
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=_same_padding(kernel_size, dilation),
                    ),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.PReLU(channels),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        for layer in self.layers:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class FullDenseEncoder(nn.Module):
    """Feature encoder with dense modeling and frequency downsampling."""

    def __init__(self, input_channels: int, channels: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dense_block = DenseBlock(channels, depth=4)
        self.frequency_downsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.dense_block(x)
        return self.frequency_downsample(x)


class InputProjection(nn.Module):
    """Initial projection used by the audio and visual AV encoder streams."""

    def __init__(self, input_channels: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class FrequencyDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(x)


class LearnableSigmoid2D(nn.Module):
    def __init__(self, frequency_bins: int, beta: float = 2.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(frequency_bins, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class MagnitudeDecoder(nn.Module):
    def __init__(self, channels: int, frequency_bins: int) -> None:
        super().__init__()
        self.dense_block = DenseBlock(channels, depth=4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=(1, 3), stride=(1, 2)),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1),
            nn.Conv2d(1, 1, kernel_size=1),
        )
        self.activation = LearnableSigmoid2D(frequency_bins, beta=2.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(self.dense_block(x))
        # [B, 1, T, F] -> [B, F, T]
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.activation(x)


class PhaseDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dense_block = DenseBlock(channels, depth=4)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=(1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.real_projection = nn.Conv2d(channels, 1, kernel_size=1)
        self.imag_projection = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(self.dense_block(x))
        return torch.atan2(self.imag_projection(x), self.real_projection(x))


@dataclass(frozen=True)
class CCRConfig:
    shift_radius: int = 5
    rank: int = 8
    ema_alpha: float = 0.95
    passthrough: float = 0.25
    trust_ratio: float = 0.25
    step_size: float = 1.0
    eps: float = 1.0e-6
    time_dim: int = 2


@dataclass(frozen=True)
class CCRStageReport:
    score_mean: float
    reference: float
    repair_ratio: float
    audio_accept_ratio: float
    visual_accept_ratio: float
    mean_best_shift: float


def _overlap_for_shift(
    audio: Tensor,
    visual: Tensor,
    shift: int,
    time_dim: int,
) -> Tuple[Tensor, Tensor]:
    length = min(audio.size(time_dim), visual.size(time_dim))
    if abs(shift) >= length:
        index = [slice(None)] * audio.ndim
        index[time_dim] = slice(0, 0)
        return audio[tuple(index)], visual[tuple(index)]

    audio_index = [slice(None)] * audio.ndim
    visual_index = [slice(None)] * visual.ndim
    if shift > 0:
        audio_index[time_dim] = slice(shift, length)
        visual_index[time_dim] = slice(0, length - shift)
    elif shift < 0:
        offset = -shift
        audio_index[time_dim] = slice(0, length - offset)
        visual_index[time_dim] = slice(offset, length)
    else:
        audio_index[time_dim] = slice(0, length)
        visual_index[time_dim] = slice(0, length)
    return audio[tuple(audio_index)], visual[tuple(visual_index)]


def _standardize_tokens(tokens: Tensor, eps: float) -> Tensor:
    mean = tokens.mean(dim=-1, keepdim=True)
    variance = tokens.var(dim=-1, keepdim=True, unbiased=False)
    return (tokens - mean) / torch.sqrt(variance + eps)


@torch.no_grad()
def shift_tolerant_consistency(
    audio: Tensor,
    visual: Tensor,
    config: CCRConfig,
) -> Tuple[Tensor, Tensor]:
    """Top-r singular-value consistency over a local temporal shift window."""
    if audio.ndim != 4 or visual.shape != audio.shape:
        raise ValueError("Consistency scoring expects equal [B, C, T, F] tensors.")

    audio = safe_nan_to_num(audio.detach().float())
    visual = safe_nan_to_num(visual.detach().float())
    candidate_scores: List[Tensor] = []

    for shift in range(-config.shift_radius, config.shift_radius + 1):
        audio_aligned, visual_aligned = _overlap_for_shift(
            audio, visual, shift, config.time_dim
        )
        if audio_aligned.numel() == 0:
            candidate_scores.append(
                audio.new_full((audio.size(0),), float("-inf"))
            )
            continue

        batch, channels = audio_aligned.shape[:2]
        audio_tokens = _standardize_tokens(
            audio_aligned.reshape(batch, channels, -1), config.eps
        )
        visual_tokens = _standardize_tokens(
            visual_aligned.reshape(batch, channels, -1), config.eps
        )
        denominator = float(max(audio_tokens.size(-1) - 1, 1))
        correlation = torch.bmm(
            audio_tokens, visual_tokens.transpose(1, 2)
        ) / denominator
        singular_values = torch.linalg.svdvals(safe_nan_to_num(correlation))
        rank = min(config.rank, singular_values.size(-1))
        candidate_scores.append(singular_values[:, :rank].mean(dim=-1))

    score_matrix = torch.stack(candidate_scores, dim=-1)
    best_score, best_index = score_matrix.max(dim=-1)
    best_shift = best_index.to(torch.long) - config.shift_radius
    return best_score, best_shift


def _bounded_residual(
    residual: Tensor,
    reference: Tensor,
    ratio: float,
    eps: float,
) -> Tensor:
    batch = residual.size(0)
    residual_norm = residual.reshape(batch, -1).norm(p=2, dim=1)
    reference_norm = reference.reshape(batch, -1).norm(p=2, dim=1)
    scale = torch.minimum(
        torch.ones_like(reference_norm),
        ratio * reference_norm / (residual_norm + eps),
    )
    return residual * scale.view(batch, *([1] * (residual.ndim - 1)))


class CCRBlock(nn.Module):
    """Controllable Consistency Repair for one encoder stage."""

    def __init__(self, channels: int, config: CCRConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_gate = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.visual_gate = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.audio_residual = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.visual_residual = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(
        self,
        audio: Tensor,
        visual: Tensor,
        previous_reference: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, CCRStageReport]:
        score_before, best_shift = shift_tolerant_consistency(
            audio, visual, self.config
        )
        stage_mean = score_before.mean().detach()
        if previous_reference is None:
            reference = stage_mean
        else:
            reference = (
                self.config.ema_alpha * previous_reference.detach().to(stage_mean)
                + (1.0 - self.config.ema_alpha) * stage_mean
            )

        repair_mask = score_before < reference
        if not bool(repair_mask.any()):
            report = CCRStageReport(
                score_mean=float(stage_mean.item()),
                reference=float(reference.item()),
                repair_ratio=0.0,
                audio_accept_ratio=0.0,
                visual_accept_ratio=0.0,
                mean_best_shift=float(best_shift.float().mean().item()),
            )
            return audio, visual, reference, report

        audio_gate = torch.sigmoid(
            self.audio_gate(torch.cat([audio, visual], dim=1))
        )
        visual_gate = torch.sigmoid(
            self.visual_gate(torch.cat([visual, audio], dim=1))
        )
        p = self.config.passthrough
        selected_audio = audio * (p + (1.0 - p) * audio_gate)
        selected_visual = visual * (p + (1.0 - p) * visual_gate)

        delta_audio = torch.tanh(
            self.audio_residual(torch.cat([selected_audio, selected_visual], dim=1))
        )
        delta_visual = torch.tanh(
            self.visual_residual(torch.cat([selected_visual, selected_audio], dim=1))
        )
        delta_audio = _bounded_residual(
            delta_audio, audio, self.config.trust_ratio, self.config.eps
        )
        delta_visual = _bounded_residual(
            delta_visual, visual, self.config.trust_ratio, self.config.eps
        )

        audio_candidate = safe_nan_to_num(
            audio + self.config.step_size * delta_audio
        )
        visual_candidate = safe_nan_to_num(
            visual + self.config.step_size * delta_visual
        )

        score_audio, _ = shift_tolerant_consistency(
            audio_candidate, visual, self.config
        )
        accept_audio = repair_mask & (score_audio > score_before)
        audio_output = torch.where(
            accept_audio.view(-1, 1, 1, 1), audio_candidate, audio
        )

        score_visual_base, _ = shift_tolerant_consistency(
            audio_output, visual, self.config
        )
        score_visual_candidate, _ = shift_tolerant_consistency(
            audio_output, visual_candidate, self.config
        )
        accept_visual = repair_mask & (score_visual_candidate > score_visual_base)
        visual_output = torch.where(
            accept_visual.view(-1, 1, 1, 1), visual_candidate, visual
        )

        report = CCRStageReport(
            score_mean=float(stage_mean.item()),
            reference=float(reference.item()),
            repair_ratio=float(repair_mask.float().mean().item()),
            audio_accept_ratio=float(accept_audio.float().mean().item()),
            visual_accept_ratio=float(accept_visual.float().mean().item()),
            mean_best_shift=float(best_shift.float().mean().item()),
        )
        return audio_output, visual_output, reference, report


class AudioVisualCCRStage(nn.Module):
    def __init__(self, channels: int, config: CCRConfig, depth: int = 3) -> None:
        super().__init__()
        self.audio_block = DenseBlock(channels, depth=depth)
        self.visual_block = DenseBlock(channels, depth=depth)
        self.ccr = CCRBlock(channels, config)

    def forward(
        self,
        audio: Tensor,
        visual: Tensor,
        reference: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, CCRStageReport]:
        audio = safe_nan_to_num(self.audio_block(audio) + audio)
        visual = safe_nan_to_num(self.visual_block(visual) + visual)
        return self.ccr(audio, visual, reference)


class VisualAlignment(nn.Module):
    """Align frame-level visual embeddings to the audio T-F layout."""

    def __init__(self, visual_channels: int, frequency_bins: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            visual_channels,
            frequency_bins,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

    def forward(self, visual_embedding: Tensor, target_frames: int) -> Tensor:
        if visual_embedding.ndim != 3:
            raise ValueError("visual_embedding must have shape [B, C_v, T_v].")
        visual = visual_embedding.unsqueeze(-1)
        visual = F.interpolate(
            visual,
            size=(target_frames, 1),
            mode="bilinear",
            align_corners=False,
        )
        # [B, F, T, 1] -> [B, 1, T, F]
        return self.projection(visual).permute(0, 3, 2, 1).contiguous()


class CCRNet(BaseAVModel):
    """Complete waveform-to-waveform CCRNet AVSS model."""

    def __init__(
        self,
        sample_rate: int = 16000,
        visual_channels: int = 512,
        feature_channels: int = 64,
        num_av_stages: int = 4,
        num_tf_blocks: int = 4,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        num_sources: int = 1,
        ccr_shift_radius: int = 5,
        ccr_rank: int = 8,
        ccr_ema_alpha: float = 0.95,
        ccr_passthrough: float = 0.25,
        ccr_trust_ratio: float = 0.25,
        ccr_step_size: float = 1.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        if num_sources != 1:
            raise ValueError("CCRNet currently estimates one target speaker per pass.")
        if n_fft != win_length:
            raise ValueError("This implementation expects n_fft == win_length.")

        self.visual_channels = visual_channels
        self.feature_channels = feature_channels
        self.num_av_stages = num_av_stages
        self.num_tf_blocks = num_tf_blocks
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_sources = num_sources
        self.frequency_bins = n_fft // 2 + 1

        self.register_buffer(
            "stft_window",
            torch.hann_window(win_length),
            persistent=False,
        )

        self.visual_alignment = VisualAlignment(
            visual_channels=visual_channels,
            frequency_bins=self.frequency_bins,
        )
        self.audio_input_encoder = InputProjection(1, feature_channels)
        self.visual_input_encoder = InputProjection(1, feature_channels)
        self.phase_magnitude_encoder = FullDenseEncoder(2, feature_channels)

        ccr_config = CCRConfig(
            shift_radius=ccr_shift_radius,
            rank=ccr_rank,
            ema_alpha=ccr_ema_alpha,
            passthrough=ccr_passthrough,
            trust_ratio=ccr_trust_ratio,
            step_size=ccr_step_size,
            time_dim=2,
        )
        self.av_stages = nn.ModuleList(
            [
                AudioVisualCCRStage(feature_channels, ccr_config, depth=3)
                for _ in range(num_av_stages)
            ]
        )
        self.final_audio_block = DenseBlock(feature_channels, depth=3)
        self.final_visual_block = DenseBlock(feature_channels, depth=3)

        self.av_fusion = nn.Conv2d(2 * feature_channels, feature_channels, 1)
        self.av_context_fusion = nn.Conv2d(3 * feature_channels, feature_channels, 1)
        self.frequency_downsample = FrequencyDownsample(feature_channels)
        self.phase_av_fusion = nn.Conv2d(2 * feature_channels, feature_channels, 1)

        self.tf_transformer = nn.ModuleList(
            [
                TimeFrequencyTransformerBlock(feature_channels, num_heads=4)
                for _ in range(num_tf_blocks)
            ]
        )
        self.magnitude_decoder = MagnitudeDecoder(
            feature_channels,
            frequency_bins=self.frequency_bins,
        )
        self.phase_decoder = PhaseDecoder(feature_channels)

        self._constructor_args = {
            "sample_rate": sample_rate,
            "visual_channels": visual_channels,
            "feature_channels": feature_channels,
            "num_av_stages": num_av_stages,
            "num_tf_blocks": num_tf_blocks,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "num_sources": num_sources,
            "ccr_shift_radius": ccr_shift_radius,
            "ccr_rank": ccr_rank,
            "ccr_ema_alpha": ccr_ema_alpha,
            "ccr_passthrough": ccr_passthrough,
            "ccr_trust_ratio": ccr_trust_ratio,
            "ccr_step_size": ccr_step_size,
        }

    def _analysis(self, waveform: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(waveform),
            center=True,
            onesided=True,
            return_complex=True,
        )
        spectrum = safe_nan_to_num(spectrum)
        magnitude = torch.abs(spectrum).clamp_min(1.0e-12).pow(0.3)
        phase = torch.angle(spectrum)
        complex_features = torch.stack(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)],
            dim=1,
        )
        return complex_features, magnitude, phase

    @staticmethod
    def _power_uncompress(magnitude: Tensor, phase: Tensor) -> Tensor:
        magnitude = magnitude.clamp_min(1.0e-12).pow(1.0 / 0.3)
        return torch.polar(magnitude, phase)

    def forward(
        self,
        input_wav: Tensor,
        visual_embedding: Tensor,
        return_details: bool = False,
    ) -> Union[Tensor, Dict[str, object]]:
        """Separate the target speech.

        Args:
            input_wav: Mixture waveform, ``[B, T]`` or ``[B, 1, T]``.
            visual_embedding: Visual frontend output, ``[B, 512, T_v]`` by default.
            return_details: Return intermediate spectral estimates and CCR reports.

        Returns:
            By default, the separated waveform with shape ``[B, 1, T]``.
        """
        if input_wav.ndim == 1:
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 3:
            if input_wav.size(1) != 1:
                raise ValueError("input_wav must be mono when provided as [B, C, T].")
            input_wav = input_wav[:, 0]
        if input_wav.ndim != 2:
            raise ValueError("input_wav must have shape [T], [B, T], or [B, 1, T].")

        original_length = input_wav.size(-1)
        complex_features, mixture_magnitude, mixture_phase = self._analysis(input_wav)
        frames = mixture_magnitude.size(-1)

        visual_tf = self.visual_alignment(visual_embedding, target_frames=frames)
        audio = self.audio_input_encoder(
            mixture_magnitude.permute(0, 2, 1).unsqueeze(1)
        )
        visual = self.visual_input_encoder(visual_tf)
        phase_magnitude = self.phase_magnitude_encoder(
            torch.cat(
                [
                    mixture_magnitude.permute(0, 2, 1).unsqueeze(1),
                    mixture_phase.permute(0, 2, 1).unsqueeze(1),
                ],
                dim=1,
            )
        )

        reference: Optional[Tensor] = None
        stage_reports: List[CCRStageReport] = []
        for stage in self.av_stages:
            audio, visual, reference, report = stage(audio, visual, reference)
            audio = safe_nan_to_num(audio)
            visual = safe_nan_to_num(visual)
            stage_reports.append(report)

        audio = safe_nan_to_num(self.final_audio_block(audio) + audio)
        visual = safe_nan_to_num(self.final_visual_block(visual) + visual)

        av_fused = safe_nan_to_num(
            self.av_fusion(torch.cat([audio, visual], dim=1))
        )
        av_context = safe_nan_to_num(
            self.av_context_fusion(torch.cat([audio, visual, av_fused], dim=1))
        )
        av_context = safe_nan_to_num(self.frequency_downsample(av_context))
        encoded = safe_nan_to_num(
            self.phase_av_fusion(torch.cat([av_context, phase_magnitude], dim=1))
        )

        for block in self.tf_transformer:
            encoded = safe_nan_to_num(block(encoded))

        magnitude_mask = safe_nan_to_num(self.magnitude_decoder(encoded))
        estimated_magnitude = safe_nan_to_num(mixture_magnitude * magnitude_mask)
        estimated_phase = safe_nan_to_num(
            self.phase_decoder(encoded).squeeze(1).permute(0, 2, 1)
        )
        estimated_compressed_complex = torch.stack(
            [
                estimated_magnitude * torch.cos(estimated_phase),
                estimated_magnitude * torch.sin(estimated_phase),
            ],
            dim=1,
        )
        estimated_spectrum = safe_nan_to_num(
            self._power_uncompress(estimated_magnitude, estimated_phase)
        )
        estimated_waveform = torch.istft(
            estimated_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(input_wav),
            center=True,
            onesided=True,
            length=original_length,
        )
        estimated_waveform = safe_nan_to_num(estimated_waveform).unsqueeze(1)

        if not return_details:
            return estimated_waveform

        return {
            "waveform": estimated_waveform,
            "magnitude": estimated_magnitude,
            "phase": estimated_phase,
            "complex_spectrum": estimated_spectrum,
            "compressed_complex": estimated_compressed_complex,
            "mixture_magnitude": mixture_magnitude,
            "mixture_phase": mixture_phase,
            "ccr_reports": stage_reports,
        }

    def get_model_args(self) -> Dict[str, object]:
        return dict(self._constructor_args)

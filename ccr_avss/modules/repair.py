"""Consistency measurement, triggering, and verified cross-modal repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

Tensor = torch.Tensor


@dataclass(frozen=True)
class RepairSettings:
    shift_radius: int = 5
    leading_rank: int = 8
    reference_momentum: float = 0.95
    passthrough: float = 0.25
    trust_ratio: float = 0.25
    proposal_scale: float = 1.0
    epsilon: float = 1.0e-6


@dataclass
class RepairTrace:
    mean_score: float
    reference: float
    trigger_fraction: float
    accepted_audio_fraction: float
    accepted_visual_fraction: float
    average_shift: float


def _shifted_overlap(first: Tensor, second: Tensor, shift: int) -> tuple[Tensor, Tensor]:
    length = min(first.size(2), second.size(2))
    if abs(shift) >= length:
        return first[:, :, :0], second[:, :, :0]
    if shift > 0:
        return first[:, :, shift:length], second[:, :, : length - shift]
    if shift < 0:
        offset = -shift
        return first[:, :, : length - offset], second[:, :, offset:length]
    return first[:, :, :length], second[:, :, :length]


def _normalize_tokens(tokens: Tensor, epsilon: float) -> Tensor:
    tokens = tokens.flatten(start_dim=2)
    mean = tokens.mean(dim=-1, keepdim=True)
    variance = tokens.var(dim=-1, unbiased=False, keepdim=True)
    return (tokens - mean) * torch.rsqrt(variance + epsilon)


class ShiftTolerantSubspaceScore(nn.Module):
    def __init__(self, settings: RepairSettings) -> None:
        super().__init__()
        self.settings = settings

    def forward(self, audio: Tensor, visual: Tensor) -> tuple[Tensor, Tensor]:
        per_shift: list[Tensor] = []
        shifts = range(-self.settings.shift_radius, self.settings.shift_radius + 1)
        for shift in shifts:
            aligned_audio, aligned_visual = _shifted_overlap(audio, visual, shift)
            if aligned_audio.numel() == 0:
                score = audio.new_full((audio.size(0),), -torch.inf)
            else:
                x = _normalize_tokens(aligned_audio, self.settings.epsilon)
                y = _normalize_tokens(aligned_visual, self.settings.epsilon)
                denominator = max(x.size(-1) - 1, 1)
                correlation = torch.bmm(x, y.transpose(1, 2)) / denominator
                singular_values = torch.linalg.svdvals(correlation.float()).to(correlation.dtype)
                rank = min(self.settings.leading_rank, singular_values.size(-1))
                score = singular_values[:, :rank].mean(dim=-1)
            per_shift.append(score)
        score_table = torch.stack(per_shift, dim=-1)
        best_score, best_index = score_table.max(dim=-1)
        best_shift = best_index.to(audio.dtype) - float(self.settings.shift_radius)
        return best_score, best_shift


class VerifiedCrossModalRepair(nn.Module):
    def __init__(self, channels: int, settings: RepairSettings) -> None:
        super().__init__()
        self.settings = settings
        self.score = ShiftTolerantSubspaceScore(settings)
        self.audio_gate = nn.Conv2d(channels * 2, channels, 1)
        self.visual_gate = nn.Conv2d(channels * 2, channels, 1)
        self.audio_proposal = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), nn.Tanh()
        )
        self.visual_proposal = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), nn.Tanh()
        )

    def _bounded(self, proposal: Tensor, anchor: Tensor) -> Tensor:
        batch = proposal.size(0)
        proposal_norm = proposal.flatten(1).norm(dim=-1)
        anchor_norm = anchor.flatten(1).norm(dim=-1)
        permitted = self.settings.trust_ratio * anchor_norm
        scale = torch.minimum(
            torch.ones_like(proposal_norm),
            permitted / (proposal_norm + self.settings.epsilon),
        )
        return proposal * scale.view(batch, 1, 1, 1)

    @staticmethod
    def _select(mask: Tensor, candidate: Tensor, original: Tensor) -> Tensor:
        return torch.where(mask.view(-1, 1, 1, 1), candidate, original)

    def forward(
        self,
        audio: Tensor,
        visual: Tensor,
        previous_reference: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, RepairTrace]:
        current_score, shift = self.score(audio, visual)
        batch_mean = current_score.mean()
        if previous_reference is None:
            reference = batch_mean.detach()
        else:
            momentum = self.settings.reference_momentum
            reference = (momentum * previous_reference + (1.0 - momentum) * batch_mean.detach())

        trigger = current_score < reference
        stacked = torch.cat((audio, visual), dim=1)
        audio_gate = torch.sigmoid(self.audio_gate(stacked))
        visual_gate = torch.sigmoid(self.visual_gate(torch.cat((visual, audio), dim=1)))
        floor = self.settings.passthrough
        selected_audio = audio * (floor + (1.0 - floor) * audio_gate)
        selected_visual = visual * (floor + (1.0 - floor) * visual_gate)

        audio_delta = self.audio_proposal(torch.cat((selected_audio, selected_visual), dim=1))
        visual_delta = self.visual_proposal(torch.cat((selected_visual, selected_audio), dim=1))
        audio_delta = self._bounded(audio_delta, audio)
        visual_delta = self._bounded(visual_delta, visual)

        audio_candidate = audio + self.settings.proposal_scale * audio_delta
        audio_score, _ = self.score(audio_candidate, visual)
        accept_audio = trigger & (audio_score > current_score)
        repaired_audio = self._select(accept_audio, audio_candidate, audio)

        visual_candidate = visual + self.settings.proposal_scale * visual_delta
        baseline_visual_score, _ = self.score(repaired_audio, visual)
        visual_score, _ = self.score(repaired_audio, visual_candidate)
        accept_visual = trigger & (visual_score > baseline_visual_score)
        repaired_visual = self._select(accept_visual, visual_candidate, visual)

        trace = RepairTrace(
            mean_score=float(current_score.detach().mean().cpu()),
            reference=float(reference.detach().cpu()),
            trigger_fraction=float(trigger.float().mean().detach().cpu()),
            accepted_audio_fraction=float(accept_audio.float().mean().detach().cpu()),
            accepted_visual_fraction=float(accept_visual.float().mean().detach().cpu()),
            average_shift=float(shift.detach().mean().cpu()),
        )
        return repaired_audio, repaired_visual, reference, trace

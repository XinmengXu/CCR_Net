from __future__ import annotations

import torch

from ccr_avss.models import ConsistencyRepairSeparator


def compact_model() -> ConsistencyRepairSeparator:
    return ConsistencyRepairSeparator(
        visual_channels=32,
        feature_channels=8,
        av_stages=2,
        context_blocks=1,
        attention_heads=2,
        fft_size=64,
        hop_size=16,
        window_size=64,
        shift_radius=1,
        leading_rank=2,
    )


def test_embedding_forward_shape() -> None:
    model = compact_model().eval()
    mixture = torch.randn(2, 1024)
    visual = torch.randn(2, 32, 12)
    with torch.no_grad():
        estimate = model(mixture, visual, visual_kind="embedding")
    assert estimate.shape == (2, 1, 1024)
    assert torch.isfinite(estimate).all()


def test_detailed_forward_has_repair_traces() -> None:
    model = compact_model()
    mixture = torch.randn(1, 1024, requires_grad=True)
    visual = torch.randn(1, 32, 12)
    output = model(mixture, visual, visual_kind="embedding", return_details=True)
    loss = output["waveform"].square().mean()
    loss.backward()
    assert len(output["repair_traces"]) == 2
    assert mixture.grad is not None

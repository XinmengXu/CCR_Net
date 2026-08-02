from __future__ import annotations

import torch

from ccr_avss.modules.repair import RepairSettings, ShiftTolerantSubspaceScore


def test_score_returns_batch_values_and_shifts() -> None:
    scorer = ShiftTolerantSubspaceScore(RepairSettings(shift_radius=2, leading_rank=3))
    audio = torch.randn(3, 4, 8, 6)
    visual = torch.randn(3, 4, 8, 6)
    score, shift = scorer(audio, visual)
    assert score.shape == (3,)
    assert shift.shape == (3,)
    assert torch.isfinite(score).all()

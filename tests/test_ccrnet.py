import tempfile
from pathlib import Path

import torch

from look2hear.losses import CCRNetLoss
from look2hear.models import CCRNet


def compact_model() -> CCRNet:
    return CCRNet(
        feature_channels=16,
        num_av_stages=2,
        num_tf_blocks=1,
        ccr_shift_radius=1,
        ccr_rank=4,
    )


def test_waveform_forward_and_details():
    model = compact_model().eval()
    mixture = torch.randn(2, 4096)
    visual = torch.randn(2, 512, 8)
    with torch.no_grad():
        output = model(mixture, visual, return_details=True)
    assert output["waveform"].shape == (2, 1, 4096)
    assert output["magnitude"].shape[0] == 2
    assert output["phase"].shape == output["magnitude"].shape
    assert len(output["ccr_reports"]) == 2
    assert torch.isfinite(output["waveform"]).all()


def test_training_objective_backpropagates():
    model = compact_model()
    mixture = torch.randn(2, 4096)
    target = torch.randn(2, 4096)
    visual = torch.randn(2, 512, 8)
    output = model(mixture, visual, return_details=True)
    objective = CCRNetLoss(
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        win_length=model.win_length,
    )
    losses = objective(output, target)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_serialization_round_trip():
    model = compact_model().eval()
    mixture = torch.randn(1, 2048)
    visual = torch.randn(1, 512, 6)
    with torch.no_grad():
        expected = model(mixture, visual)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pth"
        torch.save(model.serialize(), path)
        restored = CCRNet.from_pretrain(path).eval()
        with torch.no_grad():
            actual = restored(mixture, visual)
    assert torch.allclose(expected, actual, atol=1.0e-5, rtol=1.0e-5)

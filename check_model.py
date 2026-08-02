"""Shape and gradient check for the complete CCRNet backbone."""

import torch

from look2hear.models import CCRNet


def main() -> None:
    model = CCRNet()
    mixture = torch.randn(1, 16000, requires_grad=True)
    visual_embedding = torch.randn(1, 512, 25)
    output = model(mixture, visual_embedding, return_details=True)
    waveform = output["waveform"]
    assert waveform.shape == (1, 1, 16000)
    assert torch.isfinite(waveform).all()
    waveform.square().mean().backward()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"CCRNet output: {tuple(waveform.shape)}")
    print(f"Trainable parameters: {trainable / 1e6:.3f} M")
    print(f"CCR stages: {len(output['ccr_reports'])}")


if __name__ == "__main__":
    main()

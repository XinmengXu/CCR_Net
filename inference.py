"""Run CCRNet separation with a mixture waveform and preprocessed mouth frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from look2hear.models import CCRNet
from look2hear.videomodels import ResNetVideoModel


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = torch.from_numpy(np.asarray(waveform)).float()
    if source_rate != sample_rate:
        target_length = int(round(waveform.numel() * sample_rate / source_rate))
        waveform = F.interpolate(
            waveform.view(1, 1, -1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return waveform


def load_mouth_frames(path: str) -> torch.Tensor:
    frames = np.load(path)
    if isinstance(frames, np.lib.npyio.NpzFile):
        frames = frames["data"]
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    if frames.ndim != 3:
        raise ValueError("Mouth frames must have shape [T, H, W].")
    frames = torch.from_numpy(frames)
    if frames.max() > 2.0:
        frames = frames / 255.0
    return frames.unsqueeze(0).unsqueeze(0)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CCRNet.from_pretrain(args.checkpoint).to(device).eval()
    video_frontend = ResNetVideoModel(pretrain=args.visual_checkpoint).to(device).eval()

    mixture = load_audio(args.mixture, model.sample_rate).unsqueeze(0).to(device)
    mouth = load_mouth_frames(args.mouth).to(device)

    with torch.no_grad():
        visual_embedding = video_frontend(mouth)
        estimate = model(mixture, visual_embedding)[0, 0].cpu().numpy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, estimate, model.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Serialized CCRNet .pth file.")
    parser.add_argument("--visual-checkpoint", required=True, help="Pretrained visual frontend checkpoint.")
    parser.add_argument("--mixture", required=True, help="Input mixture waveform.")
    parser.add_argument("--mouth", required=True, help="Preprocessed mouth frames in .npy or .npz format.")
    parser.add_argument("--output", required=True, help="Output waveform path.")
    parser.add_argument("--device", default=None)
    main(parser.parse_args())

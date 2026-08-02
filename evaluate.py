"""Evaluate a trained CCRNet checkpoint on the configured AVSS test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from look2hear.datas import AVSpeechDynamicDataModule
from look2hear.models import CCRNet
from look2hear.videomodels import ResNetVideoModel


def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(-1, keepdim=True) * target / (target.square().sum(-1, keepdim=True) + eps)
    noise = estimate - projection
    return 10.0 * torch.log10(projection.square().sum(-1) / (noise.square().sum(-1) + eps) + eps)


def sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    error = target - estimate
    return 10.0 * torch.log10(target.square().sum(-1) / (error.square().sum(-1) + eps) + eps)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CCRNet.from_pretrain(args.checkpoint).to(device).eval()
    video_frontend = ResNetVideoModel(**config["video_frontend"]).to(device).eval()

    data = AVSpeechDynamicDataModule(**config["data"])
    data.setup()
    loader = data.test_dataloader()

    rows = []
    with torch.no_grad():
        for mixture, target, mouth, key in tqdm(loader, desc="Evaluating"):
            mixture = mixture.to(device)
            target = target.to(device)
            if target.ndim == 3:
                target = target[:, 0]
            if mouth.ndim == 4:
                mouth = mouth.unsqueeze(1)
            mouth = mouth.to(device=device, dtype=mixture.dtype)
            visual_embedding = video_frontend(mouth)
            estimate = model(mixture, visual_embedding)[:, 0]

            sisnri = si_snr(estimate, target) - si_snr(mixture, target)
            sdri = sdr(estimate, target) - sdr(mixture, target)
            for index, name in enumerate(key):
                rows.append({"key": name, "SI-SNRi": float(sisnri[index]), "SDRi": float(sdri[index])})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["key", "SI-SNRi", "SDRi"])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        print(f"SI-SNRi: {sum(row['SI-SNRi'] for row in rows) / len(rows):.3f} dB")
        print(f"SDRi: {sum(row['SDRi'] for row in rows) / len(rows):.3f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="evaluation.csv")
    parser.add_argument("--device", default=None)
    main(parser.parse_args())

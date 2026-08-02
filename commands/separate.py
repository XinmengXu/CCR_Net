"""Separate one mixture using mouth frames or visual embeddings."""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
import torch

from ccr_avss.models import ConsistencyRepairSeparator
from ccr_avss.utilities import read_configuration


def entrypoint() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--visual", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = read_configuration(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ConsistencyRepairSeparator.from_state(payload["model"], device).eval()
    audio, sample_rate = sf.read(args.mixture, dtype="float32", always_2d=False)
    if sample_rate != cfg["data"]["sample_rate"]:
        raise ValueError("mixture sample rate does not match the configuration")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    visual = torch.from_numpy(np.asarray(np.load(args.visual), dtype=np.float32)).unsqueeze(0).to(device)
    mixture = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        estimate = model(mixture, visual, cfg["data"]["visual_kind"])[0, 0].cpu().numpy()
    sf.write(args.output, estimate, sample_rate)


if __name__ == "__main__":
    entrypoint()

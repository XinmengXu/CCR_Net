"""Evaluate SI-SNRi and SDRi on a JSONL test manifest."""

from __future__ import annotations

import argparse
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccr_avss.data import AudioVisualManifestDataset, collate_examples
from ccr_avss.models import ConsistencyRepairSeparator
from ccr_avss.training.metrics import sdr, si_snr
from ccr_avss.utilities import read_configuration


def entrypoint() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="evaluation.csv")
    args = parser.parse_args()
    cfg = read_configuration(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ConsistencyRepairSeparator.from_state(payload["model"], device).eval()
    data_cfg = cfg["data"]
    dataset = AudioVisualManifestDataset(data_cfg["test_manifest"], data_cfg["sample_rate"], None, data_cfg["visual_kind"], False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_examples)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader):
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            estimate = model(mixture, batch["visual"].to(device), data_cfg["visual_kind"])[:, 0]
            sisnri = si_snr(estimate, target) - si_snr(mixture, target)
            sdri = sdr(estimate, target) - sdr(mixture, target)
            rows.append({"id": batch["id"][0], "SI-SNRi": float(sisnri.item()), "SDRi": float(sdri.item())})
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "SI-SNRi", "SDRi"])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        print(f"SI-SNRi={sum(row['SI-SNRi'] for row in rows)/len(rows):.3f} dB")
        print(f"SDRi={sum(row['SDRi'] for row in rows)/len(rows):.3f} dB")


if __name__ == "__main__":
    entrypoint()

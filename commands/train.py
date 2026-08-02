"""Train the end-to-end separator with a pure PyTorch loop."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccr_avss.data import AudioVisualManifestDataset, collate_examples
from ccr_avss.models import ConsistencyRepairSeparator
from ccr_avss.training.checkpoints import save_checkpoint
from ccr_avss.training.objective import SeparationObjective
from ccr_avss.utilities import read_configuration, seed_all


def run(configuration: dict) -> None:
    seed_all(int(configuration.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = configuration["data"]
    train_set = AudioVisualManifestDataset(
        data_cfg["train_manifest"], data_cfg["sample_rate"], data_cfg.get("segment_seconds", 4.0), data_cfg["visual_kind"], True
    )
    valid_set = AudioVisualManifestDataset(
        data_cfg["valid_manifest"], data_cfg["sample_rate"], data_cfg.get("segment_seconds", 4.0), data_cfg["visual_kind"], False
    )
    train_loader = DataLoader(train_set, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=data_cfg.get("workers", 4), collate_fn=collate_examples, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=data_cfg.get("valid_batch_size", 1), shuffle=False, num_workers=data_cfg.get("workers", 4), collate_fn=collate_examples, pin_memory=True)

    model = ConsistencyRepairSeparator(**configuration["model"]).to(device)
    objective = SeparationObjective(
        fft_size=configuration["model"].get("fft_size", 512),
        hop_size=configuration["model"].get("hop_size", 256),
        window_size=configuration["model"].get("window_size", 512),
        compression=configuration["model"].get("compression", 0.3),
    ).to(device)
    optim_cfg = configuration["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["learning_rate"], weight_decay=optim_cfg.get("weight_decay", 1.0e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(configuration["experiment"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(int(optim_cfg["epochs"])):
        model.train()
        progress = tqdm(train_loader, desc=f"train {epoch + 1}")
        for batch in progress:
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            visual = batch["visual"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(mixture, visual, data_cfg["visual_kind"], return_details=True)
                losses = objective(prediction, target)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.get("gradient_clip", 5.0))
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=float(losses["total"].detach()))

        model.eval()
        validation = 0.0
        count = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"valid {epoch + 1}"):
                prediction = model(batch["mixture"].to(device), batch["visual"].to(device), data_cfg["visual_kind"], return_details=True)
                loss = objective(prediction, batch["target"].to(device))["total"]
                validation += float(loss)
                count += 1
        validation /= max(count, 1)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, min(best, validation))
        if validation < best:
            best = validation
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1, best)
        print(f"epoch={epoch + 1} validation={validation:.6f} best={best:.6f}")


def entrypoint() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(read_configuration(args.config))


if __name__ == "__main__":
    entrypoint()

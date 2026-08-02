"""Checkpoint serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_value: float) -> None:
    payload: dict[str, Any] = {
        "model": model.export_state(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_value": float(best_value),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)

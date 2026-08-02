"""Base class and checkpoint helpers for audio-visual separation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


class BaseAVModel(nn.Module):
    def __init__(self, sample_rate: int, in_chan: int = 1) -> None:
        super().__init__()
        self._sample_rate = int(sample_rate)
        self._in_chan = int(in_chan)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @classmethod
    def from_pretrain(cls, checkpoint_path: str | Path, **overrides: Any):
        """Restore a serialized model produced by :meth:`serialize`."""
        from . import get

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_class = get(checkpoint.get("model_name", cls.__name__))
        model_args = dict(checkpoint.get("model_args", {}))
        model_args.update(overrides)
        model = model_class(**model_args)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model

    def serialize(self) -> Dict[str, Any]:
        return {
            "model_name": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "model_args": self.get_model_args(),
            "software_versions": {"torch": str(torch.__version__)},
        }

    def get_model_args(self) -> Dict[str, Any]:
        raise NotImplementedError

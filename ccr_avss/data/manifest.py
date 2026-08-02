"""JSONL manifest dataset for audio-visual target speech separation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

Tensor = torch.Tensor


class AudioVisualManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str,
        sample_rate: int = 16000,
        segment_seconds: float | None = 4.0,
        visual_kind: str = "embedding",
        training: bool = False,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.segment_samples = None if segment_seconds is None else int(segment_seconds * sample_rate)
        self.visual_kind = visual_kind
        self.training = training
        with open(manifest, "r", encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if not self.records:
            raise ValueError(f"manifest is empty: {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _read_audio(self, path: str) -> Tensor:
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if rate != self.sample_rate:
            raise ValueError(f"expected {self.sample_rate} Hz but found {rate} Hz in {path}")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return torch.from_numpy(np.asarray(audio, dtype=np.float32))

    def _crop(self, mixture: Tensor, target: Tensor, visual: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.segment_samples is None:
            return mixture, target, visual
        length = min(mixture.numel(), target.numel())
        if length < self.segment_samples:
            pad = self.segment_samples - length
            mixture = torch.nn.functional.pad(mixture[:length], (0, pad))
            target = torch.nn.functional.pad(target[:length], (0, pad))
            return mixture, target, visual
        start = random.randint(0, length - self.segment_samples) if self.training else 0
        stop = start + self.segment_samples
        if visual.ndim >= 2:
            time_axis = -1 if self.visual_kind == "embedding" else 0
            total_frames = visual.shape[time_axis]
            begin_ratio = start / max(length, 1)
            end_ratio = stop / max(length, 1)
            frame_start = int(begin_ratio * total_frames)
            frame_stop = max(frame_start + 1, int(end_ratio * total_frames))
            slices = [slice(None)] * visual.ndim
            slices[time_axis] = slice(frame_start, min(frame_stop, total_frames))
            visual = visual[tuple(slices)]
        return mixture[start:stop], target[start:stop], visual

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        mixture = self._read_audio(record["mixture"])
        target = self._read_audio(record["target"])
        visual_array = np.load(record["visual"])
        visual = torch.from_numpy(np.asarray(visual_array, dtype=np.float32))
        mixture, target, visual = self._crop(mixture, target, visual)
        return {"id": record.get("id", Path(record["mixture"]).stem), "mixture": mixture, "target": target, "visual": visual}


def collate_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    minimum_audio = min(item["mixture"].numel() for item in batch)
    minimum_visual = min(item["visual"].shape[-1] if item["visual"].ndim == 2 else item["visual"].shape[0] for item in batch)
    mixtures = torch.stack([item["mixture"][:minimum_audio] for item in batch])
    targets = torch.stack([item["target"][:minimum_audio] for item in batch])
    visuals = []
    for item in batch:
        visual = item["visual"]
        visuals.append(visual[..., :minimum_visual] if visual.ndim == 2 else visual[:minimum_visual])
    return {"id": [item["id"] for item in batch], "mixture": mixtures, "target": targets, "visual": torch.stack(visuals)}

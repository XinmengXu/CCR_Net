"""PyTorch Lightning training system for CCRNet."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from look2hear.losses import CCRNetLoss


class AudioVisualLightningModule(pl.LightningModule):
    def __init__(
        self,
        audio_model: nn.Module,
        video_model: nn.Module,
        learning_rate: float = 5.0e-5,
        weight_decay: float = 1.0e-4,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        freeze_video_frontend: bool = True,
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.freeze_video_frontend = freeze_video_frontend
        self.objective = CCRNetLoss(
            n_fft=audio_model.n_fft,
            hop_length=audio_model.hop_length,
            win_length=audio_model.win_length,
        )
        self.save_hyperparameters(ignore=["audio_model", "video_model"])

        if freeze_video_frontend:
            for parameter in self.video_model.parameters():
                parameter.requires_grad = False
            self.video_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_video_frontend:
            self.video_model.eval()
        return self

    def _visual_features(self, mouth: torch.Tensor, waveform: torch.Tensor) -> torch.Tensor:
        if mouth.ndim == 4:
            mouth = mouth.unsqueeze(1)
        if mouth.ndim != 5:
            raise ValueError("mouth must have shape [B, T, H, W] or [B, 1, T, H, W].")
        mouth = mouth.to(device=waveform.device, dtype=waveform.dtype)
        if self.freeze_video_frontend:
            with torch.no_grad():
                return self.video_model(mouth)
        return self.video_model(mouth)

    def forward(self, waveform: torch.Tensor, mouth: torch.Tensor, return_details: bool = False):
        visual_embedding = self._visual_features(mouth, waveform)
        return self.audio_model(
            waveform,
            visual_embedding,
            return_details=return_details,
        )

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        mixture, target, mouth, _ = batch
        output = self(mixture, mouth, return_details=True)
        losses = self.objective(output, target)
        batch_size = mixture.size(0)
        self.log(
            f"{stage}/loss",
            losses["total"],
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name in ("si_snr", "magnitude", "complex", "phase"):
            self.log(
                f"{stage}/{name}",
                losses[name],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return losses["total"]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

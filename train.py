"""Train CCRNet on an AVSS dataset described by a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from look2hear.datas import AVSpeechDynamicDataModule
from look2hear.models import CCRNet
from look2hear.system import AudioVisualLightningModule
from look2hear.videomodels import ResNetVideoModel


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main(config: dict) -> None:
    pl.seed_everything(int(config.get("seed", 0)), workers=True)

    data = AVSpeechDynamicDataModule(**config["data"])
    data.setup()

    model = CCRNet(**config["model"])
    video_frontend = ResNetVideoModel(**config["video_frontend"])
    system = AudioVisualLightningModule(
        audio_model=model,
        video_model=video_frontend,
        **config["optimization"],
    )

    experiment_dir = Path(config["experiment"]["output_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    checkpoint = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="ccrnet-{epoch:03d}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint, RichProgressBar()]
    patience = config["trainer"].get("early_stopping_patience")
    if patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=int(patience),
            )
        )

    devices = config["trainer"].get("devices", "auto")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = "ddp" if accelerator == "gpu" and isinstance(devices, list) and len(devices) > 1 else "auto"
    logger = TensorBoardLogger(
        save_dir=str(experiment_dir / "logs"),
        name="ccrnet",
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices if accelerator == "gpu" else 1,
        strategy=strategy,
        max_epochs=int(config["trainer"]["max_epochs"]),
        accumulate_grad_batches=int(config["trainer"].get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(config["trainer"].get("gradient_clip_val", 5.0)),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(experiment_dir),
        log_every_n_steps=int(config["trainer"].get("log_every_n_steps", 20)),
    )
    trainer.fit(system, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())

    best_checkpoint = checkpoint.best_model_path or checkpoint.last_model_path
    restored = AudioVisualLightningModule.load_from_checkpoint(
        best_checkpoint,
        audio_model=CCRNet(**config["model"]),
        video_model=ResNetVideoModel(**config["video_frontend"]),
    )
    torch.save(restored.audio_model.cpu().serialize(), experiment_dir / "CCRNet.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a CCRNet YAML configuration.")
    arguments = parser.parse_args()
    main(load_config(arguments.config))

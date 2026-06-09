import argparse
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def main(config):
    dataset_cfg = config["datasets"]
    floorplan_size = tuple(dataset_cfg["floorplan_img_size"])
    num_workers = int(config.get("num_workers", 4))

    train_dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split="train",
        floorplan_img_size=floorplan_size,
        pose_aug_params=dataset_cfg.get("pose_aug", {"enable": False}),
        dataset_cfg=dataset_cfg,
    )
    val_split = dataset_cfg.get("val_split", "val")
    val_dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=val_split,
        floorplan_img_size=floorplan_size,
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("val_batch_size", config["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = PoseQueryDiffusionLocalizer(config)
    run_dir = os.path.join("logs", "pose_diffusion_runs", config["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    logger = True
    if config.get("use_wandb", False):
        logger = WandbLogger(
            project=config.get("project_name", "disco_model"),
            name=config["run_name"],
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="{epoch:02d}-{val_1m_recall:.3f}-"
        "{val_best_of_64_1m_recall:.3f}_"
        + timestamp,
        save_top_k=3,
        monitor="val_1m_recall",
        mode="max",
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.get("gpu_ids", 1),
        max_epochs=int(config.get("epochs", 30)),
        callbacks=[checkpoint],
        logger=logger,
        default_root_dir=run_dir,
        log_every_n_steps=int(config.get("log_every_n_steps", 10)),
        precision=config.get("precision", "32-true"),
        gradient_clip_val=float(config.get("gradient_clip_val", 1.0)),
        limit_train_batches=config.get("limit_train_batches", None),
        limit_val_batches=config.get("limit_val_batches", None),
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 1)),
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PoseQueryDiffusion_S3D.yaml")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run_name")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.run_name:
        config["run_name"] = args.run_name
    main(config)

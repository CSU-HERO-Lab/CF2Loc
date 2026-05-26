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
from DisCo_model.orienternet_likelihood import OrienterNetLikelihoodModel


def main(config):
    dataset_cfg = config["datasets"]
    fp_size = tuple(dataset_cfg["floorplan_img_size"])
    num_workers = int(config.get("num_workers", 4))
    pose_aug_cfg = dataset_cfg.get(
        "pose_aug", {"enable": True, "trans_range": 25, "rot_range": 0.26}
    )

    print("Loading Datasets...")
    train_dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split="train",
        floorplan_img_size=fp_size,
        pose_aug_params=pose_aug_cfg,
        dataset_cfg=dataset_cfg,
    )

    val_split = dataset_cfg.get("val_split", "val")
    with open(dataset_cfg["data_splits"], "r", encoding="utf-8") as f:
        available_splits = yaml.safe_load(f) or {}
    if val_split not in available_splits:
        if "test" in available_splits:
            print(f"[WARN] Requested val split '{val_split}' not found; fallback to 'test'.")
            val_split = "test"
        else:
            raise ValueError(
                f"Validation split '{val_split}' not found in {dataset_cfg['data_splits']}. "
                f"Available splits: {list(available_splits.keys())}"
            )
    print(f"Using validation split: {val_split}")

    val_dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=val_split,
        floorplan_img_size=fp_size,
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
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = OrienterNetLikelihoodModel(config)

    project_folder = os.path.join("logs", "orienternet_runs", config["run_name"])
    os.makedirs(project_folder, exist_ok=True)

    logger = True
    if config.get("use_wandb", False):
        logger = WandbLogger(
            project=config.get("project_name", "disco_model"),
            name=config["run_name"],
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_folder, "checkpoints"),
        filename="{epoch:02d}-{val_1m_recall:.3f}_" + timestamp,
        save_top_k=3,
        monitor="val_1m_recall",
        mode="max",
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.get("gpu_ids", 1),
        max_epochs=int(config.get("epochs", 30)),
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=project_folder,
        log_every_n_steps=int(config.get("log_every_n_steps", 10)),
        precision=config.get("precision", "32-true"),
    )

    print("Starting OrienterNet-style dense likelihood training...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="OrienterNet_FLoc.yaml", type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--run_name", default=None, type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.run_name:
        config["run_name"] = args.run_name
    elif not config.get("run_name"):
        config["run_name"] = "orienternet_likelihood_" + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs

    if "dptv2_ckpt_path" not in config:
        config["dptv2_ckpt_path"] = "checkpoints/depth_anything_v2_vits.pth"

    main(config)

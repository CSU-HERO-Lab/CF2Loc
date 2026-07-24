import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger

from RRP_model.RRP_lightning_datamodule import RRPDataModule
from training.RRP_callbacks import ImageLoggerCallback
from training.RRP_lightning_module import RRPLightningModule


def main(config, logger=True):
    pl.seed_everything(int(config.get("seed", 0)), workers=True)
    data_module = RRPDataModule(
        data_config=config["datasets"],
        batch_size=config["batch_size"],
        eval_batch_size=config.get("eval_batch_size"),
        num_workers=config["num_workers"],
    )

    model = RRPLightningModule(
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["project_folder"], "checkpoints"),
        filename="{epoch:02d}-{val_action_loss:.2f}",
        save_top_k=3,
        monitor="val_action_loss",
        mode="min",
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-4)

    image_log_callback = ImageLoggerCallback(
        num_images_log=config.get("num_images_log", 8),
        image_log_freq=config.get("image_log_freq", 10),
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.get("gpu_ids", [0]) if torch.cuda.is_available() else 1,
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.is_available() and len(config.get("gpu_ids", [0])) > 1
            else "auto"
        ),
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, swa_callback, image_log_callback],
        logger=logger,
        default_root_dir=config["project_folder"],
        log_every_n_steps=config["wandb_log_freq"],
    )

    ckpt_path = None
    if "load_run" in config and "load_checkpoint" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        ckpt_path = os.path.join(
            load_project_folder,
            "checkpoints",
            config["load_checkpoint"],
        )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Configured resume checkpoint does not exist: '{ckpt_path}'."
            )
        print(f"Resuming RRP training from: {ckpt_path}")

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    print("RRP training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RRP")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/RRP.yaml",
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        type=str,
        help="Optional experiment name. Overrides run_name in config before appending timestamp.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_run_name = args.exp_name or config.get("run_name", "rrp_model")
    run_name = base_run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
    config["run_name"] = run_name
    project_folder = os.path.join("logs", "rrp_runs", run_name)
    config["project_folder"] = project_folder
    os.makedirs(project_folder, exist_ok=True)

    logger = True
    if config.get("use_wandb", False):
        try:
            logger = WandbLogger(
                project=config.get("project_name", "rrp_model"),
                name=run_name,
                config=config,
            )
        except ImportError:
            print(
                "Warning: use_wandb=true, but wandb is not installed. "
                "Falling back to the default Lightning logger."
            )
            logger = True

    main(config, logger=logger)

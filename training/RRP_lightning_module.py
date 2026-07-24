import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from RRP_model.depth_models import DepthPredModels


class RRPLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.image_log_freq = int(self.config["image_log_freq"])
        self.model = DepthPredModels(
            config=self.config,
            encoder_type=self.config["encoder_type"],
            decoder_type=self.config["decoder_type"],
        )

    def forward(self, func_name, **kwargs):
        return self.model(func_name, **kwargs)

    def training_step(self, batch, batch_idx):
        del batch_idx
        batch_obs_image, _pose, ray, _floorplan_img, _wh = batch
        features = self.model("encode", obs_img=batch_obs_image)
        output = self.model(
            "decoder_train",
            depth_cond=features,
            gt_ray=ray,
        )
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite RRP training loss: {loss.detach().item()}"
            )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        should_log_prediction = (
            self.trainer.is_global_zero
            and self.logger is not None
            and self.image_log_freq > 0
            and (self.global_step + 1) % self.image_log_freq == 0
        )
        if should_log_prediction:
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(
                    "decoder_inference",
                    depth_cond=features,
                )
                action_loss = F.mse_loss(prediction, ray)
                self.logger.log_metrics(
                    {"train_action_loss": action_loss},
                    step=self.global_step,
                )
            self.model.train()
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        batch_obs_image, _pose, ray, _floorplan_img, _wh = batch
        features = self.model("encode", obs_img=batch_obs_image)
        prediction = self.model(
            "decoder_inference",
            depth_cond=features,
        )
        val_loss = F.mse_loss(prediction, ray)
        if not torch.isfinite(val_loss):
            raise FloatingPointError(
                f"Non-finite RRP validation loss: {val_loss.detach().item()}"
            )
        self.log(
            "val_action_loss",
            val_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.config["lr"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=10,
            factor=0.5,
        )

        if self.config.get("warmup", False):
            try:
                from warmup_scheduler import GradualWarmupScheduler
            except ImportError as exc:
                raise ImportError(
                    "RRP warmup requires the optional 'warmup-scheduler' package."
                ) from exc
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=self.config["warmup_epochs"],
                after_scheduler=scheduler,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_action_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, on_epoch=True, logger=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb


def visualize_floorplan_rays(
    model,
    obs_image,
    floorplan_image,
    pose,
    gt_ray,
    wh_tensor,
    config,
    device,
    epoch,
    sample_idx,
):
    obs_image = obs_image.unsqueeze(0).to(device)
    floorplan = floorplan_image.cpu().numpy().transpose(1, 2, 0)
    if floorplan.shape[2] == 1:
        floorplan = floorplan.squeeze(2)
    floorplan = np.ascontiguousarray(floorplan)
    pose = pose.cpu().numpy()

    with torch.no_grad():
        features = model("encode", obs_img=obs_image)
        prediction = model("decoder_inference", depth_cond=features)

    gt_ray = gt_ray.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()
    map_res = float(config.get("datasets", {}).get("map_res", 0.02))
    pixels_per_meter = 1.0 / map_res

    original_width = wh_tensor[0].item()
    original_height = wh_tensor[1].item()
    resized_height, resized_width = floorplan.shape[:2]
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    agent_x = pose[0] * scale_x
    agent_y = pose[1] * scale_y

    ray_count = len(gt_ray)
    focal_length = (ray_count / 2.0) / np.tan(np.deg2rad(80.0) / 2.0)
    image_columns = np.arange(ray_count) - (ray_count - 1) / 2.0
    ray_angles = pose[2] + np.flip(np.arctan2(image_columns, focal_length))

    figure, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(floorplan, cmap="gray")
    for distance, predicted_distance, angle in zip(
        gt_ray,
        prediction,
        ray_angles,
    ):
        for ray_distance, color, alpha in (
            (distance, "green", 0.6),
            (predicted_distance, "red", 0.8),
        ):
            distance_px = ray_distance * pixels_per_meter
            end_x = (pose[0] + distance_px * np.cos(angle)) * scale_x
            end_y = (pose[1] + distance_px * np.sin(angle)) * scale_y
            axis.plot(
                [agent_x, end_x],
                [agent_y, end_y],
                color=color,
                linewidth=0.5,
                alpha=alpha,
            )

    axis.add_artist(plt.Circle((agent_x, agent_y), radius=3, color="blue"))
    axis.set_title(f"Sample {sample_idx} - Rays (epoch {epoch})")
    axis.axis("off")
    wandb.log({f"Validation/Floorplan_Rays_{sample_idx}": wandb.Image(figure)})
    plt.close(figure)


class ImageLoggerCallback(pl.Callback):
    def __init__(self, num_images_log=8, image_log_freq=1000):
        super().__init__()
        self.num_images_log = num_images_log
        self.image_log_freq = image_log_freq

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        if self.image_log_freq <= 0:
            return
        if (trainer.global_step + 1) % self.image_log_freq != 0:
            return
        if not pl_module.config.get("use_wandb", False) or not trainer.is_global_zero:
            return

        obs_image, pose, ray, floorplan_image, wh_tensor = batch
        num_to_log = min(self.num_images_log, obs_image.shape[0])
        pl_module.eval()
        with torch.no_grad():
            for index in range(num_to_log):
                visualize_floorplan_rays(
                    model=pl_module.model,
                    obs_image=obs_image[index],
                    floorplan_image=floorplan_image[index],
                    pose=pose[index],
                    gt_ray=ray[index],
                    wh_tensor=wh_tensor[index],
                    config=pl_module.config,
                    device=pl_module.device,
                    epoch=trainer.current_epoch,
                    sample_idx=index,
                )
        pl_module.train()

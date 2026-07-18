import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_local_refiner import (
    PoseLocalRefinerLightning,
    apply_local_delta_to_pose,
    crop_to_refiner_tensor,
    load_refiner_map_np,
    pose_delta_to_local_m,
    wrap_to_pi,
)
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the two-stage dense local pose refiner."
    )
    parser.add_argument("--config", default="PoseLocalRefiner_S3D_Dense.yaml")
    parser.add_argument("--diffusion_ckpt")
    parser.add_argument(
        "--refiner_ckpt",
        default="checkpoints/release_s3d/s3d_no_sem_dense_refiner_best.ckpt",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        default="paper_assets/refiner_visualizations",
    )
    return parser.parse_args()


def pose_error(pred_pose, gt_pose, map_res):
    xy_m = torch.linalg.norm(pred_pose[:2] - gt_pose[:2]) * map_res
    theta = torch.abs(wrap_to_pi(pred_pose[2] - gt_pose[2]))
    return float(xy_m), float(torch.rad2deg(theta))


def draw_pose(ax, pose, color, label, arrow_px=35.0, linewidth=2.2):
    x, y, theta = [float(value) for value in pose[:3]]
    ax.scatter([x], [y], s=45, color=color, edgecolors="white", linewidths=0.8, zorder=5)
    ax.arrow(
        x,
        y,
        arrow_px * math.cos(theta),
        arrow_px * math.sin(theta),
        width=max(1.0, arrow_px * 0.035),
        head_width=max(5.0, arrow_px * 0.18),
        head_length=max(6.0, arrow_px * 0.2),
        color=color,
        linewidth=linewidth,
        length_includes_head=True,
        zorder=4,
        label=label,
    )


def local_m_to_pixel(local_xy_m, crop_size_m, image_size):
    half = crop_size_m * 0.5
    local_x, local_y = [float(value) for value in local_xy_m]
    pixel_x = (local_y / half + 1.0) * 0.5 * image_size
    pixel_y = (1.0 - local_x / half) * 0.5 * image_size
    return np.array([pixel_x, pixel_y], dtype=np.float32)


@torch.no_grad()
def dense_forward_with_attention(model, obs_img, local_map, candidate_pose, wh):
    image_tokens = model.encode_image(obs_img)
    map_tokens = model.encode_local_map(local_map)
    side = int(round(math.sqrt(map_tokens.shape[1])))
    coordinates = model.build_dense_coordinates(
        side,
        side,
        map_tokens.device,
        map_tokens.dtype,
    )
    image_global = image_tokens.mean(dim=1)
    pose_features = model.encode_candidate_pose(candidate_pose, wh)
    conditioned_map_tokens = map_tokens
    conditioned_map_tokens = conditioned_map_tokens + model.image_global_projection(
        image_global
    ).unsqueeze(1)
    conditioned_map_tokens = conditioned_map_tokens + model.pose_mlp(
        pose_features
    ).unsqueeze(1)
    attended, attention = model.dense_image_attn(
        query=conditioned_map_tokens,
        key=image_tokens,
        value=image_tokens,
        need_weights=True,
        average_attn_weights=False,
    )
    fused_tokens = model.dense_image_attn_norm(
        conditioned_map_tokens + model.dropout(attended)
    )
    fused_tokens = model.dense_ffn_norm(
        fused_tokens + model.dropout(model.dense_ffn(fused_tokens))
    )
    heatmap_logits = model.heatmap_head(fused_tokens).squeeze(-1)
    heatmap_prob = F.softmax(
        heatmap_logits / max(model.dense_temperature, 1e-6), dim=-1
    )
    local_coordinates_m = model.coordinates_to_local_m(
        coordinates, model.crop_size_meters
    )
    delta_xy_m = torch.einsum("bn,nd->bd", heatmap_prob, local_coordinates_m)
    theta_vectors = model.theta_head(fused_tokens)
    theta_vector = torch.einsum("bn,bnd->bd", heatmap_prob, theta_vectors)
    delta_theta = torch.atan2(theta_vector[:, 0], theta_vector[:, 1]).clamp(
        -model.max_delta_theta, model.max_delta_theta
    )
    pooled_features = torch.einsum("bn,bnd->bd", heatmap_prob, fused_tokens)
    score_logit = model.dense_score_head(pooled_features).squeeze(-1)
    return {
        "image_tokens": image_tokens,
        "attention": attention,
        "heatmap_prob": heatmap_prob,
        "local_coordinates_m": local_coordinates_m,
        "delta_xy_m": delta_xy_m,
        "delta_theta": delta_theta,
        "score": torch.sigmoid(score_logit),
        "grid_side": side,
    }


def attention_overlay(obs_rgb, attention_6x40):
    height, width = obs_rgb.shape[:2]
    attention = cv2.resize(
        attention_6x40.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    attention -= attention.min()
    attention /= max(float(attention.max()), 1e-8)
    return attention


def map_for_display(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def make_figure(
    index,
    obs_rgb,
    floorplan_rgb,
    local_rgb,
    gt_pose,
    candidate_pose,
    refined_pose,
    heatmap,
    argmax_local_m,
    pred_local_m,
    gt_local_m,
    image_attention,
    score,
    map_res,
    crop_size_m,
):
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(obs_rgb)
    ax.set_title("Observation")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(obs_rgb)
    ax.imshow(image_attention, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    ax.set_title("Image attention from heatmap peak")
    ax.axis("off")

    before_xy, before_theta = pose_error(candidate_pose, gt_pose, map_res)
    after_xy, after_theta = pose_error(refined_pose, gt_pose, map_res)
    ax = axes[0, 2]
    ax.imshow(floorplan_rgb)
    draw_pose(ax, gt_pose, "#21a366", "GT")
    draw_pose(ax, candidate_pose, "#3977d5", "Coarse")
    draw_pose(ax, refined_pose, "#e2473f", "Refined")
    ax.plot(
        [float(candidate_pose[0]), float(refined_pose[0])],
        [float(candidate_pose[1]), float(refined_pose[1])],
        color="#e2473f",
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_xlim(0, floorplan_rgb.shape[1])
    ax.set_ylim(floorplan_rgb.shape[0], 0)
    ax.set_title(
        f"Full map | before {before_xy:.2f}m/{before_theta:.1f}deg"
        f" -> after {after_xy:.2f}m/{after_theta:.1f}deg"
    )
    ax.axis("off")

    points = {
        "Candidate center": (np.array([local_rgb.shape[1] / 2, local_rgb.shape[0] / 2]), "#3977d5", "o"),
        "GT residual": (local_m_to_pixel(gt_local_m, crop_size_m, local_rgb.shape[0]), "#21a366", "*"),
        "Heatmap argmax": (local_m_to_pixel(argmax_local_m, crop_size_m, local_rgb.shape[0]), "#f5a623", "o"),
        "Soft-argmax": (local_m_to_pixel(pred_local_m, crop_size_m, local_rgb.shape[0]), "#e2473f", "x"),
    }

    ax = axes[1, 0]
    ax.imshow(local_rgb)
    for label, (point, color, marker) in points.items():
        ax.scatter(
            [point[0]],
            [point[1]],
            color=color,
            marker=marker,
            s=85 if marker != "*" else 130,
            linewidths=2.0,
            label=label,
            zorder=5,
        )
    center = points["Candidate center"][0]
    pred = points["Soft-argmax"][0]
    ax.arrow(
        center[0],
        center[1],
        pred[0] - center[0],
        pred[1] - center[1],
        color="#e2473f",
        width=1.0,
        head_width=8.0,
        length_includes_head=True,
        zorder=4,
    )
    ax.set_title("Candidate-aligned 5m local crop")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.axis("off")

    heatmap_up = cv2.resize(
        heatmap.astype(np.float32),
        (local_rgb.shape[1], local_rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap_up = np.maximum(heatmap_up, 0.0)
    heatmap_up /= max(float(heatmap_up.max()), 1e-8)
    ax = axes[1, 1]
    ax.imshow(local_rgb)
    ax.imshow(heatmap_up, cmap="turbo", alpha=0.62, vmin=0.0, vmax=1.0)
    for label in ("GT residual", "Heatmap argmax", "Soft-argmax"):
        point, color, marker = points[label]
        ax.scatter(
            [point[0]], [point[1]], color=color, marker=marker, s=100, linewidths=2.0
        )
    ax.set_title("Dense 32x32 position probability")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(heatmap, cmap="turbo")
    ax.set_title(
        "Raw 32x32 heatmap\n"
        f"refiner score={score:.3f}, peak={float(heatmap.max()):.4f}"
    )
    ax.set_xlabel("local-map token x")
    ax.set_ylabel("local-map token y")

    figure.suptitle(f"Dense local refiner | test index {index}", fontsize=16)
    return figure


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["diffusion_val_particles"] = int(args.particles)
    config["diffusion_sample_steps"] = int(args.sample_steps)
    diffusion_ckpt = args.diffusion_ckpt or config["baseline_checkpoint_path"]
    dataset_cfg = config["datasets"]
    map_res = float(dataset_cfg.get("map_res", 0.02))
    representation = dataset_cfg.get(
        "refiner_floorplan_representation",
        dataset_cfg.get("floorplan_representation", "rgb"),
    )
    crop_size_m = float(config.get("refiner_crop_size_meters", 5.0))
    crop_output_size = int(config.get("refiner_crop_output_size", 256))
    oriented = bool(dataset_cfg.get("refiner_oriented_crop", True))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=args.split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )
    if args.indices:
        indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(dataset), size=args.count, replace=False).tolist())

    diffusion = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        diffusion_ckpt,
        config=config,
        map_location="cpu",
    ).to(device).eval()
    lightning_refiner = PoseLocalRefinerLightning.load_from_checkpoint(
        args.refiner_ckpt,
        config=config,
        map_location="cpu",
    ).to(device).eval()
    refiner = lightning_refiner.refiner

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with torch.no_grad():
        for index in indices:
            obs_img, gt_pose, _ray, floorplan_img, wh, *_ = dataset[index]
            item = dataset.data[index]
            obs_batch = obs_img.unsqueeze(0).to(device)
            floorplan_batch = floorplan_img.unsqueeze(0).to(device)
            wh_batch = wh.unsqueeze(0).to(device)
            gt_pose_device = gt_pose.to(device)

            map_tokens, map_coordinates, image_global = diffusion.condition_encoder(
                obs_batch, floorplan_batch
            )
            pose_samples = diffusion.sample_from_context(
                map_tokens,
                map_coordinates,
                image_global,
                wh_batch,
                num_particles=args.particles,
                num_steps=args.sample_steps,
            )
            selected_pose, _density = diffusion.select_pose_mode(pose_samples)
            candidate_pose = selected_pose[0]

            raw_map = load_refiner_map_np(dataset, item["floorplan_image"])
            local_map = crop_to_refiner_tensor(
                raw_map,
                candidate_pose.detach().cpu().numpy(),
                crop_size_meters=crop_size_m,
                map_res=map_res,
                output_size=crop_output_size,
                representation=representation,
                oriented=oriented,
            ).unsqueeze(0).to(device)
            outputs = dense_forward_with_attention(
                refiner,
                obs_batch,
                local_map,
                candidate_pose.unsqueeze(0),
                wh_batch,
            )
            refined_pose = apply_local_delta_to_pose(
                candidate_pose.unsqueeze(0),
                outputs["delta_xy_m"],
                outputs["delta_theta"],
                map_res=map_res,
                wh=wh_batch,
            )[0]
            gt_local_m, _ = pose_delta_to_local_m(
                gt_pose_device, candidate_pose, map_res=map_res
            )
            heatmap_prob = outputs["heatmap_prob"][0]
            peak_index = int(torch.argmax(heatmap_prob).item())
            argmax_local_m = outputs["local_coordinates_m"][peak_index]
            pred_local_m = outputs["delta_xy_m"][0]

            attention = outputs["attention"][0, :, peak_index].mean(dim=0)
            attention = attention.reshape(6, 40).detach().cpu().numpy()
            obs_rgb = np.asarray(Image.open(item["rgb_image"]).convert("RGB"))
            attention_up = attention_overlay(obs_rgb, attention)
            floorplan_rgb = map_for_display(item["floorplan_image"])
            local_display = local_map[0].detach().cpu()
            if local_display.shape[0] == 1:
                local_gray = (local_display[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
                local_rgb = cv2.cvtColor(local_gray, cv2.COLOR_GRAY2RGB)
            else:
                labels = torch.argmax(local_display, dim=0).numpy().astype(np.uint8)
                palette = np.array(
                    [[245, 245, 245], [30, 30, 30], [215, 80, 60], [70, 135, 210], [170, 170, 170]],
                    dtype=np.uint8,
                )
                local_rgb = palette[np.clip(labels, 0, len(palette) - 1)]

            figure = make_figure(
                index=index,
                obs_rgb=obs_rgb,
                floorplan_rgb=floorplan_rgb,
                local_rgb=local_rgb,
                gt_pose=gt_pose.cpu(),
                candidate_pose=candidate_pose.cpu(),
                refined_pose=refined_pose.cpu(),
                heatmap=heatmap_prob.reshape(outputs["grid_side"], outputs["grid_side"]).cpu().numpy(),
                argmax_local_m=argmax_local_m.cpu(),
                pred_local_m=pred_local_m.cpu(),
                gt_local_m=gt_local_m.cpu(),
                image_attention=attention_up,
                score=float(outputs["score"][0]),
                map_res=map_res,
                crop_size_m=crop_size_m,
            )
            output_path = output_dir / f"refiner_test_{index:05d}.png"
            figure.savefig(output_path, dpi=170, bbox_inches="tight")
            plt.close(figure)
            saved.append(output_path)
            print(output_path.resolve())

    montage_images = [np.asarray(Image.open(path).convert("RGB")) for path in saved]
    if montage_images:
        thumb_width = 1100
        thumbnails = []
        for image in montage_images:
            scale = thumb_width / image.shape[1]
            thumbnails.append(
                cv2.resize(
                    image,
                    (thumb_width, int(round(image.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            )
        montage = np.concatenate(thumbnails, axis=0)
        montage_path = output_dir / "refiner_random_samples_montage.jpg"
        Image.fromarray(montage).save(montage_path, quality=92)
        print(montage_path.resolve())


if __name__ == "__main__":
    main()

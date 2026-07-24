import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_local_refiner import crop_to_refiner_tensor, load_refiner_map_np
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Export top-K refiner local-map crops.")
    parser.add_argument("--config", default="configs/PoseLocalRefiner_S3D_Dense.yaml")
    parser.add_argument("--diffusion_ckpt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene", default="scene_03400")
    parser.add_argument("--observation", default="004")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default="paper_assets/scene_03400_observation_004_top8_local_crops",
    )
    return parser.parse_args()


def find_dataset_index(dataset, scene, observation):
    expected = f"/{scene}/imgs/{observation}.png"
    matches = [
        index
        for index, item in enumerate(dataset.data)
        if str(item["rgb_image"]).replace("\\", "/").endswith(expected)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {expected}, found {matches}")
    return matches[0]


def draw_pose(ax, pose, color, label, length=30.0):
    x, y, theta = [float(value) for value in pose[:3]]
    ax.scatter(x, y, s=40, color=color, edgecolors="white", linewidths=0.7, zorder=4)
    ax.arrow(
        x,
        y,
        length * math.cos(theta),
        length * math.sin(theta),
        color=color,
        width=max(1.0, length * 0.03),
        head_width=max(5.0, length * 0.16),
        length_includes_head=True,
        zorder=3,
        label=label,
    )


def align_map_input_mode_with_checkpoint(config, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    weight = state_dict["condition_encoder.map_encoder.conv1.weight"]
    channels_to_mode = {1: "gray", 3: "gray_edges", 5: "semantic_onehot"}
    input_channels = int(weight.shape[1])
    if input_channels not in channels_to_mode:
        raise ValueError(f"Unsupported checkpoint map input channels: {input_channels}")
    config["diffusion_map_input_mode"] = channels_to_mode[input_channels]


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    diffusion_ckpt = args.diffusion_ckpt or config["baseline_checkpoint_path"]
    align_map_input_mode_with_checkpoint(config, diffusion_ckpt)
    dataset_cfg = config["datasets"]
    representation = dataset_cfg.get(
        "refiner_floorplan_representation",
        dataset_cfg.get("floorplan_representation", "gray"),
    )
    crop_size_m = float(config.get("refiner_crop_size_meters", 5.0))
    crop_output_size = int(config.get("refiner_crop_output_size", 256))
    map_res = float(dataset_cfg.get("map_res", 0.02))
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
    index = find_dataset_index(dataset, args.scene, args.observation)
    obs_img, gt_pose, _ray, floorplan_img, wh, *_ = dataset[index]
    item = dataset.data[index]

    diffusion = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        diffusion_ckpt,
        config=config,
        map_location="cpu",
    ).to(device).eval()
    with torch.no_grad():
        obs_batch = obs_img.unsqueeze(0).to(device)
        floorplan_batch = floorplan_img.unsqueeze(0).to(device)
        wh_batch = wh.unsqueeze(0).to(device)
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
        _selected_pose, density = diffusion.select_pose_mode(pose_samples)

    top_k = min(args.top_k, pose_samples.shape[1])
    top_indices = torch.topk(density[0], k=top_k).indices
    top_poses = pose_samples[0, top_indices].detach().cpu()
    top_density = density[0, top_indices].detach().cpu()
    raw_map = load_refiner_map_np(dataset, item["floorplan_image"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    metadata = []
    for rank, (pose, score, particle_index) in enumerate(
        zip(top_poses, top_density, top_indices.detach().cpu()), start=1
    ):
        crop = crop_to_refiner_tensor(
            raw_map,
            pose.numpy(),
            crop_size_meters=crop_size_m,
            map_res=map_res,
            output_size=crop_output_size,
            representation=representation,
            oriented=oriented,
        )
        if crop.shape[0] != 1:
            raise ValueError("This exporter currently expects a one-channel binary/gray map.")
        crop_u8 = (crop[0].numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        output_path = output_dir / f"top{rank:02d}_local_crop.png"
        Image.fromarray(crop_u8, mode="L").save(output_path)
        crops.append(crop_u8)
        metadata.append(
            {
                "rank": rank,
                "particle_index": int(particle_index),
                "density": float(score),
                "pose_x_px": float(pose[0]),
                "pose_y_px": float(pose[1]),
                "pose_theta_rad": float(pose[2]),
                "path": str(output_path.resolve()),
            }
        )

    figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for rank, (ax, crop, score) in enumerate(zip(axes.flat, crops, top_density), start=1):
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Top-{rank} | density {float(score):.3f}")
        ax.axis("off")
    montage_path = output_dir / "top8_local_crops_montage.png"
    figure.savefig(montage_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    with Image.open(item["floorplan_image"]) as image:
        floorplan_rgb = np.asarray(image.convert("RGB"))
    figure, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(floorplan_rgb)
    draw_pose(ax, gt_pose, "#22a06b", "GT", length=35.0)
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, top_k))
    for rank, (pose, color) in enumerate(zip(top_poses, colors), start=1):
        draw_pose(ax, pose, color, f"Top-{rank}", length=24.0)
        ax.text(float(pose[0]) + 5, float(pose[1]) - 5, str(rank), color=color, fontsize=9)
    ax.set_xlim(0, floorplan_rgb.shape[1])
    ax.set_ylim(floorplan_rgb.shape[0], 0)
    ax.set_title("Top-8 diffusion particles used to crop local maps")
    ax.axis("off")
    overview_path = output_dir / "top8_candidates_on_full_map.png"
    figure.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    payload = {
        "scene": args.scene,
        "observation": args.observation,
        "dataset_index": index,
        "seed": args.seed,
        "particles": args.particles,
        "sample_steps": args.sample_steps,
        "crop_size_m": crop_size_m,
        "candidates": metadata,
    }
    metadata_path = output_dir / "top8_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(montage_path.resolve())
    print(overview_path.resolve())
    print(metadata_path.resolve())


if __name__ == "__main__":
    main()

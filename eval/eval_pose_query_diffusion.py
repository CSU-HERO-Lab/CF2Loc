import argparse
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the first-stage pose diffusion localizer."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--subset-fraction", type=float)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument(
        "--flip-target-map-vertical",
        action="store_true",
        help=(
            "Express the target map and GT pose in a vertically reflected map "
            "frame before inference. The observation image is unchanged."
        ),
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(xy_errors, theta_errors):
    xy_errors = torch.cat(xy_errors)
    theta_errors = torch.cat(theta_errors)
    return {
        "samples": int(xy_errors.numel()),
        "0.1m_recall": float((xy_errors <= 0.1).float().mean()),
        "0.5m_recall": float((xy_errors <= 0.5).float().mean()),
        "1m_recall": float((xy_errors <= 1.0).float().mean()),
        "1m_30deg_recall": float(
            (
                (xy_errors <= 1.0)
                & (theta_errors <= math.radians(30.0))
            )
            .float()
            .mean()
        ),
        "mean_xy_err_m": float(xy_errors.mean()),
        "median_xy_err_m": float(xy_errors.median()),
        "mean_theta_err_deg": float(torch.rad2deg(theta_errors).mean()),
    }


def summarize_angle_offset_sweep(xy_errors, theta_deltas):
    xy_errors = torch.cat(xy_errors)
    theta_deltas = torch.cat(theta_deltas)
    recalls = {}
    for offset_deg in range(-180, 180, 5):
        corrected_error = torch.abs(
            torch.remainder(
                theta_deltas - math.radians(offset_deg) + math.pi,
                2.0 * math.pi,
            )
            - math.pi
        )
        recall = float(
            (
                (xy_errors <= 1.0)
                & (corrected_error <= math.radians(30.0))
            )
            .float()
            .mean()
        )
        recalls[offset_deg] = recall
    best_offset = max(recalls, key=recalls.get)
    return {
        "best_global_angle_offset_deg": int(best_offset),
        "best_offset_1m_30deg_recall": recalls[best_offset],
        "quarter_turn_1m_30deg_recall": {
            str(offset): recalls[offset] for offset in (-180, -90, 0, 90)
        },
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    seed_everything(args.seed)
    dataset_cfg = config["datasets"]
    split = args.split
    dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )
    total_samples = len(dataset)
    subset_indices = None
    if args.max_samples is not None and args.subset_fraction is not None:
        raise ValueError("Use either --max-samples or --subset-fraction, not both.")
    if args.subset_fraction is not None:
        if not 0.0 < args.subset_fraction <= 1.0:
            raise ValueError("--subset-fraction must be in (0, 1].")
        subset_size = max(1, int(round(total_samples * args.subset_fraction)))
        rng = np.random.default_rng(args.subset_seed)
        subset_indices = np.sort(
            rng.choice(total_samples, size=subset_size, replace=False)
        ).tolist()
        dataset = Subset(dataset, subset_indices)
    if args.max_samples is not None:
        subset_indices = list(range(min(args.max_samples, len(dataset))))
        dataset = Subset(dataset, subset_indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config.get("val_batch_size", 4)),
        shuffle=False,
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(config.get("num_workers", 4))
        ),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        args.ckpt,
        config=config,
        map_location="cpu",
    ).to(device)
    model.eval()

    xy_errors = []
    theta_errors = []
    theta_deltas = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"eval_{split}"):
            obs_img, gt_pose, _ray, floorplan_img, wh, *_ = batch
            obs_img = obs_img.to(device, non_blocking=True)
            gt_pose = gt_pose.to(device, non_blocking=True)
            floorplan_img = floorplan_img.to(device, non_blocking=True)
            wh = wh.to(device, non_blocking=True)
            if args.flip_target_map_vertical:
                floorplan_img = torch.flip(floorplan_img, dims=(-2,))
                gt_pose = gt_pose.clone()
                gt_pose[:, 1] = wh[:, 1] - 1.0 - gt_pose[:, 1]
                gt_pose[:, 2] = torch.remainder(
                    -gt_pose[:, 2],
                    2.0 * math.pi,
                )

            map_tokens, map_coordinates, image_global = model.condition_encoder(
                obs_img,
                floorplan_img,
            )
            pose_samples = model.sample_from_context(
                map_tokens,
                map_coordinates,
                image_global,
                wh,
                num_particles=int(config.get("diffusion_val_particles", 64)),
                num_steps=int(config.get("diffusion_sample_steps", 20)),
            )
            selected_pose, _density = model.select_pose_mode(pose_samples)
            xy_error = torch.linalg.norm(
                selected_pose[:, :2] - gt_pose[:, :2], dim=-1
            ) * model.map_res
            theta_error = model.angular_error(selected_pose[:, 2], gt_pose[:, 2])
            theta_delta = torch.remainder(
                selected_pose[:, 2] - gt_pose[:, 2] + math.pi,
                2.0 * math.pi,
            ) - math.pi
            xy_errors.append(xy_error.cpu())
            theta_errors.append(theta_error.cpu())
            theta_deltas.append(theta_delta.cpu())

    metrics = summarize(xy_errors, theta_errors)
    result = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.ckpt),
        "split": split,
        "seed": args.seed,
        "num_particles": int(config.get("diffusion_val_particles", 64)),
        "num_steps": int(config.get("diffusion_sample_steps", 20)),
        "target_total_samples": total_samples,
        "subset_fraction": args.subset_fraction,
        "subset_seed": args.subset_seed if args.subset_fraction is not None else None,
        "flip_target_map_vertical": args.flip_target_map_vertical,
        "subset_indices_sha256": (
            hashlib.sha256(
                np.asarray(subset_indices, dtype=np.int64).tobytes()
            ).hexdigest()
            if subset_indices is not None
            else None
        ),
        "angle_offset_diagnostic": summarize_angle_offset_sweep(
            xy_errors,
            theta_deltas,
        ),
        **metrics,
    }
    result_text = json.dumps(result, indent=2)
    print(result_text)
    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            output_file.write(result_text + "\n")


if __name__ == "__main__":
    main()

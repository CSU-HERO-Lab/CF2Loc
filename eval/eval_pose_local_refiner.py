import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_local_refiner import (
    PoseLocalRefinerLightning,
    apply_local_delta_to_pose,
    crop_to_refiner_tensor,
    load_refiner_map_np,
    wrap_to_pi,
)
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the complete diffusion and local-refiner pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/PoseLocalRefiner_S3D_Dense.yaml",
    )
    parser.add_argument(
        "--diffusion-ckpt",
        help="Override the Stage-1 checkpoint specified by the config.",
    )
    parser.add_argument("--refiner-ckpt", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def angular_error(pred_theta, target_theta):
    return torch.abs(wrap_to_pi(pred_theta - target_theta))


def update_metrics(store, prefix, pred_pose, target_pose, map_res):
    xy_error = torch.linalg.norm(pred_pose[:2] - target_pose[:2]) * map_res
    theta_error = angular_error(pred_pose[2], target_pose[2])
    store.setdefault(f"{prefix}_xy_errors", []).append(float(xy_error.item()))
    store.setdefault(f"{prefix}_theta_errors", []).append(float(theta_error.item()))
    for threshold in (0.1, 0.5, 1.0):
        key = f"{prefix}_{threshold}m_hits"
        store[key] = store.get(key, 0) + int((xy_error <= threshold).item())
    key = f"{prefix}_1m_30deg_hits"
    store[key] = store.get(key, 0) + int(
        ((xy_error <= 1.0) & (theta_error <= math.radians(30.0))).item()
    )


def summarize_metrics(store, prefix, count):
    xy_errors = torch.tensor(store[f"{prefix}_xy_errors"])
    theta_errors = torch.tensor(store[f"{prefix}_theta_errors"])
    return {
        "0.1m_recall": store[f"{prefix}_0.1m_hits"] / count,
        "0.5m_recall": store[f"{prefix}_0.5m_hits"] / count,
        "1m_recall": store[f"{prefix}_1.0m_hits"] / count,
        "1m_30deg_recall": store[f"{prefix}_1m_30deg_hits"] / count,
        "mean_xy_err_m": float(xy_errors.mean()),
        "median_xy_err_m": float(xy_errors.median()),
        "mean_theta_err_deg": float(torch.rad2deg(theta_errors).mean()),
    }


def build_local_maps(
    raw_map,
    candidate_poses,
    crop_size_meters,
    map_res,
    output_size,
    representation,
    oriented,
):
    local_maps = []
    for candidate_pose in candidate_poses.detach().cpu().numpy():
        local_maps.append(
            crop_to_refiner_tensor(
                raw_map,
                candidate_pose,
                crop_size_meters=crop_size_meters,
                map_res=map_res,
                output_size=output_size,
                representation=representation,
                oriented=oriented,
            ).float()
        )
    return torch.stack(local_maps)


def refine_candidates(
    refiner,
    image_tokens,
    raw_map,
    candidate_poses,
    wh,
    config,
    device,
):
    dataset_cfg = config["datasets"]
    candidate_count = candidate_poses.shape[0]
    local_maps = build_local_maps(
        raw_map,
        candidate_poses,
        crop_size_meters=float(config.get("refiner_crop_size_meters", 5.0)),
        map_res=float(dataset_cfg.get("map_res", 0.02)),
        output_size=int(config.get("refiner_crop_output_size", 256)),
        representation=dataset_cfg.get(
            "refiner_floorplan_representation",
            dataset_cfg.get("floorplan_representation", "rgb"),
        ),
        oriented=bool(dataset_cfg.get("refiner_oriented_crop", True)),
    ).to(device)
    wh_batch = wh.expand(candidate_count, -1)
    outputs = refiner(
        obs_img=None,
        local_map=local_maps,
        candidate_pose=candidate_poses,
        wh=wh_batch,
        image_tokens=image_tokens.expand(candidate_count, -1, -1),
    )
    refined_poses = apply_local_delta_to_pose(
        candidate_poses,
        outputs["delta_xy_m"],
        outputs["delta_theta"],
        map_res=float(dataset_cfg.get("map_res", 0.02)),
        wh=wh_batch,
    )
    return refined_poses, outputs["score_logit"]


def build_results(args, diffusion_ckpt, count, metrics):
    return {
        "diffusion_checkpoint": os.path.abspath(diffusion_ckpt),
        "refiner_checkpoint": os.path.abspath(args.refiner_ckpt),
        "split": args.split,
        "samples": count,
        "seed": args.seed,
        "top_k": args.top_k,
        "stage1": summarize_metrics(metrics, "stage1", count),
        "refined": summarize_metrics(metrics, "refined", count),
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    diffusion_ckpt = args.diffusion_ckpt or config["baseline_checkpoint_path"]
    config["baseline_checkpoint_path"] = diffusion_ckpt
    dataset_cfg = config["datasets"]
    map_res = float(dataset_cfg.get("map_res", 0.02))
    num_particles = int(config.get("diffusion_val_particles", 64))
    num_steps = int(config.get("diffusion_sample_steps", 20))
    if args.top_k > num_particles:
        raise ValueError(
            f"--top-k ({args.top_k}) exceeds the configured particle count "
            f"({num_particles})."
        )
    seed_everything(args.seed)

    dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=args.split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )
    sample_count = (
        min(args.max_samples, len(dataset))
        if args.max_samples is not None
        else len(dataset)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        diffusion_ckpt,
        map_location="cpu",
    ).to(device)
    diffusion.eval()
    refiner = PoseLocalRefinerLightning.load_from_checkpoint(
        args.refiner_ckpt,
        config=config,
        map_location="cpu",
    ).to(device)
    refiner.eval()

    metrics = {}
    with torch.inference_mode():
        for index in tqdm(range(sample_count), desc=f"eval_{args.split}"):
            obs_img, target_pose, _ray, floorplan_img, wh, *_ = dataset[index]
            obs_img = obs_img.unsqueeze(0).to(device)
            target_pose = target_pose.to(device)
            floorplan_img = floorplan_img.unsqueeze(0).to(device)
            wh = wh.unsqueeze(0).to(device)

            map_tokens, map_coordinates, image_global = diffusion.condition_encoder(
                obs_img, floorplan_img
            )
            pose_samples = diffusion.sample_from_context(
                map_tokens,
                map_coordinates,
                image_global,
                wh,
                num_particles=num_particles,
                num_steps=num_steps,
            )
            stage1_pose, density = diffusion.select_pose_mode(pose_samples)
            stage1_pose = stage1_pose[0]
            density = density[0]
            update_metrics(metrics, "stage1", stage1_pose, target_pose, map_res)

            candidate_indices = torch.topk(density, k=args.top_k).indices
            candidate_poses = pose_samples[0, candidate_indices]
            image_tokens = refiner.refiner.encode_image(obs_img)
            raw_map = load_refiner_map_np(
                dataset,
                dataset.data[index]["floorplan_image"],
            )
            refined_poses, scores = refine_candidates(
                refiner,
                image_tokens,
                raw_map,
                candidate_poses,
                wh,
                config,
                device,
            )
            final_pose = refined_poses[torch.argmax(scores)]
            update_metrics(metrics, "refined", final_pose, target_pose, map_res)

    results = build_results(args, diffusion_ckpt, sample_count, metrics)
    result_text = json.dumps(results, indent=2)
    print(result_text)
    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(result_text + "\n")


if __name__ == "__main__":
    main()

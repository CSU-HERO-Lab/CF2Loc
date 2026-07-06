import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_local_refiner import (
    PoseLocalRefinerLightning,
    apply_local_delta_to_pose,
    crop_local_map_np,
    wrap_to_pi,
)
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def angular_error(pred_theta: torch.Tensor, gt_theta: torch.Tensor) -> torch.Tensor:
    return torch.abs(wrap_to_pi(pred_theta - gt_theta))


def update_metrics(store, prefix, pred_pose, gt_pose, map_res):
    xy_error = torch.linalg.norm(pred_pose[:2] - gt_pose[:2]) * map_res
    theta_error = angular_error(pred_pose[2], gt_pose[2])
    store.setdefault(f"{prefix}_xy_errors", []).append(float(xy_error.item()))
    store.setdefault(f"{prefix}_theta_errors", []).append(float(theta_error.item()))
    for threshold in (0.1, 0.5, 1.0):
        key = f"{prefix}_{threshold}m_hits"
        store[key] = store.get(key, 0) + int(xy_error <= threshold)
    key = f"{prefix}_1m_30deg_hits"
    store[key] = store.get(key, 0) + int(
        xy_error <= 1.0 and theta_error <= math.radians(30.0)
    )


def summarize_metrics(store, prefix, count):
    xy_errors = torch.tensor(store[f"{prefix}_xy_errors"], dtype=torch.float32)
    theta_errors = torch.tensor(store[f"{prefix}_theta_errors"], dtype=torch.float32)
    return {
        "0.1m_recall": store[f"{prefix}_0.1m_hits"] / count,
        "0.5m_recall": store[f"{prefix}_0.5m_hits"] / count,
        "1m_recall": store[f"{prefix}_1.0m_hits"] / count,
        "1m_30deg_recall": store[f"{prefix}_1m_30deg_hits"] / count,
        "mean_xy_err_m": float(xy_errors.mean().item()),
        "median_xy_err_m": float(xy_errors.median().item()),
        "mean_theta_err_deg": float(torch.rad2deg(theta_errors).mean().item()),
    }


def build_results(diffusion_ckpt, refiner_ckpt, split, count, top_k, metrics):
    return {
        "diffusion_ckpt": diffusion_ckpt,
        "refiner_ckpt": refiner_ckpt,
        "split": split,
        "samples": count,
        "top_k": int(top_k),
        "baseline": summarize_metrics(metrics, "baseline", count),
        "selected_refined": summarize_metrics(metrics, "selected_refined", count),
        "topk_refined": summarize_metrics(metrics, "topk_refined", count),
    }


def build_local_maps(raw_map, candidate_poses, crop_size_meters, map_res, output_size):
    local_maps = []
    for candidate_pose in candidate_poses.detach().cpu().numpy():
        local_map = crop_local_map_np(
            raw_map,
            candidate_pose,
            crop_size_meters=crop_size_meters,
            map_res=map_res,
            output_size=output_size,
        )
        local_maps.append(torch.from_numpy(local_map).float().unsqueeze(0) / 255.0)
    return torch.stack(local_maps, dim=0)


@torch.no_grad()
def refine_candidates(
    refiner,
    obs_img,
    raw_map,
    candidate_poses,
    wh,
    crop_size_meters,
    crop_output_size,
    map_res,
    device,
):
    local_maps = build_local_maps(
        raw_map,
        candidate_poses,
        crop_size_meters=crop_size_meters,
        map_res=map_res,
        output_size=crop_output_size,
    ).to(device)
    obs_batch = obs_img.expand(candidate_poses.shape[0], -1, -1, -1)
    wh_batch = wh.expand(candidate_poses.shape[0], -1)
    outputs = refiner(
        obs_batch,
        local_maps,
        candidate_poses,
        wh_batch,
    )
    refined = apply_local_delta_to_pose(
        candidate_poses,
        outputs["delta_xy_m"],
        outputs["delta_theta"],
        map_res=map_res,
        wh=wh_batch,
    )
    return refined, outputs["score_logit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PoseLocalRefiner_S3D_Baseline.yaml")
    parser.add_argument("--diffusion_ckpt")
    parser.add_argument("--refiner_ckpt", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--output_json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    diffusion_ckpt = args.diffusion_ckpt or config["baseline_checkpoint_path"]
    dataset_cfg = config["datasets"]
    split = args.split or dataset_cfg.get("val_split", "val")
    map_res = float(dataset_cfg.get("map_res", 0.02))
    crop_size_meters = float(config.get("refiner_crop_size_meters", 5.0))
    crop_output_size = int(config.get("refiner_crop_output_size", 256))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )
    if args.subset_fraction < 1.0:
        stride = max(1, int(round(1.0 / args.subset_fraction)))
        indices = list(range(0, len(dataset), stride))
    else:
        indices = list(range(len(dataset)))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    diffusion = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        diffusion_ckpt,
        config=config,
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
    count = 0
    for index in tqdm(indices, desc="eval_refiner"):
        obs_img, gt_pose, _ray, floorplan_img, wh, *_ = dataset[index]
        obs_img = obs_img.unsqueeze(0).to(device)
        gt_pose = gt_pose.to(device)
        floorplan_img = floorplan_img.unsqueeze(0).to(device)
        wh = wh.unsqueeze(0).to(device)

        item = dataset.data[index]
        raw_map = cv2.imread(item["floorplan_image"], cv2.IMREAD_GRAYSCALE)
        if raw_map is None:
            raise FileNotFoundError(f"Failed to load floorplan {item['floorplan_image']}")

        map_tokens, map_coordinates, image_global = diffusion.condition_encoder(
            obs_img,
            floorplan_img,
        )
        pose_samples = diffusion.sample_from_context(
            map_tokens,
            map_coordinates,
            image_global,
            wh,
            num_particles=int(config.get("diffusion_val_particles", 64)),
            num_steps=int(config.get("diffusion_sample_steps", 20)),
        )
        selected_pose, density = diffusion.select_pose_mode(pose_samples)
        selected_pose = selected_pose[0]
        density = density[0]
        update_metrics(metrics, "baseline", selected_pose, gt_pose, map_res)

        selected_refined, selected_score = refine_candidates(
            refiner,
            obs_img,
            raw_map,
            selected_pose.view(1, 3),
            wh,
            crop_size_meters,
            crop_output_size,
            map_res,
            device,
        )
        update_metrics(
            metrics,
            "selected_refined",
            selected_refined[0],
            gt_pose,
            map_res,
        )

        top_k = min(int(args.top_k), pose_samples.shape[1])
        top_indices = torch.topk(density, k=top_k).indices
        top_candidates = pose_samples[0, top_indices]
        top_refined, top_scores = refine_candidates(
            refiner,
            obs_img,
            raw_map,
            top_candidates,
            wh,
            crop_size_meters,
            crop_output_size,
            map_res,
            device,
        )
        selected_idx = torch.argmax(top_scores)
        update_metrics(
            metrics,
            "topk_refined",
            top_refined[selected_idx],
            gt_pose,
            map_res,
        )
        count += 1
        if args.log_every > 0 and count % args.log_every == 0:
            partial_results = build_results(
                diffusion_ckpt,
                args.refiner_ckpt,
                split,
                count,
                args.top_k,
                metrics,
            )
            print(
                "[partial_metrics] "
                + json.dumps(
                    {
                        "samples": count,
                        "baseline": partial_results["baseline"],
                        "selected_refined": partial_results["selected_refined"],
                        "topk_refined": partial_results["topk_refined"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.output_json:
                partial_path = args.output_json + ".partial"
                with open(partial_path, "w", encoding="utf-8") as output_file:
                    json.dump(partial_results, output_file, indent=2)

    results = build_results(
        diffusion_ckpt,
        args.refiner_ckpt,
        split,
        count,
        args.top_k,
        metrics,
    )
    print(json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)


if __name__ == "__main__":
    main()

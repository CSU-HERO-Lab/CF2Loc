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
    crop_to_refiner_tensor,
    load_refiner_map_np,
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
    result = {
        "diffusion_ckpt": diffusion_ckpt,
        "refiner_ckpt": refiner_ckpt,
        "split": split,
        "samples": count,
        "top_k": int(top_k),
        "baseline": summarize_metrics(metrics, "baseline", count),
        "selected_refined": summarize_metrics(metrics, "selected_refined", count),
        "topk_refined": summarize_metrics(metrics, "topk_refined", count),
    }
    if "effective_top_k_sum" in metrics:
        result["mean_effective_top_k"] = metrics["effective_top_k_sum"] / count
    return result


def choose_adaptive_top_k(
    density: torch.Tensor,
    max_top_k: int,
    min_top_k: int,
    mass: float,
) -> int:
    max_top_k = min(int(max_top_k), density.numel())
    min_top_k = min(max(1, int(min_top_k)), max_top_k)
    if max_top_k <= min_top_k:
        return max_top_k
    sorted_density = torch.sort(density, descending=True).values[:max_top_k]
    weights = sorted_density.clamp_min(0.0)
    if float(weights.sum().item()) <= 1e-8:
        return max_top_k
    cumulative = torch.cumsum(weights / weights.sum().clamp_min(1e-8), dim=0)
    reached = torch.nonzero(cumulative >= float(mass), as_tuple=False)
    if reached.numel() == 0:
        return max_top_k
    return max(min_top_k, int(reached[0, 0].item()) + 1)


def expand_cached_image_tokens(
    image_tokens: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if image_tokens.shape[0] == count:
        return image_tokens
    if image_tokens.shape[0] != 1:
        raise ValueError(
            f"Cannot expand cached image tokens with batch {image_tokens.shape[0]} "
            f"to candidate count {count}"
        )
    return image_tokens.expand(count, -1, -1)


def standardize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values * 0.0
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def weighted_pose_mean(poses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1e-6)
    xy = (poses[:, :2] * weights[:, None]).sum(dim=0)
    sin_theta = (torch.sin(poses[:, 2]) * weights).sum()
    cos_theta = (torch.cos(poses[:, 2]) * weights).sum()
    theta = torch.atan2(sin_theta, cos_theta)
    return torch.cat([xy, theta.view(1)], dim=0)


def build_local_maps(
    raw_map,
    candidate_poses,
    crop_size_meters,
    map_res,
    output_size,
    representation,
):
    local_maps = []
    for candidate_pose in candidate_poses.detach().cpu().numpy():
        local_map = crop_to_refiner_tensor(
            raw_map,
            candidate_pose,
            crop_size_meters=crop_size_meters,
            map_res=map_res,
            output_size=output_size,
            representation=representation,
        )
        local_maps.append(local_map.float())
    return torch.stack(local_maps, dim=0)


@torch.no_grad()
def refine_candidates(
    refiner,
    obs_img,
    cached_image_tokens,
    raw_map,
    candidate_poses,
    wh,
    crop_size_meters,
    crop_output_size,
    map_res,
    representation,
    device,
    refine_iters,
    delta_scale,
):
    refined = candidate_poses
    score_logit = None
    for _ in range(max(1, int(refine_iters))):
        local_maps = build_local_maps(
            raw_map,
            refined,
            crop_size_meters=crop_size_meters,
            map_res=map_res,
            output_size=crop_output_size,
            representation=representation,
        ).to(device)
        obs_batch = obs_img.expand(refined.shape[0], -1, -1, -1)
        wh_batch = wh.expand(refined.shape[0], -1)
        image_tokens_batch = None
        if cached_image_tokens is not None:
            image_tokens_batch = expand_cached_image_tokens(
                cached_image_tokens,
                refined.shape[0],
            )
        outputs = refiner(
            obs_batch,
            local_maps,
            refined,
            wh_batch,
            image_tokens_batch,
        )
        refined = apply_local_delta_to_pose(
            refined,
            outputs["delta_xy_m"] * float(delta_scale),
            outputs["delta_theta"] * float(delta_scale),
            map_res=map_res,
            wh=wh_batch,
        )
        score_logit = outputs["score_logit"]
    return refined, score_logit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PoseLocalRefiner_S3D_Baseline.yaml")
    parser.add_argument("--diffusion_ckpt")
    parser.add_argument("--refiner_ckpt", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help=(
            "Number of KDE-ranked candidates to refine. The default refines only "
            "the KDE mode; values greater than one enable the slower candidate "
            "quality reranking ablation."
        ),
    )
    parser.add_argument("--val_particles", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--mode_sigma_m", type=float, default=None)
    parser.add_argument("--mode_sigma_deg", type=float, default=None)
    parser.add_argument("--crop_size_meters", type=float, default=None)
    parser.add_argument("--refine_iters", type=int, default=1)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument(
        "--cache_refiner_image",
        action="store_true",
        help="Encode the observation image once per sample and reuse it for all refiner candidates.",
    )
    parser.add_argument(
        "--adaptive_top_k",
        action="store_true",
        help="Use density mass to reduce the number of top-k candidates refined per sample.",
    )
    parser.add_argument("--adaptive_min_top_k", type=int, default=4)
    parser.add_argument("--adaptive_top_k_mass", type=float, default=0.9)
    parser.add_argument(
        "--topk_aggregate",
        choices=("none", "mean", "score_mean", "density_mean", "mixed_mean"),
        default="none",
    )
    parser.add_argument(
        "--mixed_score_alpha",
        type=float,
        default=None,
        help=(
            "If set, choose the refined top-k pose with "
            "zscore(diffusion_density)+alpha*zscore(refiner_score). "
            "By default, keep the original refiner-score-only selection."
        ),
    )
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--output_json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top_k must be at least 1")

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if args.val_particles is not None:
        config["diffusion_val_particles"] = int(args.val_particles)
    if args.sample_steps is not None:
        config["diffusion_sample_steps"] = int(args.sample_steps)
    if args.mode_sigma_m is not None:
        config["diffusion_mode_sigma_m"] = float(args.mode_sigma_m)
    if args.mode_sigma_deg is not None:
        config["diffusion_mode_sigma_deg"] = float(args.mode_sigma_deg)
    diffusion_ckpt = args.diffusion_ckpt or config["baseline_checkpoint_path"]
    dataset_cfg = config["datasets"]
    split = args.split or dataset_cfg.get("val_split", "val")
    map_res = float(dataset_cfg.get("map_res", 0.02))
    refiner_floorplan_representation = dataset_cfg.get(
        "refiner_floorplan_representation",
        dataset_cfg.get("floorplan_representation", "rgb"),
    )
    crop_size_meters = float(
        args.crop_size_meters
        if args.crop_size_meters is not None
        else config.get("refiner_crop_size_meters", 5.0)
    )
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
        raw_map = load_refiner_map_np(dataset, item["floorplan_image"])
        cached_image_tokens = None
        if args.cache_refiner_image:
            cached_image_tokens = refiner.refiner.encode_image(obs_img)

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
            cached_image_tokens,
            raw_map,
            selected_pose.view(1, 3),
            wh,
            crop_size_meters,
            crop_output_size,
            map_res,
            refiner_floorplan_representation,
            device,
            args.refine_iters,
            args.delta_scale,
        )
        update_metrics(
            metrics,
            "selected_refined",
            selected_refined[0],
            gt_pose,
            map_res,
        )

        max_top_k = min(int(args.top_k), pose_samples.shape[1])
        if max_top_k == 1:
            # The KDE-selected pose is already the highest-density particle.
            # Reuse its refinement instead of cropping and encoding it twice.
            top_k = 1
            topk_pose = selected_refined[0]
        else:
            if args.adaptive_top_k:
                top_k = choose_adaptive_top_k(
                    density,
                    max_top_k=max_top_k,
                    min_top_k=args.adaptive_min_top_k,
                    mass=args.adaptive_top_k_mass,
                )
            else:
                top_k = max_top_k
            top_indices = torch.topk(density, k=top_k).indices
            top_candidates = pose_samples[0, top_indices]
            top_refined, top_scores = refine_candidates(
                refiner,
                obs_img,
                cached_image_tokens,
                raw_map,
                top_candidates,
                wh,
                crop_size_meters,
                crop_output_size,
                map_res,
                refiner_floorplan_representation,
                device,
                args.refine_iters,
                args.delta_scale,
            )
            if args.mixed_score_alpha is None:
                selection_scores = top_scores
            else:
                top_density = density[top_indices]
                selection_scores = standardize(top_density) + float(
                    args.mixed_score_alpha
                ) * standardize(top_scores)
            if args.topk_aggregate == "none":
                topk_pose = top_refined[torch.argmax(selection_scores)]
            elif args.topk_aggregate == "mean":
                topk_pose = weighted_pose_mean(
                    top_refined,
                    torch.ones_like(top_scores),
                )
            elif args.topk_aggregate == "score_mean":
                topk_pose = weighted_pose_mean(
                    top_refined,
                    torch.softmax(top_scores, dim=0),
                )
            elif args.topk_aggregate == "density_mean":
                topk_pose = weighted_pose_mean(
                    top_refined,
                    torch.softmax(density[top_indices], dim=0),
                )
            else:
                topk_pose = weighted_pose_mean(
                    top_refined,
                    torch.softmax(selection_scores, dim=0),
                )
        metrics["effective_top_k_sum"] = metrics.get("effective_top_k_sum", 0.0) + top_k
        update_metrics(
            metrics,
            "topk_refined",
            topk_pose,
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

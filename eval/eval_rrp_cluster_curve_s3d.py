import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from attrdict import AttrDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.RRP_lightning_module import RRPLightningModule
from utils.data_utils import GridSeqDataset
from utils.localization_utils import get_ray_from_depth, localize


def parse_int_values(raw_values):
    values = []
    for chunk in raw_values.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    if not values:
        raise ValueError("No valid integer values were provided.")
    return values


def load_eval_dataset(dataset_dir, all_imgs):
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r", encoding="utf-8") as f:
        split = AttrDict(yaml.safe_load(f))

    return GridSeqDataset(
        dataset_dir,
        split.test,
        L=3,
        depth_dir=dataset_dir,
        depth_suffix="depth40",
        add_rp=False,
        net_type="rrp",
        all_imgs=all_imgs,
    )


def load_scene_metadata(test_set, desdf_path, dataset_dir):
    desdfs = {}
    gt_poses = {}

    for scene in tqdm.tqdm(test_set.scene_names, desc="Loading scenes"):
        desdf = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"),
            allow_pickle=True,
        ).item()
        desdf["desdf"][desdf["desdf"] > 20] = 20
        desdfs[scene] = desdf

        with open(os.path.join(dataset_dir, scene, "poses_map.txt"), "r", encoding="utf-8") as f:
            poses = []
            for line in f:
                parts = list(map(float, line.strip().split()))
                poses.append(parts[:3])
        gt_poses[scene] = np.array(poses, dtype=np.float32)

    return desdfs, gt_poses


def cluster_topk_candidates(topk_vals, topk_indices, width, radius_m, meters_per_cell):
    if topk_indices.numel() == 0 or radius_m <= 0:
        keep_positions = torch.arange(topk_indices.shape[0], dtype=torch.long)
        cluster_sizes = torch.ones(topk_indices.shape[0], dtype=torch.long)
        cluster_scores_sum = topk_vals.clone()
        return topk_vals, topk_indices, keep_positions, cluster_sizes, cluster_scores_sum

    radius_cells = radius_m / meters_per_cell
    radius_sq = radius_cells * radius_cells

    topk_y = (topk_indices // width).to(torch.float32)
    topk_x = (topk_indices % width).to(torch.float32)

    suppressed = torch.zeros(topk_indices.shape[0], dtype=torch.bool)
    keep_positions = []
    cluster_sizes = []
    cluster_scores_sum = []

    for idx in range(topk_indices.shape[0]):
        if suppressed[idx]:
            continue

        dx = topk_x - topk_x[idx]
        dy = topk_y - topk_y[idx]
        cluster_mask = (~suppressed) & ((dx * dx + dy * dy) <= radius_sq)

        keep_positions.append(idx)
        cluster_sizes.append(int(cluster_mask.sum().item()))
        cluster_scores_sum.append(topk_vals[cluster_mask].sum())
        suppressed |= cluster_mask

    keep_positions = torch.tensor(keep_positions, dtype=torch.long)
    cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.long)
    cluster_scores_sum = torch.stack(cluster_scores_sum)
    return (
        topk_vals[keep_positions],
        topk_indices[keep_positions],
        keep_positions,
        cluster_sizes,
        cluster_scores_sum,
    )


def sorted_cluster_representatives(rep_vals, rep_indices, cluster_sizes, cluster_scores_sum, rank_by):
    if rank_by == "peak":
        order = torch.arange(rep_indices.shape[0], dtype=torch.long)
    elif rank_by == "sum":
        order = torch.argsort(cluster_scores_sum, descending=True)
    elif rank_by == "size":
        order = torch.argsort(cluster_sizes, descending=True)
    else:
        raise ValueError(f"Unsupported rank_by '{rank_by}'. Expected peak, sum, or size.")

    return (
        rep_vals[order],
        rep_indices[order],
        cluster_sizes[order],
        cluster_scores_sum[order],
    )


def evaluate_rrp_cluster_curve(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    cluster_n_values = parse_int_values(args.cluster_n_values)
    hit_radii = [float(x) for x in args.hit_radii_m.split(",") if x.strip()]
    if not hit_radii:
        raise ValueError("No valid hit radii were provided.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Cluster N values: {cluster_n_values}")
    print(
        "Clustering top "
        f"{args.cluster_source_top_k} RRP cells with radius {args.cluster_radius_m:.2f}m "
        f"and rank_by={args.rank_by}"
    )

    rrp_plt = RRPLightningModule.load_from_checkpoint(args.rrp_model_ckpt, map_location=device)
    rrp_model = rrp_plt.model.to(device)
    rrp_model.eval()

    test_set = load_eval_dataset(args.dataset_path, args.all_imgs)
    desdfs, gt_poses = load_scene_metadata(test_set, args.desdf_path, args.dataset_path)

    fov_factor = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    desdf_stride = 5
    meters_per_desdf_cell = desdf_stride * 0.02
    max_cluster_n = max(cluster_n_values)

    hit_counts = {
        radius: np.zeros(len(cluster_n_values), dtype=np.int64)
        for radius in hit_radii
    }
    total_samples = 0
    rep_counts = []
    pool_counts = []
    cluster_size_means = []

    if args.scene_name is not None and args.img_id is not None:
        scene_idx = test_set.scene_names.index(args.scene_name)
        loop_range = [test_set.scene_start_idx[scene_idx] + args.img_id]
        print(f"Debug mode: evaluating {args.scene_name} image {args.img_id}")
    else:
        loop_range = range(len(test_set))

    progress = tqdm.tqdm(loop_range, desc="RRP Cluster Top-N")
    for data_idx in progress:
        data = test_set[data_idx]

        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
        ref_idx = idx_within_scene if args.all_imgs else idx_within_scene * 4 + 3

        desdf_data = desdfs[scene]
        gt_pose_map = gt_poses[scene][ref_idx, :].copy()
        gt_pose_desdf = gt_pose_map.copy()
        gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf_data["l"]) / desdf_stride
        gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf_data["t"]) / desdf_stride

        obs_img_tensor = data["obs_tensor"].unsqueeze(0).to(device)

        with torch.no_grad():
            features = rrp_model("encode", obs_img=obs_img_tensor)
            pred_depths = rrp_model("decoder_inference", depth_cond=features)
            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()

            pred_rays = get_ray_from_depth(pred_depths, V=9, F_W=fov_factor)
            pred_rays = torch.tensor(pred_rays, device="cpu")

            _, prob_dist, _, _ = localize(
                torch.tensor(desdf_data["desdf"]),
                pred_rays,
                return_np=False,
            )

        flat_probs = prob_dist.flatten()
        source_k = min(args.cluster_source_top_k, flat_probs.numel())
        topk_vals, topk_indices = torch.topk(flat_probs, k=source_k)
        height, width = prob_dist.shape

        rep_vals, rep_indices, _, cluster_sizes, cluster_scores_sum = cluster_topk_candidates(
            topk_vals,
            topk_indices,
            width=width,
            radius_m=args.cluster_radius_m,
            meters_per_cell=meters_per_desdf_cell,
        )
        rep_vals, rep_indices, cluster_sizes, cluster_scores_sum = sorted_cluster_representatives(
            rep_vals,
            rep_indices,
            cluster_sizes,
            cluster_scores_sum,
            args.rank_by,
        )

        rep_counts.append(int(rep_indices.numel()))
        pool_counts.append(int(source_k))
        cluster_size_means.append(float(cluster_sizes.float().mean().item()) if cluster_sizes.numel() else 0.0)

        take_n = min(max_cluster_n, rep_indices.numel())
        rep_indices_eval = rep_indices[:take_n]
        rep_y = (rep_indices_eval // width).to(torch.float32)
        rep_x = (rep_indices_eval % width).to(torch.float32)
        gt_x = float(gt_pose_desdf[0])
        gt_y = float(gt_pose_desdf[1])
        dists_m = torch.sqrt((rep_x - gt_x) ** 2 + (rep_y - gt_y) ** 2) * meters_per_desdf_cell

        for radius in hit_radii:
            for idx, cluster_n in enumerate(cluster_n_values):
                current_n = min(cluster_n, rep_indices_eval.numel())
                if current_n > 0 and torch.any(dists_m[:current_n] <= radius):
                    hit_counts[radius][idx] += 1

        total_samples += 1
        last_radius = hit_radii[-1]
        progress.set_postfix(
            {
                f"top{cluster_n_values[-1]}@{last_radius:g}m": (
                    f"{hit_counts[last_radius][-1] / total_samples:.4f}"
                ),
                "avg_rep": f"{np.mean(rep_counts):.1f}",
            }
        )

    recalls = {
        f"{radius:g}m": (counts / total_samples).tolist()
        for radius, counts in hit_counts.items()
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "rrp_model_ckpt": args.rrp_model_ckpt,
        "dataset_path": args.dataset_path,
        "desdf_path": args.desdf_path,
        "all_imgs": args.all_imgs,
        "fov": args.fov,
        "cluster_source_top_k": args.cluster_source_top_k,
        "cluster_radius_m": args.cluster_radius_m,
        "rank_by": args.rank_by,
        "cluster_n_values": cluster_n_values,
        "hit_radii_m": hit_radii,
        "total_samples": total_samples,
        "recalls": recalls,
        "avg_source_k": float(np.mean(pool_counts)),
        "avg_rep_count": float(np.mean(rep_counts)),
        "avg_cluster_size": float(np.mean(cluster_size_means)),
        "note": (
            "Clusters are formed by radius NMS over RRP top cells. A hit means any "
            "of the first N cluster representatives is within the requested radius "
            "of the ground-truth DESDF position."
        ),
    }

    json_path = os.path.join(args.output_dir, f"rrp_cluster_curve_s3d_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plt.figure(figsize=(10, 5))
    x_positions = np.arange(len(cluster_n_values))
    for radius in hit_radii:
        plt.plot(
            x_positions,
            hit_counts[radius] / total_samples,
            marker="o",
            linewidth=2,
            label=f"{radius:g}m",
        )
    plt.xticks(x_positions, [str(x) for x in cluster_n_values], rotation=30)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Top-N clusters")
    plt.ylabel("Hit Rate")
    plt.title(
        "S3D RRP Cluster Top-N Hit Rate "
        f"(source_top_k={args.cluster_source_top_k}, radius={args.cluster_radius_m:g}m)"
    )
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, f"rrp_cluster_curve_s3d_{timestamp}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\nS3D RRP cluster top-N hit rates")
    print("-" * 60)
    print(f"avg source k: {result['avg_source_k']:.1f}")
    print(f"avg representative clusters: {result['avg_rep_count']:.1f}")
    print(f"avg cluster size: {result['avg_cluster_size']:.2f}")
    for radius in hit_radii:
        print(f"\nwithin {radius:g}m")
        for cluster_n, recall in zip(cluster_n_values, hit_counts[radius] / total_samples):
            print(f"top-{cluster_n} clusters: {recall:.4f}")
    print("-" * 60)
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {plot_path}")

    return json_path, plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate S3D RRP top-N cluster hit rate curve")
    parser.add_argument("--rrp_model_ckpt", type=str, default="checkpoints/RRP_s3d_best.ckpt")
    parser.add_argument("--dataset_path", type=str, default="./datasets_s3d/Structured3D/")
    parser.add_argument("--desdf_path", type=str, default="./datasets_s3d/desdf/")
    parser.add_argument("--cluster_source_top_k", type=int, default=1000)
    parser.add_argument("--cluster_radius_m", type=float, default=0.6)
    parser.add_argument("--cluster_n_values", type=str, default="1,2,5,10,20,50,100,200")
    parser.add_argument("--hit_radii_m", type=str, default="0.5,1.0")
    parser.add_argument("--rank_by", type=str, default="peak", choices=("peak", "sum", "size"))
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--output_dir", type=str, default="./eval/logs/rrp_cluster")
    parser.add_argument("--all_imgs", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--img_id", type=int, default=None)
    evaluate_rrp_cluster_curve(parser.parse_args())

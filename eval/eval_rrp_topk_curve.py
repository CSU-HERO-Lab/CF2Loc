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


def parse_k_values(raw_k_values):
    values = []
    for chunk in raw_k_values.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("No valid k values were provided.")
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


def evaluate_rrp_topk(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_k_values = parse_k_values(args.k_values)
    effective_k_values = [1 if k <= 0 else k for k in requested_k_values]
    hit_radius_m = float(args.hit_radius_m)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Requested k values: {requested_k_values}")

    rrp_plt = RRPLightningModule.load_from_checkpoint(args.rrp_model_ckpt, map_location=device)
    rrp_model = rrp_plt.model.to(device)
    rrp_model.eval()

    test_set = load_eval_dataset(args.dataset_path, args.all_imgs)
    desdfs, gt_poses = load_scene_metadata(test_set, args.desdf_path, args.dataset_path)

    fov_factor = 1 / np.tan(0.698132) / 2
    desdf_stride = 5
    max_effective_k = max(effective_k_values)

    hit_counts = np.zeros(len(requested_k_values), dtype=np.int64)
    total_samples = 0

    if args.scene_name is not None and args.img_id is not None:
        try:
            scene_idx = test_set.scene_names.index(args.scene_name)
            start_idx = test_set.scene_start_idx[scene_idx]
            target_idx = start_idx + args.img_id
            loop_range = [target_idx]
            print(
                f"Debug mode: evaluating only {args.scene_name}, image {args.img_id} "
                f"(global index {target_idx})"
            )
        except ValueError as exc:
            raise ValueError(f"Scene {args.scene_name} not found in test split.") from exc
    else:
        loop_range = range(len(test_set))

    progress = tqdm.tqdm(loop_range, desc="RRP Top-k")
    for data_idx in progress:
        data = test_set[data_idx]

        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        if args.all_imgs:
            ref_idx = idx_within_scene
        else:
            ref_idx = idx_within_scene * 4 + 3

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
        max_k = min(max_effective_k, flat_probs.numel())
        topk_indices = torch.topk(flat_probs, k=max_k).indices
        height, width = prob_dist.shape

        if hit_radius_m <= 0:
            gt_x = int(np.clip(np.round(gt_pose_desdf[0]), 0, width - 1))
            gt_y = int(np.clip(np.round(gt_pose_desdf[1]), 0, height - 1))
            gt_flat_idx = gt_y * width + gt_x
            matches = torch.nonzero(topk_indices == gt_flat_idx, as_tuple=False)
            gt_rank = int(matches[0].item()) + 1 if matches.numel() > 0 else flat_probs.numel() + 1

            for idx, effective_k in enumerate(effective_k_values):
                if gt_rank <= min(effective_k, flat_probs.numel()):
                    hit_counts[idx] += 1
        else:
            topk_y = (topk_indices // width).to(torch.float32)
            topk_x = (topk_indices % width).to(torch.float32)
            gt_x = float(gt_pose_desdf[0])
            gt_y = float(gt_pose_desdf[1])
            dists_m = torch.sqrt((topk_x - gt_x) ** 2 + (topk_y - gt_y) ** 2) * 0.1

            for idx, effective_k in enumerate(effective_k_values):
                current_k = min(effective_k, flat_probs.numel())
                if torch.any(dists_m[:current_k] <= hit_radius_m):
                    hit_counts[idx] += 1

        total_samples += 1
        current_top1 = hit_counts[0] / total_samples
        current_last = hit_counts[-1] / total_samples
        progress.set_postfix(
            {
                "top1_like": f"{current_top1:.4f}",
                f"top{requested_k_values[-1]}": f"{current_last:.4f}",
            }
        )

    recalls = hit_counts / total_samples

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "rrp_model_ckpt": args.rrp_model_ckpt,
        "dataset_path": args.dataset_path,
        "desdf_path": args.desdf_path,
        "all_imgs": args.all_imgs,
        "k_values_requested": requested_k_values,
        "k_values_effective": effective_k_values,
        "total_samples": total_samples,
        "topk_recall": recalls.tolist(),
        "hit_radius_m": hit_radius_m,
        "note": (
            "Requested k=0 is treated as top-1 / argmax hit. "
            "If hit_radius_m > 0, a hit means any top-k candidate falls within the radius."
        ),
    }

    json_path = os.path.join(args.output_dir, f"rrp_topk_curve_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    x_positions = np.arange(len(requested_k_values))
    x_labels = ["0(top1)" if k <= 0 else str(k) for k in requested_k_values]

    plt.figure(figsize=(10, 5))
    plt.plot(x_positions, recalls, marker="o", linewidth=2)
    plt.xticks(x_positions, x_labels, rotation=30)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Top-k")
    plt.ylabel("Hit Rate")
    if hit_radius_m > 0:
        plt.title(f"RRP Top-k Hit Rate Curve ({hit_radius_m:.1f}m radius)")
    else:
        plt.title("RRP Top-k Hit Rate Curve (exact cell)")
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, f"rrp_topk_curve_{timestamp}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    if hit_radius_m > 0:
        print(f"\nRRP top-k hit rates within {hit_radius_m:.1f}m")
    else:
        print("\nRRP top-k exact-cell hit rates")
    print("-" * 40)
    for requested_k, recall in zip(requested_k_values, recalls):
        label = "0(top1)" if requested_k <= 0 else str(requested_k)
        print(f"top-{label}: {recall:.4f}")
    print("-" * 40)
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {plot_path}")

    return json_path, plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pure RRP top-k hit rate curve")
    parser.add_argument("--rrp_model_ckpt", type=str, default="checkpoints/RRP_s3d_best.ckpt")
    parser.add_argument("--dataset_path", type=str, default="./datasets_s3d/Structured3D/")
    parser.add_argument("--desdf_path", type=str, default="./datasets_s3d/desdf/")
    parser.add_argument(
        "--k_values",
        type=str,
        default="0,10,100,200,400,800,1200,1600,2000,3000,4000,5000",
    )
    parser.add_argument("--hit_radius_m", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./eval/logs/rrp_topk")
    parser.add_argument("--all_imgs", action="store_true", default=True)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--img_id", type=int, default=None)
    evaluate_rrp_topk(parser.parse_args())

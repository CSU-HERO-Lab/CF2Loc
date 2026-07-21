#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import yaml
from attrdict import AttrDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from training.RRP_lightning_module import RRPLightningModule
from utils.data_utils import GridSeqDataset
from utils.localization_utils import get_ray_from_depth, localize_fast


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark batch-one RRP latency.")
    parser.add_argument(
        "--checkpoint",
        default="/home/ros/meng/DisCo-FLoc/checkpoints/RRP_s3d_best.ckpt",
    )
    parser.add_argument("--dataset", default="datasets_s3d/Structured3D")
    parser.add_argument("--desdf", default="datasets_s3d/desdf")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--fov-deg", type=float, default=80.0)
    parser.add_argument("--output-json")
    return parser.parse_args()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(function):
    synchronize()
    start = time.perf_counter()
    output = function()
    synchronize()
    return (time.perf_counter() - start) * 1000.0, output


def summarize(values):
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p90_ms": ordered[p90_index],
    }


def scene_for_index(dataset, index):
    starts = np.asarray(dataset.scene_start_idx)
    scene_index = int(np.sum(index >= starts) - 1)
    return dataset.scene_names[scene_index]


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    with open(os.path.join(args.dataset, "split.yaml"), encoding="utf-8") as file:
        split = AttrDict(yaml.safe_load(file))
    dataset = GridSeqDataset(
        args.dataset,
        split.test,
        L=3,
        depth_dir=args.dataset,
        depth_suffix="depth40",
        add_rp=False,
        net_type="rrp",
        all_imgs=True,
    )
    data = dataset[args.sample_index]
    scene = scene_for_index(dataset, args.sample_index)
    desdf_data = np.load(
        os.path.join(args.desdf, scene, "desdf.npy"),
        allow_pickle=True,
    ).item()
    desdf_np = desdf_data["desdf"].astype(np.float32, copy=True)
    desdf_np[desdf_np > 20.0] = 20.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observation = data["obs_tensor"].unsqueeze(0).to(device)
    desdf = torch.from_numpy(desdf_np).to(device)
    module = RRPLightningModule.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    model = module.model.to(device).eval()
    focal_width = 1.0 / (2.0 * np.tan(np.deg2rad(args.fov_deg) / 2.0))

    def predict_depth():
        features = model("encode", obs_img=observation)
        return model("decoder_inference", depth_cond=features)

    def infer():
        predicted_depth = predict_depth()
        depth_np = predicted_depth.squeeze(0).detach().cpu().numpy()
        rays = get_ray_from_depth(depth_np, V=9, F_W=focal_width)
        rays = torch.as_tensor(rays, device=device, dtype=desdf.dtype)
        _probability, _orientation, pose = localize_fast(
            desdf,
            rays,
            return_np=False,
        )
        return pose

    with torch.inference_mode():
        for _ in range(args.warmup):
            infer()
        synchronize()

        depth_times = []
        total_times = []
        for _ in range(args.repeats):
            elapsed, _ = measure(predict_depth)
            depth_times.append(elapsed)
            elapsed, _ = measure(infer)
            total_times.append(elapsed)

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "scene": scene,
        "sample_index": args.sample_index,
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "depth_network": summarize(depth_times),
        "end_to_end": summarize(total_times),
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

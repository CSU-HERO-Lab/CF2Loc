#!/usr/bin/env python3
import argparse
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark batch-one S3D stage-one inference latency."
    )
    parser.add_argument("--config", default="configs/PoseQueryDiffusion_S3D.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json")
    return parser.parse_args()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_once(function):
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


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if args.num_particles < 1 or args.num_steps < 1:
        raise ValueError("num-particles and num-steps must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    dataset_cfg = config["datasets"]
    dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=args.split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )
    obs_img, _pose, _ray, floorplan_img, wh, *_ = dataset[args.sample_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_img = obs_img.unsqueeze(0).to(device)
    floorplan_img = floorplan_img.unsqueeze(0).to(device)
    wh = wh.unsqueeze(0).to(device)

    model = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        args.ckpt,
        config=config,
        map_location="cpu",
    ).to(device)
    model.eval()

    def encode_condition():
        return model.condition_encoder(obs_img, floorplan_img)

    with torch.inference_mode():
        map_tokens, map_coordinates, image_global = encode_condition()

        def sample(cache_map_kv):
            samples = model.sample_from_context(
                map_tokens,
                map_coordinates,
                image_global,
                wh,
                num_particles=args.num_particles,
                num_steps=args.num_steps,
                cache_map_kv=cache_map_kv,
            )
            return model.select_pose_mode(samples)[0]

        def infer(cache_map_kv):
            context = encode_condition()
            samples = model.sample_from_context(
                *context,
                wh,
                num_particles=args.num_particles,
                num_steps=args.num_steps,
                cache_map_kv=cache_map_kv,
            )
            return model.select_pose_mode(samples)[0]

        torch.manual_seed(args.seed)
        uncached_output = sample(False)
        torch.manual_seed(args.seed)
        cached_output = sample(True)
        max_abs_difference = float(
            (uncached_output - cached_output).abs().max().item()
        )
        torch.testing.assert_close(
            cached_output,
            uncached_output,
            atol=0.0,
            rtol=0.0,
        )

        for _ in range(args.warmup):
            infer(False)
            infer(True)
        synchronize()

        condition_times = []
        sampling_times = {"uncached": [], "cached": []}
        end_to_end_times = {"uncached": [], "cached": []}
        for repeat in range(args.repeats):
            elapsed, _ = measure_once(encode_condition)
            condition_times.append(elapsed)
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for cache_map_kv in order:
                name = "cached" if cache_map_kv else "uncached"
                elapsed, _ = measure_once(lambda c=cache_map_kv: sample(c))
                sampling_times[name].append(elapsed)
                elapsed, _ = measure_once(lambda c=cache_map_kv: infer(c))
                end_to_end_times[name].append(elapsed)

    result = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.ckpt),
        "device": str(device),
        "batch_size": 1,
        "particles": args.num_particles,
        "sampling_steps": args.num_steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_abs_output_difference": max_abs_difference,
        "condition_encoder": summarize(condition_times),
        "sampling": {
            name: summarize(values) for name, values in sampling_times.items()
        },
        "end_to_end": {
            name: summarize(values) for name, values in end_to_end_times.items()
        },
    }
    for section in ("sampling", "end_to_end"):
        baseline = result[section]["uncached"]["median_ms"]
        cached = result[section]["cached"]["median_ms"]
        result[section]["speedup"] = baseline / cached
        result[section]["reduction_percent"] = (1.0 - cached / baseline) * 100.0

    result_text = json.dumps(result, indent=2)
    print(result_text)
    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            output_file.write(result_text + "\n")


if __name__ == "__main__":
    main()

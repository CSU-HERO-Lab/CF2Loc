import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_local_refiner import (
    PoseLocalRefinerLightning,
    apply_local_delta_to_pose,
    crop_to_refiner_tensor,
)
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer
from training.DisCo_lightning_module import DisCoLocModel


def csv_values(text, cast):
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerank S3D diffusion KDE modes with a trained DisCo model."
    )
    parser.add_argument("--config", default="configs/PoseQueryDiffusion_S3D.yaml")
    parser.add_argument("--diffusion-ckpt", required=True)
    parser.add_argument("--disco-ckpt", required=True)
    parser.add_argument("--depth-ckpt", required=True)
    parser.add_argument("--refiner-config")
    parser.add_argument("--refiner-ckpt")
    parser.add_argument("--data-root")
    parser.add_argument("--split-yaml")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--subset-fraction", type=float)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-particles", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--mode-counts", default="2,3,5,8")
    parser.add_argument("--nms-xy-m", default="0.5,0.75,1.0")
    parser.add_argument("--nms-theta-deg", default="20,30,45")
    parser.add_argument("--crop-sizes-m", default="3,5,7")
    parser.add_argument("--disco-weights", default="0.5,0.75,1.0")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    args.mode_counts = csv_values(args.mode_counts, int)
    args.nms_xy_m = csv_values(args.nms_xy_m, float)
    args.nms_theta_deg = csv_values(args.nms_theta_deg, float)
    args.crop_sizes_m = csv_values(args.crop_sizes_m, float)
    args.disco_weights = csv_values(args.disco_weights, float)
    if min(args.mode_counts) < 1:
        parser.error("--mode-counts must be positive")
    if min(args.nms_xy_m) <= 0 or min(args.nms_theta_deg) <= 0:
        parser.error("NMS radii must be positive")
    if min(args.crop_sizes_m) <= 0:
        parser.error("--crop-sizes-m must be positive")
    if min(args.disco_weights) < 0 or max(args.disco_weights) > 1:
        parser.error("--disco-weights must lie in [0, 1]")
    if args.subset_fraction is not None and args.max_samples is not None:
        parser.error("Use either --subset-fraction or --max-samples")
    if args.subset_fraction is not None and not 0 < args.subset_fraction <= 1:
        parser.error("--subset-fraction must lie in (0, 1]")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if bool(args.refiner_config) != bool(args.refiner_ckpt):
        parser.error("--refiner-config and --refiner-ckpt must be used together")
    parameter_count = (
        len(args.mode_counts)
        * len(args.nms_xy_m)
        * len(args.nms_theta_deg)
        * len(args.crop_sizes_m)
        * len(args.disco_weights)
    )
    if args.refiner_ckpt and parameter_count != 1:
        parser.error("Refiner evaluation requires exactly one reranking parameter set")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        return source_index, self.dataset[source_index]


class FloorplanCache:
    def __init__(self, max_items=64):
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, path, representation, dataset):
        key = (path, representation)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if representation == "semantic_onehot":
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            floorplan = dataset._build_semantic_onehot_labels(rgb)
        else:
            floorplan = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if floorplan is None:
                raise FileNotFoundError(f"Failed to read floorplan: {path}")
        self.cache[key] = floorplan
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return floorplan


class Metrics:
    def __init__(self):
        self.xy_errors = []
        self.theta_errors = []

    def update(self, pose, target, map_res):
        xy_error = torch.linalg.norm(pose[:2] - target[:2]) * map_res
        theta_error = torch.abs(
            torch.remainder(pose[2] - target[2] + math.pi, 2 * math.pi) - math.pi
        )
        self.xy_errors.append(float(xy_error.item()))
        self.theta_errors.append(float(theta_error.item()))

    def summarize(self):
        xy = torch.tensor(self.xy_errors)
        theta = torch.tensor(self.theta_errors)
        return {
            "samples": len(self.xy_errors),
            "0.1m_recall": float((xy <= 0.1).float().mean()),
            "0.5m_recall": float((xy <= 0.5).float().mean()),
            "1m_recall": float((xy <= 1.0).float().mean()),
            "1m_30deg_recall": float(
                ((xy <= 1.0) & (theta <= math.radians(30))).float().mean()
            ),
            "mean_xy_err_m": float(xy.mean()),
            "median_xy_err_m": float(xy.median()),
            "mean_theta_err_deg": float(torch.rad2deg(theta).mean()),
        }


def extract_kde_modes(
    pose_samples,
    density,
    max_modes,
    map_res,
    nms_xy_m,
    nms_theta_deg,
):
    order = torch.argsort(density, descending=True)
    selected = []
    theta_radius = math.radians(nms_theta_deg)
    for index in order.tolist():
        if not selected:
            selected.append(index)
        else:
            candidate = pose_samples[index]
            modes = pose_samples[selected]
            xy_distance = torch.linalg.norm(
                (modes[:, :2] - candidate[:2]) * map_res,
                dim=-1,
            )
            theta_distance = torch.abs(
                torch.remainder(modes[:, 2] - candidate[2] + math.pi, 2 * math.pi)
                - math.pi
            )
            normalized_distance = (
                (xy_distance / nms_xy_m).square()
                + (theta_distance / theta_radius).square()
            )
            if torch.all(normalized_distance > 1.0):
                selected.append(index)
        if len(selected) == max_modes:
            break
    return torch.tensor(selected, device=pose_samples.device, dtype=torch.long)


def normalize_scores(scores):
    if scores.numel() <= 1:
        return torch.zeros_like(scores)
    return (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)


def fused_candidate_scores(disco_scores, density, disco_weight):
    disco = normalize_scores(disco_scores)
    kde = normalize_scores(torch.log(density.clamp_min(1e-8)))
    return disco_weight * disco + (1.0 - disco_weight) * kde


def slice_image_features(image_features, index):
    if isinstance(image_features, tuple):
        return tuple(feature[index : index + 1] for feature in image_features)
    return image_features[index : index + 1]


def expand_image_features(image_features, index, count):
    sample_features = slice_image_features(image_features, index)
    if isinstance(sample_features, tuple):
        return tuple(
            feature.expand(count, *feature.shape[1:])
            for feature in sample_features
        )
    return sample_features.expand(count, *sample_features.shape[1:])


def concatenate_image_features(image_features):
    if isinstance(image_features[0], tuple):
        return tuple(
            torch.cat([features[item] for features in image_features], dim=0)
            for item in range(len(image_features[0]))
        )
    return torch.cat(image_features, dim=0)


def project_patch_tokens(image_encoder, patch_tokens, grid_h, grid_w):
    batch_size = patch_tokens.shape[0]
    features = patch_tokens.permute(0, 2, 1).reshape(
        batch_size,
        patch_tokens.shape[-1],
        grid_h,
        grid_w,
    )
    features = F.interpolate(
        features,
        size=image_encoder.target_size,
        mode="bilinear",
        align_corners=False,
    )
    features = image_encoder.proj(features)
    height, width = image_encoder.target_size
    tokens = features.flatten(2).transpose(1, 2)
    return tokens + image_encoder._build_2d_positional_encoding(
        height,
        width,
        batch_size,
        tokens.device,
        tokens.dtype,
    )


def encode_shared_observation(obs_img, diffusion, disco):
    diffusion_condition = diffusion.condition_encoder
    diffusion_encoder = diffusion_condition.image_encoder
    disco_encoder = disco.image_encoder
    if diffusion_encoder.use_cls_token or disco_encoder.use_cls_token:
        raise ValueError("Shared encoding currently requires patch-token-only encoders.")
    if (
        diffusion_encoder.patch_size != disco_encoder.patch_size
        or diffusion_encoder.intermediate_layer_idx
        != disco_encoder.intermediate_layer_idx
    ):
        raise ValueError("Diffusion and DisCo image backbones are incompatible.")

    batch_size, _channels, image_height, image_width = obs_img.shape
    patch_size = diffusion_encoder.patch_size
    padded_height = int(math.ceil(image_height / patch_size) * patch_size)
    padded_width = int(math.ceil(image_width / patch_size) * patch_size)
    padded = F.pad(
        obs_img,
        (0, padded_width - image_width, 0, padded_height - image_height),
    )
    patch_tokens = diffusion_encoder.backbone.get_intermediate_layers(
        padded,
        [diffusion_encoder.intermediate_layer_idx],
        return_class_token=False,
    )[0]
    grid_h = padded_height // patch_size
    grid_w = padded_width // patch_size

    diffusion_tokens = project_patch_tokens(
        diffusion_encoder,
        patch_tokens,
        grid_h,
        grid_w,
    )
    diffusion_tokens = diffusion_condition.image_mixer(
        diffusion_condition.image_norm(diffusion_tokens)
    )
    disco_tokens = project_patch_tokens(
        disco_encoder,
        patch_tokens,
        grid_h,
        grid_w,
    )
    disco_tokens = disco.image_self_attn(disco.image_token_norm(disco_tokens))
    if diffusion_tokens.shape[0] != batch_size:
        raise RuntimeError("Unexpected shared image-token batch size.")
    return diffusion_tokens, disco_tokens


def encode_diffusion_map_context(condition_encoder, floorplan_img, image_tokens):
    map_features = condition_encoder.map_projection(
        condition_encoder.map_encoder(floorplan_img)
    )
    _batch_size, _channels, height, width = map_features.shape
    map_tokens = map_features.flatten(2).transpose(1, 2)
    map_coordinates = condition_encoder.build_map_coordinates(
        height,
        width,
        map_tokens.device,
        map_tokens.dtype,
        condition_encoder.coordinate_convention,
    )
    map_tokens = map_tokens + condition_encoder.map_pos_mlp(
        map_coordinates
    ).unsqueeze(0)
    map_tokens = condition_encoder.map_norm(map_tokens)
    attended_map, _ = condition_encoder.map_image_attn(
        query=map_tokens,
        key=image_tokens,
        value=image_tokens,
        need_weights=False,
    )
    map_tokens = condition_encoder.map_image_norm(map_tokens + attended_map)
    map_tokens = condition_encoder.map_ffn_norm(
        map_tokens + condition_encoder.map_ffn(map_tokens)
    )
    return map_tokens, map_coordinates, image_tokens.mean(dim=1)


def build_candidate_maps(floorplan, poses, crop_size_m, map_res, representation):
    maps = []
    for pose in poses.detach().cpu().numpy():
        maps.append(
            crop_to_refiner_tensor(
                floorplan,
                pose,
                crop_size_meters=crop_size_m,
                map_res=map_res,
                output_size=128,
                representation=representation,
                oriented=True,
            )
        )
    return torch.stack(maps)


def build_refiner_maps(floorplans, poses, config):
    dataset_cfg = config["datasets"]
    representation = dataset_cfg.get(
        "refiner_floorplan_representation",
        dataset_cfg.get("floorplan_representation", "gray"),
    )
    maps = []
    for floorplan, pose in zip(floorplans, poses.detach().cpu().numpy()):
        maps.append(
            crop_to_refiner_tensor(
                floorplan,
                pose,
                crop_size_meters=float(
                    config.get("refiner_crop_size_meters", 5.0)
                ),
                map_res=float(dataset_cfg.get("map_res", 0.02)),
                output_size=int(config.get("refiner_crop_output_size", 256)),
                representation=representation,
                oriented=bool(dataset_cfg.get("refiner_oriented_crop", True)),
            ).float()
        )
    return torch.stack(maps)


def assert_refiner_image_compatibility(diffusion, refiner):
    module_pairs = (
        (
            diffusion.condition_encoder.image_encoder,
            refiner.refiner.image_encoder,
            "image_encoder",
        ),
        (
            diffusion.condition_encoder.image_norm,
            refiner.refiner.image_norm,
            "image_norm",
        ),
        (
            diffusion.condition_encoder.image_mixer,
            refiner.refiner.image_mixer,
            "image_mixer",
        ),
    )
    for diffusion_module, refiner_module, name in module_pairs:
        diffusion_state = diffusion_module.state_dict()
        refiner_state = refiner_module.state_dict()
        if diffusion_state.keys() != refiner_state.keys():
            raise ValueError(f"Refiner {name} structure differs from Stage-1.")
        for key in diffusion_state:
            if not torch.equal(diffusion_state[key].cpu(), refiner_state[key].cpu()):
                raise ValueError(
                    f"Refiner {name}.{key} differs from the Stage-1 checkpoint."
                )


def make_subset_indices(total, subset_fraction, max_samples, seed):
    if subset_fraction is not None:
        count = max(1, int(round(total * subset_fraction)))
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(total, size=count, replace=False)).tolist()
    if max_samples is not None:
        return list(range(min(total, max_samples)))
    return list(range(total))


def parameter_key(nms_xy_m, nms_theta_deg, mode_count, crop_size_m, disco_weight):
    return (
        f"nms{nms_xy_m:g}m_{nms_theta_deg:g}deg_"
        f"k{mode_count}_crop{crop_size_m:g}m_w{disco_weight:g}"
    )


def parameter_record(
    key,
    nms_xy_m,
    nms_theta_deg,
    mode_count,
    crop_size_m,
    disco_weight,
    metrics,
    oracle_metrics,
):
    return {
        "name": key,
        "parameters": {
            "nms_xy_m": nms_xy_m,
            "nms_theta_deg": nms_theta_deg,
            "mode_count": mode_count,
            "crop_size_m": crop_size_m,
            "disco_weight": disco_weight,
        },
        "metrics": metrics.summarize(),
        "candidate_oracle": oracle_metrics.summarize(),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["dptv2_ckpt_path"] = args.depth_ckpt
    dataset_cfg = config["datasets"]
    if args.data_root:
        dataset_cfg["data_folder"] = args.data_root
    if args.split_yaml:
        dataset_cfg["data_splits"] = args.split_yaml

    dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=args.split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )
    source_indices = make_subset_indices(
        len(dataset),
        args.subset_fraction,
        args.max_samples,
        args.subset_seed,
    )
    indexed_dataset = IndexedDataset(dataset, source_indices)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        args.diffusion_ckpt,
        config=config,
        map_location="cpu",
    ).to(device)
    diffusion.eval()

    disco_checkpoint = torch.load(
        args.disco_ckpt,
        map_location="cpu",
        weights_only=False,
    )
    disco_config = dict(disco_checkpoint["hyper_parameters"])
    disco_config["dptv2_ckpt_path"] = args.depth_ckpt
    disco = DisCoLocModel.load_from_checkpoint(
        args.disco_ckpt,
        config=disco_config,
        map_location="cpu",
    ).to(device)
    disco.eval()

    refiner = None
    refiner_config = None
    if args.refiner_ckpt:
        with open(args.refiner_config, "r", encoding="utf-8") as config_file:
            refiner_config = yaml.safe_load(config_file)
        refiner_config["baseline_checkpoint_path"] = args.diffusion_ckpt
        refiner_config["dptv2_ckpt_path"] = args.depth_ckpt
        refiner_config["datasets"]["map_res"] = dataset_cfg.get("map_res", 0.02)
        refiner = PoseLocalRefinerLightning.load_from_checkpoint(
            args.refiner_ckpt,
            config=refiner_config,
            map_location="cpu",
        ).to(device)
        refiner.eval()
        assert_refiner_image_compatibility(diffusion, refiner)

    num_particles = args.num_particles or int(
        config.get("diffusion_val_particles", 64)
    )
    num_steps = args.num_steps or int(config.get("diffusion_sample_steps", 20))
    max_modes = max(args.mode_counts)
    if max_modes > num_particles:
        raise ValueError("Maximum mode count cannot exceed the particle count.")

    parameter_metrics = {}
    oracle_metrics = {}
    for nms_xy_m in args.nms_xy_m:
        for nms_theta_deg in args.nms_theta_deg:
            for mode_count in args.mode_counts:
                oracle_key = (nms_xy_m, nms_theta_deg, mode_count)
                oracle_metrics[oracle_key] = Metrics()
                for crop_size_m in args.crop_sizes_m:
                    for disco_weight in args.disco_weights:
                        key = parameter_key(
                            nms_xy_m,
                            nms_theta_deg,
                            mode_count,
                            crop_size_m,
                            disco_weight,
                        )
                        parameter_metrics[key] = Metrics()

    baseline_metrics = Metrics()
    refined_metrics = Metrics() if refiner is not None else None
    map_cache = FloorplanCache()
    map_res = float(dataset_cfg.get("map_res", 0.02))
    with torch.inference_mode():
        for source_index, batch in tqdm(loader, desc=f"eval_{args.split}"):
            obs_img, target_pose, _ray, floorplan_img, wh, *_ = batch
            obs_img = obs_img.to(device, non_blocking=True)
            target_pose = target_pose.to(device, non_blocking=True)
            floorplan_img = floorplan_img.to(device, non_blocking=True)
            wh = wh.to(device, non_blocking=True)

            diffusion_image_tokens, image_features = encode_shared_observation(
                obs_img,
                diffusion,
                disco,
            )
            map_tokens, map_coordinates, image_global = encode_diffusion_map_context(
                diffusion.condition_encoder,
                floorplan_img,
                diffusion_image_tokens,
            )
            pose_samples = diffusion.sample_from_context(
                map_tokens,
                map_coordinates,
                image_global,
                wh,
                num_particles=num_particles,
                num_steps=num_steps,
            )
            baseline_pose, density = diffusion.select_pose_mode(pose_samples)

            sample_records = []
            for batch_index, dataset_index in enumerate(source_index.tolist()):
                sample_poses = pose_samples[batch_index]
                sample_density = density[batch_index]
                target = target_pose[batch_index]
                baseline_metrics.update(baseline_pose[batch_index], target, map_res)

                mode_sets = {}
                union_indices = set()
                for nms_xy_m in args.nms_xy_m:
                    for nms_theta_deg in args.nms_theta_deg:
                        indices = extract_kde_modes(
                            sample_poses,
                            sample_density,
                            max_modes=max_modes,
                            map_res=map_res,
                            nms_xy_m=nms_xy_m,
                            nms_theta_deg=nms_theta_deg,
                        )
                        mode_sets[(nms_xy_m, nms_theta_deg)] = indices
                        union_indices.update(indices.tolist())

                union_indices = sorted(union_indices)
                union_tensor = torch.tensor(
                    union_indices,
                    device=device,
                    dtype=torch.long,
                )
                index_to_union = {
                    particle_index: union_index
                    for union_index, particle_index in enumerate(union_indices)
                }
                union_poses = sample_poses[union_tensor]
                floorplan_path = dataset.data[dataset_index]["floorplan_image"]
                disco_representation = dataset.local_map_representation
                floorplan = map_cache.get(
                    floorplan_path,
                    disco_representation,
                    dataset,
                )
                sample_records.append(
                    {
                        "batch_index": batch_index,
                        "target": target,
                        "poses": sample_poses,
                        "density": sample_density,
                        "mode_sets": mode_sets,
                        "union_indices": union_indices,
                        "index_to_union": index_to_union,
                        "union_poses": union_poses,
                        "floorplan": floorplan,
                        "crop_scores": {},
                    }
                )

            for crop_size_m in args.crop_sizes_m:
                map_batches = []
                image_batches = []
                for record in sample_records:
                    candidate_maps = build_candidate_maps(
                        record["floorplan"],
                        record["union_poses"],
                        crop_size_m,
                        map_res,
                        disco_representation,
                    )
                    map_batches.append(candidate_maps)
                    image_batches.append(
                        expand_image_features(
                            image_features,
                            record["batch_index"],
                            candidate_maps.shape[0],
                        )
                    )
                candidate_maps = torch.cat(map_batches, dim=0).to(
                    device,
                    non_blocking=True,
                )
                candidate_images = concatenate_image_features(image_batches)
                batch_scores = disco.score_candidates(
                    candidate_images,
                    candidate_maps,
                )
                offset = 0
                for record, candidate_maps in zip(sample_records, map_batches):
                    count = candidate_maps.shape[0]
                    record["crop_scores"][crop_size_m] = batch_scores[
                        offset : offset + count
                    ]
                    offset += count

            for record in sample_records:
                sample_poses = record["poses"]
                sample_density = record["density"]
                target = record["target"]
                for (
                    nms_xy_m,
                    nms_theta_deg,
                ), all_mode_indices in record["mode_sets"].items():
                    for mode_count in args.mode_counts:
                        mode_indices = all_mode_indices[:mode_count]
                        candidate_poses = sample_poses[mode_indices]
                        xy_errors = torch.linalg.norm(
                            (candidate_poses[:, :2] - target[:2]) * map_res,
                            dim=-1,
                        )
                        theta_errors = torch.abs(
                            torch.remainder(
                                candidate_poses[:, 2] - target[2] + math.pi,
                                2 * math.pi,
                            )
                            - math.pi
                        )
                        oracle_index = torch.argmin(
                            xy_errors
                            + (theta_errors > math.radians(30)).float() * 1e-3
                        )
                        oracle_key = (nms_xy_m, nms_theta_deg, mode_count)
                        oracle_metrics[oracle_key].update(
                            candidate_poses[oracle_index],
                            target,
                            map_res,
                        )

                        union_positions = torch.tensor(
                            [
                                record["index_to_union"][index]
                                for index in mode_indices.tolist()
                            ],
                            device=device,
                            dtype=torch.long,
                        )
                        candidate_density = sample_density[mode_indices]
                        for crop_size_m in args.crop_sizes_m:
                            candidate_disco = record["crop_scores"][crop_size_m][
                                union_positions
                            ]
                            for disco_weight in args.disco_weights:
                                scores = fused_candidate_scores(
                                    candidate_disco,
                                    candidate_density,
                                    disco_weight,
                                )
                                selected_pose = candidate_poses[torch.argmax(scores)]
                                key = parameter_key(
                                    nms_xy_m,
                                    nms_theta_deg,
                                    mode_count,
                                    crop_size_m,
                                    disco_weight,
                                )
                                parameter_metrics[key].update(
                                    selected_pose,
                                    target,
                                    map_res,
                                )
                                if refiner is not None:
                                    record["selected_pose"] = selected_pose

            if refiner is not None:
                selected_poses = torch.stack(
                    [record["selected_pose"] for record in sample_records]
                )
                local_maps = build_refiner_maps(
                    [record["floorplan"] for record in sample_records],
                    selected_poses,
                    refiner_config,
                ).to(device, non_blocking=True)
                outputs = refiner(
                    obs_img=None,
                    local_map=local_maps,
                    candidate_pose=selected_poses,
                    wh=wh,
                    image_tokens=diffusion_image_tokens,
                )
                refined_poses = apply_local_delta_to_pose(
                    selected_poses,
                    outputs["delta_xy_m"],
                    outputs["delta_theta"],
                    map_res=map_res,
                    wh=wh,
                )
                for record, refined_pose in zip(sample_records, refined_poses):
                    refined_metrics.update(
                        refined_pose,
                        record["target"],
                        map_res,
                    )

    records = []
    for nms_xy_m in args.nms_xy_m:
        for nms_theta_deg in args.nms_theta_deg:
            for mode_count in args.mode_counts:
                oracle_key = (nms_xy_m, nms_theta_deg, mode_count)
                for crop_size_m in args.crop_sizes_m:
                    for disco_weight in args.disco_weights:
                        key = parameter_key(
                            nms_xy_m,
                            nms_theta_deg,
                            mode_count,
                            crop_size_m,
                            disco_weight,
                        )
                        records.append(
                            parameter_record(
                                key,
                                nms_xy_m,
                                nms_theta_deg,
                                mode_count,
                                crop_size_m,
                                disco_weight,
                                parameter_metrics[key],
                                oracle_metrics[oracle_key],
                            )
                        )
    records.sort(
        key=lambda record: (
            record["metrics"]["1m_recall"],
            record["metrics"]["1m_30deg_recall"],
            record["metrics"]["0.5m_recall"],
        ),
        reverse=True,
    )
    result = {
        "config": os.path.abspath(args.config),
        "diffusion_checkpoint": os.path.abspath(args.diffusion_ckpt),
        "disco_checkpoint": os.path.abspath(args.disco_ckpt),
        "split": args.split,
        "seed": args.seed,
        "num_particles": num_particles,
        "num_steps": num_steps,
        "total_split_samples": len(dataset),
        "evaluated_samples": len(source_indices),
        "subset_fraction": args.subset_fraction,
        "subset_seed": args.subset_seed if args.subset_fraction else None,
        "subset_indices_sha256": hashlib.sha256(
            np.asarray(source_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "baseline": baseline_metrics.summarize(),
        "best": records[0],
        "results": records,
    }
    if refined_metrics is not None:
        result["refiner_checkpoint"] = os.path.abspath(args.refiner_ckpt)
        result["refined"] = refined_metrics.summarize()
    output_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")
    summary = {"baseline": result["baseline"], "best": result["best"]}
    if refined_metrics is not None:
        summary["refined"] = result["refined"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

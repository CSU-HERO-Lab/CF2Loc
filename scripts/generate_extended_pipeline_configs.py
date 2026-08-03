#!/usr/bin/env python3
import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "configs" / "experiments" / "extended_pipeline_11m"
CHECKPOINT_ROOT = Path("checkpoints/extended_pipeline_11m")
PALMS_ROOT = Path("datasets_palms/full_s3d_like_oracle_noflip_rgborder")


def load_config(name):
    with (ROOT / "configs" / name).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_config(name, config):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return path


def set_run(config, run_name):
    config["run_name"] = run_name
    config["checkpoint_dir"] = str(CHECKPOINT_ROOT / run_name)
    config["use_wandb"] = True
    config["seed"] = 42
    return config


def make_disco(source_config, run_name):
    config = set_run(copy.deepcopy(source_config), run_name)
    config.update(
        {
            "epochs": 30,
            "batch_size": 64,
            "val_batch_size": 64,
            "precision": "32-true",
            "feature_dim": 128,
            "num_heads": 4,
            "image_self_attn_layers": 1,
            "pairwise_chunk_size": 16,
            "lr": 1.0e-4,
            "min_lr": 1.0e-5,
            "lr_scheduler": "cosine",
        }
    )
    dataset = config["datasets"]
    dataset["local_map_crop_size_meters"] = 5.0
    dataset["local_map_representation"] = (
        "semantic_onehot"
        if dataset.get("floorplan_representation") == "semantic_onehot"
        else "gray"
    )
    dataset["hard_negative"] = {"mode": "mixed"}
    dataset["pose_aug"] = {
        "enable": True,
        "trans_range": int(round(0.5 / float(dataset.get("map_res", 0.02)))),
        "rot_range": 0.26,
    }
    return config


def make_refiner(source_config, run_name):
    config = set_run(copy.deepcopy(source_config), run_name)
    config["refiner_crop_size_meters"] = 11.0
    config["epochs"] = 30
    return config


def palms_dataset(fold):
    return {
        "dataset_type": "s3d",
        "map_res": 0.1,
        "data_folder": str(PALMS_ROOT),
        "data_splits": str(PALMS_ROOT / f"split_5fold_seed0_fold{fold}_disco.yaml"),
        "val_split": "val",
        "floorplan_img_size": [256, 256],
        "floorplan_representation": "rgb",
        "local_map_representation": "gray",
        "refiner_floorplan_representation": "gray",
        "refiner_oriented_crop": True,
        "local_map_crop_size_meters": 5.0,
        "map_pose_rot_aug": {
            "enable": True,
            "p": 0.5,
            "angles": [90, 180, 270],
        },
        "hard_negative": {"mode": "none"},
        "pose_aug": {"enable": False},
    }


def main():
    s3d_stage1 = load_config("PoseQueryDiffusion_S3D.yaml")
    s3d_sem_stage1 = load_config("PoseQueryDiffusion_SemRayLoc_SemanticOneHot.yaml")
    zind_stage1 = load_config("PoseQueryDiffusion_ZInD_MainDiffusion.yaml")
    zind_sem_stage1 = load_config("PoseQueryDiffusion_ZInD_SemanticOneHot.yaml")
    s3d_refiner = load_config("PoseLocalRefiner_S3D_Dense.yaml")
    s3d_sem_refiner = load_config(
        "PoseLocalRefiner_SemRayLoc_SemanticOneHot_Dense.yaml"
    )
    zind_refiner = load_config("PoseLocalRefiner_ZInD_MainDiffusion.yaml")
    zind_sem_refiner = load_config(
        "PoseLocalRefiner_ZInD_SemanticOneHot_Dense.yaml"
    )

    configs = {
        "disco_s3d_semantic_5m_seed42": make_disco(
            s3d_sem_stage1, "disco_s3d_semantic_5m_seed42"
        ),
        "refiner_s3d_semantic_11m_seed42": make_refiner(
            s3d_sem_refiner, "refiner_s3d_semantic_11m_seed42"
        ),
        "disco_zind_no_sem_5m_seed42": make_disco(
            zind_stage1, "disco_zind_no_sem_5m_seed42"
        ),
        "refiner_zind_no_sem_11m_seed42": make_refiner(
            zind_refiner, "refiner_zind_no_sem_11m_seed42"
        ),
        "disco_zind_semantic_5m_seed42": make_disco(
            zind_sem_stage1, "disco_zind_semantic_5m_seed42"
        ),
        "refiner_zind_semantic_11m_seed42": make_refiner(
            zind_sem_refiner, "refiner_zind_semantic_11m_seed42"
        ),
    }

    for fold in range(1, 6):
        fold_name = f"palms_fold{fold}"
        stage1 = set_run(
            copy.deepcopy(s3d_stage1),
            f"stage1_{fold_name}_seed42",
        )
        stage1["datasets"] = palms_dataset(fold)
        stage1["diffusion_coordinate_convention"] = "metric_cell_center_v1"
        stage1["diffusion_metric_relative_max_m"] = 20.0
        stage1["diffusion_metric_relative_scale_m"] = 5.0
        configs[stage1["run_name"]] = stage1

        disco = make_disco(stage1, f"disco_{fold_name}_5m_seed42")
        disco["datasets"]["pose_aug"] = {
            "enable": True,
            "trans_range": 5,
            "rot_range": 0.26,
        }
        configs[disco["run_name"]] = disco

        refiner = make_refiner(s3d_refiner, f"refiner_{fold_name}_11m_seed42")
        refiner["datasets"] = palms_dataset(fold)
        refiner["baseline_checkpoint_path"] = str(
            CHECKPOINT_ROOT / stage1["run_name"] / "best.ckpt"
        )
        refiner["diffusion_coordinate_convention"] = "metric_cell_center_v1"
        refiner["diffusion_metric_relative_max_m"] = 20.0
        refiner["diffusion_metric_relative_scale_m"] = 5.0
        configs[refiner["run_name"]] = refiner

    paths = [write_config(name, config) for name, config in configs.items()]
    print(f"Wrote {len(paths)} configs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import gc
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DisCo_model.pose_local_refiner import PoseLocalRefinerLightning
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


RELEASE_CASES = (
    (
        "s3d_no_sem_stage1",
        PoseQueryDiffusionLocalizer,
        "PoseQueryDiffusion_S3D.yaml",
        "checkpoints/release_s3d/s3d_no_sem_stage1_best.ckpt",
    ),
    (
        "s3d_no_sem_refiner",
        PoseLocalRefinerLightning,
        "PoseLocalRefiner_S3D_Dense.yaml",
        "checkpoints/release_s3d/s3d_no_sem_dense_refiner_best.ckpt",
    ),
    (
        "s3d_semantic_stage1",
        PoseQueryDiffusionLocalizer,
        "PoseQueryDiffusion_SemRayLoc_SemanticOneHot.yaml",
        "checkpoints/release_s3d/s3d_semantic_onehot_stage1_best.ckpt",
    ),
    (
        "s3d_semantic_refiner",
        PoseLocalRefinerLightning,
        "PoseLocalRefiner_SemRayLoc_SemanticOneHot_Dense.yaml",
        "checkpoints/release_s3d/s3d_semantic_onehot_dense_refiner_best.ckpt",
    ),
    (
        "zind_no_sem_stage1",
        PoseQueryDiffusionLocalizer,
        "PoseQueryDiffusion_ZInD_MainDiffusion.yaml",
        "checkpoints/release_zind/zind_no_sem_stage1_best.ckpt",
    ),
    (
        "zind_no_sem_refiner",
        PoseLocalRefinerLightning,
        "PoseLocalRefiner_ZInD_MainDiffusion.yaml",
        "checkpoints/release_zind/zind_no_sem_dense_refiner_best.ckpt",
    ),
    (
        "zind_semantic_stage1",
        PoseQueryDiffusionLocalizer,
        "PoseQueryDiffusion_ZInD_SemanticOneHot.yaml",
        "checkpoints/release_zind/zind_semantic_onehot_stage1_best.ckpt",
    ),
    (
        "zind_semantic_refiner",
        PoseLocalRefinerLightning,
        "PoseLocalRefiner_ZInD_SemanticOneHot_Dense.yaml",
        "checkpoints/release_zind/zind_semantic_onehot_dense_refiner_best.ckpt",
    ),
)


def main() -> None:
    for name, model_class, config_name, checkpoint_name in RELEASE_CASES:
        config_path = ROOT / config_name
        checkpoint_path = ROOT / checkpoint_name
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing release checkpoint: {checkpoint_path}")
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            config=config,
            map_location="cpu",
        )
        model.eval()
        print(f"PASS {name}: {checkpoint_path.name}")
        del model
        gc.collect()


if __name__ == "__main__":
    main()

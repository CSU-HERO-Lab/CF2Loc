# CF2Loc

Official implementation of CF2Loc, a coarse-to-fine visual floorplan
localization framework. CF2Loc first samples globally plausible camera poses
with a full-map diffusion model, then applies a dense local refiner to the
KDE-selected coarse pose.

![CF2Loc framework](assets/framework.png)

## Installation

The release environment is tested with Python 3.8.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Depth Anything V2 ViT-S checkpoint to:

```text
checkpoints/depth_anything_v2_vits.pth
```

The checkpoint is available from the
[Depth Anything V2 repository](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth).

Dataset preparation and expected directory layouts are documented in
[`docs/DATASET_SWITCHING.md`](docs/DATASET_SWITCHING.md) and
[`docs/ZIND.md`](docs/ZIND.md).

## Configurations

| Dataset | Floorplan representation | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| Structured3D | grayscale | `configs/PoseQueryDiffusion_S3D.yaml` | `configs/PoseLocalRefiner_S3D_Dense.yaml` |
| Structured3D | semantic one-hot | `configs/PoseQueryDiffusion_SemRayLoc_SemanticOneHot.yaml` | `configs/PoseLocalRefiner_SemRayLoc_SemanticOneHot_Dense.yaml` |
| ZInD | grayscale | `configs/PoseQueryDiffusion_ZInD_MainDiffusion.yaml` | `configs/PoseLocalRefiner_ZInD_MainDiffusion.yaml` |
| ZInD | semantic one-hot | `configs/PoseQueryDiffusion_ZInD_SemanticOneHot.yaml` | `configs/PoseLocalRefiner_ZInD_SemanticOneHot_Dense.yaml` |

Non-semantic models use a single grayscale floorplan channel. Semantic models
use hard one-hot floorplan labels. The default local refiner uses an oriented
`5 m x 5 m` grayscale crop without additional edge channels.

## Training

Train the full-map diffusion model:

```bash
python training/train_pose_query_diffusion.py \
  --config configs/PoseQueryDiffusion_S3D.yaml
```

Train the local refiner with a frozen Stage-1 checkpoint:

```bash
python training/train_pose_local_refiner.py \
  --config configs/PoseLocalRefiner_S3D_Dense.yaml
```

Use the corresponding configuration from the table to switch datasets or
floorplan representations. Each refiner configuration specifies its frozen
Stage-1 checkpoint. Non-semantic Stage-1 models and local refiners train for
30 epochs. Semantic Stage-1 models train for 60 epochs.

## Evaluation

Run the complete Structured3D test split:

```bash
python eval/eval_pose_local_refiner.py \
  --config configs/PoseLocalRefiner_S3D_Dense.yaml \
  --refiner-ckpt checkpoints/release_s3d/s3d_no_sem_dense_refiner_best.ckpt \
  --split test
```

## Checkpoints

See [`checkpoints/README.md`](checkpoints/README.md) for checkpoint paths.

```bash
python scripts/check_release_checkpoints.py
```

## License

This project is released under the MIT License. Vendored Depth Anything V2 and
DINOv2 components retain their Apache-2.0 terms; the license text is included
under `LICENSES/`.

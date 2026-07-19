# DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization

<p align="center">
    <a href="https://arxiv.org/abs/2601.01822"><img src="https://img.shields.io/badge/arXiv-2601.01822-b31b1b.svg"></a>
    <a href="https://xiaowuguiovo.github.io/DisCo-FLoc_Project_Website/"><img src="https://img.shields.io/badge/Project-Website-blue.svg"></a>
    <a href="https://arxiv.org/pdf/2601.01822.pdf"><img src="https://img.shields.io/badge/Paper-PDF-green.svg"></a>
</p>

<p align="center">
    <strong>Ping Zhong, Shiyong Meng, Tao Zou, Bolei Chen*, Chaoxu Mu, Jianxin Wang</strong>
    <br>
</p>

<div align="center">
  <img src="assets/framework.png" width="100%">
</div>

This branch contains the current two-stage floorplan localization method: a
full-map pose diffusion model followed by a dense local pose refiner.

## Environment Setup

1.  **Prerequisites**: Ensure you have Python installed (recommended version >= 3.8).
2.  **Install Dependencies**: Run the following command to install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Pretrained Backbone

You need the **Depth Anything V2** checkpoint (ViT-S version).

* **Location**: `checkpoints/depth_anything_v2_vits.pth`
* **Download**: Download the `depth_anything_v2_vits.pth` from [**[HERE]**](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth).

## Canonical Configurations

| Dataset | Floorplan | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| S3D | non-semantic gray | `PoseQueryDiffusion_S3D.yaml` | `PoseLocalRefiner_S3D_Dense.yaml` |
| S3D | semantic one-hot | `PoseQueryDiffusion_SemRayLoc_SemanticOneHot.yaml` | `PoseLocalRefiner_SemRayLoc_SemanticOneHot_Dense.yaml` |
| ZInD | non-semantic gray | `PoseQueryDiffusion_ZInD_MainDiffusion.yaml` | `PoseLocalRefiner_ZInD_MainDiffusion.yaml` |
| ZInD | semantic one-hot | `PoseQueryDiffusion_ZInD_SemanticOneHot.yaml` | not reported |

S3D and ZInD non-semantic stage-1 models use one grayscale channel. Semantic
models use five hard one-hot channels. All stage-1 models use the standard
noise-prediction diffusion objective without a reconstructed-pose auxiliary
loss. Semantic stage-1 models train for 60 epochs; non-semantic stage-1 models
and all local refiners train for 30 epochs.

## Training

Train stage 1 and then pass its best checkpoint to the stage-2 configuration:

```bash
python training/train_pose_query_diffusion.py --config PoseQueryDiffusion_S3D.yaml
python training/train_pose_local_refiner.py \
  --config PoseLocalRefiner_S3D_Dense.yaml \
  --baseline_checkpoint_path /path/to/stage1.ckpt
```

Use the corresponding semantic or ZInD configuration from the table above to
switch datasets and floorplan representations.

## Evaluation

```bash
python eval/eval_pose_local_refiner.py \
  --config PoseLocalRefiner_S3D_Dense.yaml \
  --diffusion_ckpt checkpoints/release_s3d/s3d_no_sem_stage1_best.ckpt \
  --refiner_ckpt checkpoints/release_s3d/s3d_no_sem_dense_refiner_best.ckpt \
  --split test
```

Evaluation refines only the KDE-selected pose by default. Pass `--top_k 8`
explicitly to run the slower candidate-quality reranking ablation.

Release checkpoint manifests are stored under `checkpoints/release_s3d` and
`checkpoints/release_zind`. Verify all checkpoints against the current code with:

```bash
python scripts/check_release_checkpoints.py
```

Run unit tests with third-party pytest plugin auto-loading disabled when the
shell inherits a ROS environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

# CLEAR-FLoc Coding Agent Guide

This document is a compact project map for future coding agents. Read it before making changes.

## Project Goal

CLEAR-FLoc / DisCo-FLoc is an image-based floor plan localization project.

The S3D pipeline has two main stages:

1. RRP predicts depth/ray observations from a single RGB image.
2. DESDF matching proposes candidate poses on the floor plan, then DisCo reranks candidates by matching image tokens against local map patches.

The current stable S3D branch used by the user is usually `main_crosatt_cluster`.
The unified S3D + Gibson branch is `main_cross_att_cluster_s3d_and_gibson`.

## Top-Level Layout

- `RRP_model/`: RRP depth/ray prediction model and dataset/datamodule.
- `DisCo_model/`: DisCo image/map encoders, dataset, visualization utilities, and bundled Depth Anything V2 / DINOv2 code.
- `training/`: PyTorch Lightning modules and train entry points.
- `eval/`: S3D evaluation scripts and localization utilities.
- `eval/utils/`: DESDF localization, ray conversion, dataset wrappers, and DESDF generation helpers.
- `checkpoints/`: local, usually untracked checkpoints. Do not assume contents are tracked by git.
- `datasets_s3d/`: S3D data root. Expected to contain `Structured3D/` and `desdf/`.
- `datasets_gibson/`: Gibson data root or symlink. On the unified branch, Gibson training/eval code is available in the same branch.
- `logs/`: training/eval outputs. Many useful checkpoints live under `logs/*/checkpoints/`.
- `DisCo_FLoc.yaml`: S3D DisCo training config.
- `RRP.yaml`: S3D RRP training config.
- `DisCo_Gibson.yaml`: Gibson DisCo training config.
- `RRP_Gibson.yaml`: Gibson RRP training config.

## Data Layout

S3D data is expected under:

```text
datasets_s3d/Structured3D/
  split.yaml
  scene_XXXXX/
    imgs/
      000.png
      ...
    map.png
    poses_map.txt
    depth40.txt
```

DESDF data is expected under:

```text
datasets_s3d/desdf/
  scene_XXXXX/
    desdf.npy
```

`desdf.npy` is a Python dict saved with NumPy. The important keys are:

- `desdf`: `(H, W, O)` directional distance field in meters.
- `l`, `t`: crop offset used to convert map pixel coordinates into DESDF grid coordinates.
- sometimes `orn_slice`: number of orientation bins.

S3D map resolution is normally `0.02 m/pixel`. DESDF grid resolution is normally `0.1 m/cell`, so the stride from map pixels to DESDF cells is `5`.

Gibson data is expected under:

```text
datasets_gibson/gibson_f/
  split.yaml
  SCENE_NAME/
    rgb/
      00000-0.png
      ...
    map.png
    poses.txt
    depth40.txt

datasets_gibson/gibson_t/
  split.yaml
  SCENE_NAME/
    rgb/
      00000.png
      ...
    map.png
    poses.txt
    depth40.txt
```

Gibson poses are meter coordinates centered at the map center. The unified datasets convert them to map pixels with:

```text
x_map = x_m / 0.01 + map_w / 2
y_map = y_m / 0.01 + map_h / 2
```

Gibson map resolution is normally `0.01 m/pixel`. DESDF grid resolution is still `0.1 m/cell`, so the stride from map pixels to DESDF cells is `10`.

## Checkpoints

Common local checkpoint names:

```text
checkpoints/depth_anything_v2_vits.pth
checkpoints/RRP_s3d_best.ckpt
checkpoints/Disco_CrossAtt_s3d_best.ckpt
```

The code default in `eval/eval_disco_model_s3d.py` may still point to `checkpoints/DisCo_s3d_best.ckpt`. In this workspace, the user has often used `checkpoints/Disco_CrossAtt_s3d_best.ckpt`, so check `Get-ChildItem checkpoints` before giving commands.

`checkpoints/` is usually untracked. If files disappear, search under:

```text
logs/rrp_runs/*/checkpoints/
logs/disco_runs/*/checkpoints/
```

## RRP Pipeline

Main files:

- `training/train_rrp_model.py`: RRP training entry point.
- `training/RRP_lightning_module.py`: Lightning wrapper, train/val loss, optimizer.
- `RRP_model/depth_models.py`: model dispatcher that wires encoder and decoder.
- `RRP_model/RRP.py`: `RRPFeatureExtractor`, DINOv2 feature extraction and attention pooling.
- `RRP_model/models.py`: ray/depth decoder heads.
- `RRP_model/RRP_dataset.py`: S3D RGB image, pose, `depth40.txt`, floorplan loading.
- `RRP_model/RRP_lightning_datamodule.py`: train/val dataloaders.

Typical flow:

```text
RGB image
-> RRPFeatureExtractor
-> 40 token features
-> decoder
-> pred_depth40
-> get_ray_from_depth(...)
-> localize(desdf, rays)
-> prob_vol(H, W, O), prob_dist(H, W), orientations(H, W), pred[x, y, theta]
```

Important implementation details:

- `RRPFeatureExtractor` loads Depth Anything V2 / DINOv2 weights from `dptv2_ckpt_path`.
- Input images are normalized with ImageNet mean/std in the dataset.
- DINOv2 is usually frozen.
- Feature grid is interpolated to about `(23 or 30, 40)` depending on branch/config/code.
- The output supervision is based on `depth40.txt`, not raw depth maps.
- `get_ray_from_depth` converts 40 predicted depths into fewer ray lengths for DESDF matching.
- `localize` flips ray order before matching DESDF orientation bins.

Training command:

```powershell
python training\train_rrp_model.py --config RRP.yaml
```

Gibson RRP training uses a separate config on the unified S3D/Gibson branch:

```powershell
python training\train_rrp_model.py --config RRP_Gibson.yaml --exp_name rrp_gibson
```

This workspace also supports:

```powershell
python training\train_rrp_model.py --config RRP.yaml --exp_name residual_RRP
```

## DisCo Pipeline

Main files:

- `training/train_disco_model.py`: DisCo training entry point.
- `training/DisCo_lightning_module.py`: DisCo model, cross attention, contrastive loss, optimizer.
- `DisCo_model/image_patch_encoder.py`: image patch token encoder using DINOv2.
- `DisCo_model/map_encoder.py`: local map encoder.
- `DisCo_model/disco_dataset.py`: positive and hard negative local map generation.
- `DisCo_model/viz_utils.py`: cross-modal visualization helpers.

Current CrossAtt design:

```text
obs_img
-> ImagePatchEncoder
-> image tokens, usually 6 x 40 = 240 tokens, dim 128
-> optional image self attention

local_map
-> MapEncoder
-> map feature grid
-> flatten to map tokens, dim 128
-> map positional MLP

image tokens query map tokens through cross attention
-> token-weighted pooling
-> scalar pair score
```

Training uses in-batch candidate scoring:

```text
B images
2B local maps = B positive maps + B hard negative maps
logits_matrix shape = (B, 2B)
target for row i = i
loss = cross_entropy(logits_matrix * exp(logit_scale), targets)
```

DisCo training command:

```powershell
python training\train_disco_model.py --config DisCo_FLoc.yaml
```

Gibson DisCo training:

```powershell
python training\train_disco_model.py --config DisCo_Gibson.yaml --run_name disco_gibson_crossatt
```

Useful overrides:

```powershell
python training\train_disco_model.py --config DisCo_FLoc.yaml --run_name my_run --epochs 30 --batch_size 64
```

## S3D Evaluation

Main S3D eval script:

```text
eval/eval_disco_model_s3d.py
```

Typical command on `main_crosatt_cluster`:

```powershell
python eval\eval_disco_model_s3d.py `
  --config DisCo_FLoc.yaml `
  --rrp_model_ckpt "checkpoints\RRP_s3d_best.ckpt" `
  --disco_model_ckpt "checkpoints\Disco_CrossAtt_s3d_best.ckpt" `
  --fov 80.0 `
  --cluster_source_top_k 1000 `
  --cluster_radius_m 0.6
```

Clustered rerank is supported by:

```powershell
--cluster_source_top_k 1000
--cluster_radius_m 0.6
```

On the unified branch these are the defaults in `eval/eval_disco_model_s3d.py`:

```text
--fov 80.0
--cluster_source_top_k 1000
--cluster_radius_m 0.6
```

For old non-cluster S3D comparisons, override:

```powershell
--cluster_source_top_k 0 --cluster_radius_m 0.0
```

Evaluation flow:

```text
1. Load RRP checkpoint.
2. Load DisCo checkpoint.
3. Load S3D test split via GridSeqDataset.
4. Load each scene's DESDF and map.png.
5. Run RRP to get pred_depth40.
6. Convert depth40 to rays with get_ray_from_depth.
7. Run DESDF localize to get a pose probability field.
8. Select candidate positions from RRP probability.
9. Optionally cluster top RRP candidates by radius.
10. Crop a local map at each candidate pose.
11. Score candidate maps with DisCo.
12. Fuse geometric and semantic scores.
```

Score fusion:

```text
final_score = rrp_prob * exp(alpha * disco_score)
```

If `--disco_only` is set:

```text
final_score = exp(alpha * disco_score)
```

RRP-only top-k evaluation:

```powershell
python eval\eval_rrp_topk_curve.py `
  --rrp_model_ckpt "checkpoints\RRP_s3d_best.ckpt" `
  --dataset_path ".\datasets_s3d\Structured3D" `
  --desdf_path ".\datasets_s3d\desdf" `
  --hit_radius_m 1.0
```

## Gibson Evaluation

Gibson single-frame RRP + optional DisCo rerank:

```powershell
python eval\eval_disco_model_gibson.py `
  --config DisCo_Gibson.yaml `
  --dataset_path ".\datasets_gibson\gibson_f" `
  --desdf_path ".\datasets_gibson\desdf" `
  --rrp_model_ckpt "checkpoints\RRP_gibson_best.ckpt" `
  --disco_model_ckpt "checkpoints\DisCo_gibson_best.ckpt" `
  --net_type rrp `
  --fov 106.2602 `
  --V 11
```

For Gibson RRP-only evaluation, pass an empty DisCo checkpoint:

```powershell
python eval\eval_disco_model_gibson.py `
  --config DisCo_Gibson.yaml `
  --dataset_path ".\datasets_gibson\gibson_f" `
  --desdf_path ".\datasets_gibson\desdf" `
  --rrp_model_ckpt "checkpoints\RRP_gibson_best.ckpt" `
  --disco_model_ckpt "" `
  --net_type rrp `
  --fov 106.2602 `
  --V 11
```

Gibson sequence filtering evaluation:

```powershell
python eval\eval_filtering_gibson.py `
  --config DisCo_Gibson.yaml `
  --dataset_path ".\datasets_gibson\gibson_t" `
  --desdf_path ".\datasets_gibson\desdf" `
  --rrp_model_ckpt "checkpoints\RRP_gibson_best.ckpt" `
  --net_type rrp `
  --fov 106.2602 `
  --V 11 `
  --traj_len 100 `
  --eval_last_n 10
```

Gibson defaults are chosen to preserve the old fixed camera factor:

```text
--fov 106.2602 -> F_W ~= 3 / 8
```

## Coordinate Conventions

Be careful with coordinate frames:

- S3D `poses_map.txt` stores pose in map pixel coordinates: `[x, y, theta]`.
- Gibson `poses.txt` stores pose in meters and is converted to map pixel coordinates by the dataset/eval code.
- Floorplan image indexing is row/col, but poses are usually handled as x/y.
- S3D `map.png` resolution is usually `0.02 m/pixel`.
- Gibson `map.png` resolution is usually `0.01 m/pixel`.
- DESDF resolution is usually `0.1 m/cell`.
- Convert map pose to DESDF pose with:

```text
x_desdf = (x_map - l) / 5
y_desdf = (y_map - t) / 5
```

For Gibson, use stride `10` instead of `5`:

```text
x_desdf = (x_map - l) / 10
y_desdf = (y_map - t) / 10
```

- `localize` returns prediction in DESDF coordinates.
- Orientation bins use `theta = orn_idx / orn_slice * 2*pi`.
- `get_ray_from_depth` uses camera geometry and returns ray lengths, not raw column depths.
- S3D eval default uses `--fov 80.0`.
- Gibson eval default uses `--fov 106.2602`.
- Local map crops rotate so the agent heading points up in the crop.

## DESDF Generation

Current checked-in helper:

```text
eval/utils/generate_desdf.py
```

There were historical 180-orientation DESDF generation experiments on branch `eval_oracle`, commit `e742ddb`, with files:

```text
eval/utils/generate_desdf_180.py
eval/utils/generate_desdf_180_fast.py
```

Despite some directory names such as `desdf_generated_numba`, a git history search did not find actual `import numba`, `@njit`, or `prange` code in this repository. The fast version was vectorized ray marching, not a strict numba implementation.

## Branches And Experiment Lines

Known branch meanings in this workspace:

- `main_crosatt_cluster`: stable S3D CrossAtt DisCo plus clustered rerank line.
- `main_cross_att_cluster_s3d_and_gibson`: unified branch that supports S3D and Gibson RRP/DisCo training and evaluation.
- `feat/disco-clustered-rerank`: earlier clustered rerank experiment.
- `feat/disco-map-8x8-wall-edge`: map tower experiment with higher-resolution map tokens / wall-edge channels.
- `feat/rrp-pose-ranking-loss`: RRP pose ranking loss experiment.
- `feat/rrp-angle-aware-ray-tokens`: RRP angle-aware decoder experiment.
- `feat/rrp-local-residual-decoder`: RRP local residual decoder experiment.
- `main_gibson`: Gibson branch with sequence/filtering evaluation code.
- `eval_oracle`: historical oracle/180-DESDF experiments.

Before editing, always run:

```powershell
git branch --show-current
git status --short
```

The worktree is often dirty with active experiments. Do not revert unrelated changes.

## Common Pitfalls

- `checkpoints/` and dataset directories are local artifacts and may not be tracked.
- Checkpoint filenames have varied between experiments. Inspect the folder before writing commands.
- `RRP.yaml` is often modified for current experiments; do not assume it represents the original baseline.
- Some source files contain mojibake comments. Avoid re-encoding broad files unless necessary.
- `eval/eval_disco_model_s3d.py --help` can fail outside the right environment because imports happen before argument parsing.
- The default DisCo checkpoint path in code may not match the actual best checkpoint name.
- On older code, Gibson evaluation code and S3D evaluation code lived on different branches. On `main_cross_att_cluster_s3d_and_gibson`, both are available in one branch.
- Do not compare frame-level filtering accuracy with f3loc-style trajectory-level success without checking the protocol.

## Minimal Sanity Checks

For syntax-only checks:

```powershell
python -m compileall training RRP_model DisCo_model eval
```

For a targeted edit, prefer compiling only touched files:

```powershell
python -m compileall path\to\file.py
```

For full evaluation, use the explicit checkpoint command in the S3D Evaluation section rather than relying on defaults.

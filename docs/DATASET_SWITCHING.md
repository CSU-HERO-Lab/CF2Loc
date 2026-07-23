# Dataset Switching

CF2Loc uses the same Stage-1 and Stage-2 model implementations for
Structured3D and ZInD. Dataset selection is controlled entirely by YAML
configuration.

| Dataset | Stage-1 config | Expected data root | Map resolution |
| --- | --- | --- | --- |
| Structured3D | `configs/PoseQueryDiffusion_S3D.yaml` | `datasets_s3d/Structured3D` | 0.02 m/pixel |
| ZInD | `configs/PoseQueryDiffusion_ZInD_MainDiffusion.yaml` | `datasets_zind/disco_floc` | 0.01 m/pixel |

The data directories are ignored by Git. They may be real directories or
symbolic links:

```bash
ln -s /path/to/Structured3D datasets_s3d/Structured3D
ln -s /path/to/converted-zind datasets_zind/disco_floc
```

Train either dataset with:

```bash
python training/train_pose_query_diffusion.py \
  --config configs/PoseQueryDiffusion_S3D.yaml

python training/train_pose_query_diffusion.py \
  --config configs/PoseQueryDiffusion_ZInD_MainDiffusion.yaml
```

Semantic configurations additionally expect SemRayLoc-compatible semantic
floorplans under `datasets_semrayloc/processed` for Structured3D or
`datasets_zind/processed` for ZInD.

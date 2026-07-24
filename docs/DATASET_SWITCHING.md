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

For the data used by the release checkpoints, the loaded frame counts are:

| Configuration | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Structured3D grayscale | 65,048 | 6,726 | 6,405 |
| Structured3D semantic | 64,951 | 6,726 | 6,405 |
| ZInD grayscale or semantic | 192,000 | 22,656 | 22,764 |

The original Structured3D split file lists some unavailable training scene
directories. The loader reports their count and examples in a runtime warning;
it does not silently discard missing scenes. Missing files or mismatched
RGB/pose/depth counts inside an available scene remain fatal errors.

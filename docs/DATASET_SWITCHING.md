# Dataset Switching

`main_diffusion` uses one pose-query diffusion architecture for both Structured3D and ZInD. Select the dataset by configuration; no model source change is required.

| Dataset | Config | Scene layout | Map resolution |
| --- | --- | --- | --- |
| Structured3D | `PoseQueryDiffusion_S3D.yaml` | `imgs/`, `poses_map.txt` | 0.02 m/pixel |
| ZInD | `PoseQueryDiffusion_ZInD_MainDiffusion.yaml` | `rgb/`, `poses_map.txt` | 0.01 m/pixel |

The local workspace needs the ignored S3D data link once:

```bash
ln -s /home/ros/data/DisCo-FLoc/datasets_s3d datasets_s3d
```

Train either dataset with:

```bash
.venv/bin/python training/train_pose_query_diffusion.py \
  --config PoseQueryDiffusion_S3D.yaml

.venv/bin/python training/train_pose_query_diffusion.py \
  --config PoseQueryDiffusion_ZInD_MainDiffusion.yaml
```

The ZInD converter is only needed before first use; its output is already available at `/home/ros/data/zind/disco_floc` on this machine.

# ZInD for DisCo-FLoc

This branch keeps the `e781938` S3D pose-query diffusion architecture unchanged
and converts SemRayLoc-compatible ZInD raw data into the localizer's existing
scene layout.

The expected raw data is the extracted archive layout:

```text
/home/ros/data/zind/
  raw_data/<home-id>/zind_data.json
  raw_data/<home-id>/panos/*.jpg
  processed/<scene>/floorplan_walls_only.png
  processed/<scene>/poses.txt
  processed/<scene>/metadata.json
  processed/split.yaml
```

The SemRayLoc-aligned converter reuses its 1 cm/pixel wall map, poses, and
home-disjoint split. It produces four 80-degree views per panorama, writes
their map-pixel poses to `poses_map.txt`, and raycasts matching 40-value depth
targets to `depth40.txt`.

Convert with SemRayLoc's existing split so homes cannot cross splits:

```bash
.venv/bin/python scripts/prepare_zind.py \
  --raw-root /home/ros/data/zind/raw_data \
  --output-root /home/ros/data/zind/disco_floc \
  --semrayloc-processed-root /home/ros/data/zind/processed \
  --skip-existing
```

Then train with:

```bash
.venv/bin/python training/train_pose_query_diffusion.py \
  --config PoseQueryDiffusion_ZInD_MainDiffusion.yaml
```

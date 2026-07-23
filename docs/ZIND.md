# ZInD Preparation

The ZInD converter consumes raw panoramas and SemRayLoc-compatible processed
floorplans:

```text
datasets_zind/
  raw_data/<home-id>/zind_data.json
  raw_data/<home-id>/panos/*.jpg
  processed/<scene>/floorplan_walls_only.png
  processed/<scene>/poses.txt
  processed/<scene>/metadata.json
  processed/split.yaml
```

It reuses the home-disjoint split, renders four 80-degree perspective views
per panorama, writes map-pixel poses, and creates matching 40-ray depth files.

```bash
python scripts/prepare_zind.py \
  --raw-root datasets_zind/raw_data \
  --output-root datasets_zind/disco_floc \
  --semrayloc-processed-root datasets_zind/processed \
  --skip-existing
```

Train the non-semantic model with:

```bash
python training/train_pose_query_diffusion.py \
  --config PoseQueryDiffusion_ZInD_MainDiffusion.yaml
```

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ros/meng/DisCo-FLoc-main-diffusion"
PYTHON="/home/ros/meng/DisCo-FLoc/.venv/bin/python"
CKPT_DIR="$ROOT/checkpoints/release_s3d"
OUTPUT_DIR="$ROOT/logs/paper_eval/s3d_test"
BLOCKING_SESSION="eval_zind_semantic_onehot_sota_full_test"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

while tmux has-session -t "$BLOCKING_SESSION" 2>/dev/null; do
  printf '[%s] Waiting for %s to finish.\n' "$(date --iso-8601=seconds)" "$BLOCKING_SESSION"
  sleep 60
done

run_eval() {
  local name="$1"
  local config="$2"
  local diffusion_ckpt="$3"
  local refiner_ckpt="$4"
  local output_json="$OUTPUT_DIR/${name}.json"
  local output_log="$OUTPUT_DIR/${name}.log"

  printf '[%s] Starting %s.\n' "$(date --iso-8601=seconds)" "$name"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" eval/eval_pose_local_refiner.py \
    --config "$config" \
    --diffusion_ckpt "$diffusion_ckpt" \
    --refiner_ckpt "$refiner_ckpt" \
    --split test \
    --seed 0 \
    --val_particles 64 \
    --sample_steps 20 \
    --top_k 1 \
    --pose_selection kde \
    --topk_aggregate none \
    --cache_refiner_image \
    --log_every 100 \
    --output_json "$output_json" \
    2>&1 | tee "$output_log"
  printf '[%s] Finished %s.\n' "$(date --iso-8601=seconds)" "$name"
}

# Each full evaluation reports the stage-1 and KDE Top-1 refined rows.
run_eval \
  "s3d_no_sem_full_test_seed0" \
  "PoseLocalRefiner_S3D_Dense.yaml" \
  "$CKPT_DIR/s3d_no_sem_stage1_best.ckpt" \
  "$CKPT_DIR/s3d_no_sem_dense_refiner_best.ckpt"

run_eval \
  "s3d_semantic_onehot_full_test_seed0" \
  "PoseLocalRefiner_SemRayLoc_SemanticOneHot_Dense.yaml" \
  "$CKPT_DIR/s3d_semantic_onehot_stage1_best.ckpt" \
  "$CKPT_DIR/s3d_semantic_onehot_dense_refiner_best.ckpt"

printf '[%s] All S3D test evaluations completed.\n' "$(date --iso-8601=seconds)"

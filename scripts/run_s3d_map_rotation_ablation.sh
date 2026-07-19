#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ros/meng/DisCo-FLoc-main-diffusion"
PYTHON="/home/ros/meng/DisCo-FLoc/.venv/bin/python"
BLOCKING_SESSION="final_seed42_rerun_queue"
CONFIG="PoseQueryDiffusion_S3D_NoMapRotAug.yaml"
RUN_NAME="pose_query_diffusion_s3d_no_sem_no_map_rot_aug_seed42"
RUN_DIR="$ROOT/logs/pose_diffusion_runs/$RUN_NAME"
EVAL_DIR="$ROOT/logs/paper_eval/s3d_no_map_rot_aug_test_seed0"

cd "$ROOT"
mkdir -p "$RUN_DIR" "$EVAL_DIR"

while tmux has-session -t "$BLOCKING_SESSION" 2>/dev/null; do
  printf '[%s] Waiting for %s to finish.\n' \
    "$(date --iso-8601=seconds)" "$BLOCKING_SESSION"
  sleep 60
done

resume_args=()
if [[ -f "$RUN_DIR/checkpoints/last.ckpt" ]]; then
  resume_args=(--ckpt_path "$RUN_DIR/checkpoints/last.ckpt")
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" training/train_pose_query_diffusion.py \
  --config "$CONFIG" \
  --run_name "$RUN_NAME" \
  "${resume_args[@]}" \
  2>&1 | tee -a "$RUN_DIR/train.log"

best_checkpoint="$($PYTHON scripts/select_best_lightning_checkpoint.py \
  "$RUN_DIR/checkpoints" --monitor val_1m_recall)"
printf '[%s] Best checkpoint: %s\n' \
  "$(date --iso-8601=seconds)" "$best_checkpoint"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" eval/eval_pose_query_diffusion.py \
  --config "$CONFIG" \
  --ckpt "$best_checkpoint" \
  --split test \
  --seed 0 \
  --output-json "$EVAL_DIR/results.json" \
  2>&1 | tee "$EVAL_DIR/eval.log"

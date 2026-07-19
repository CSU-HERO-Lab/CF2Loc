#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ros/meng/DisCo-FLoc-main-diffusion"
PYTHON="/home/ros/meng/DisCo-FLoc/.venv/bin/python"
GPU_ID="${GPU_ID:-0}"
BLOCKING_SESSION="pose_local_refiner_s3d_dense_baseline"
EVAL_ROOT="$ROOT/logs/paper_eval/final_seed42"

mkdir -p "$EVAL_ROOT"
cd "$ROOT"

while tmux has-session -t "$BLOCKING_SESSION" 2>/dev/null; do
  printf '[%s] Waiting for %s to finish.\n' \
    "$(date --iso-8601=seconds)" "$BLOCKING_SESSION"
  sleep 60
done

select_best() {
  "$PYTHON" scripts/select_best_lightning_checkpoint.py \
    "$1/checkpoints" --monitor "$2"
}

run_reached_target_epoch() {
  local config="$1"
  local checkpoint="$2"
  [[ -f "$checkpoint" ]] || return 1
  "$PYTHON" - "$config" "$checkpoint" <<'PY'
import sys

import torch
import yaml

config_path, checkpoint_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as config_file:
    target_epochs = int(yaml.safe_load(config_file)["epochs"])
checkpoint = torch.load(checkpoint_path, map_location="cpu")
completed_epochs = int(checkpoint.get("epoch", -1)) + 1
raise SystemExit(0 if completed_epochs >= target_epochs else 1)
PY
}

run_stage1() {
  local config="$1"
  local run_name="$2"
  local run_dir="$ROOT/logs/pose_diffusion_runs/$run_name"
  local marker="$run_dir/seed42.complete"
  local resume_args=()
  mkdir -p "$run_dir"
  if run_reached_target_epoch "$config" "$run_dir/checkpoints/last.ckpt"; then
    touch "$marker"
  else
    rm -f "$marker"
  fi
  if [[ ! -f "$marker" ]]; then
    if [[ -f "$run_dir/checkpoints/last.ckpt" ]]; then
      resume_args=(--ckpt_path "$run_dir/checkpoints/last.ckpt")
      printf '[%s] Resuming stage 1 %s.\n' "$(date --iso-8601=seconds)" "$run_name" >&2
    else
      printf '[%s] Starting stage 1 %s.\n' "$(date --iso-8601=seconds)" "$run_name" >&2
    fi
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" training/train_pose_query_diffusion.py \
      --config "$config" \
      --run_name "$run_name" \
      "${resume_args[@]}" \
      2>&1 | tee "$run_dir/train_seed42.log" >&2
    touch "$marker"
  fi
  select_best "$run_dir" "val_1m_recall"
}

run_refiner() {
  local config="$1"
  local run_name="$2"
  local stage1_checkpoint="$3"
  local run_dir="$ROOT/logs/pose_diffusion_runs/$run_name"
  local marker="$run_dir/seed42.complete"
  local resume_args=()
  mkdir -p "$run_dir"
  if run_reached_target_epoch "$config" "$run_dir/checkpoints/last.ckpt"; then
    touch "$marker"
  else
    rm -f "$marker"
  fi
  if [[ ! -f "$marker" ]]; then
    if [[ -f "$run_dir/checkpoints/last.ckpt" ]]; then
      resume_args=(--ckpt_path "$run_dir/checkpoints/last.ckpt")
      printf '[%s] Resuming refiner %s.\n' "$(date --iso-8601=seconds)" "$run_name" >&2
    else
      printf '[%s] Starting refiner %s.\n' "$(date --iso-8601=seconds)" "$run_name" >&2
    fi
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" training/train_pose_local_refiner.py \
      --config "$config" \
      --run_name "$run_name" \
      --baseline_checkpoint_path "$stage1_checkpoint" \
      "${resume_args[@]}" \
      2>&1 | tee "$run_dir/train_seed42.log" >&2
    touch "$marker"
  fi
  select_best "$run_dir" "val_refined_0.5m_recall"
}

run_full_test() {
  local name="$1"
  local config="$2"
  local stage1_checkpoint="$3"
  local refiner_checkpoint="$4"
  local output_dir="$EVAL_ROOT/$name"
  mkdir -p "$output_dir"
  if [[ -f "$output_dir/results.json" ]]; then
    printf '[%s] Skipping completed eval %s.\n' "$(date --iso-8601=seconds)" "$name"
    return
  fi
  printf '[%s] Starting full test %s.\n' "$(date --iso-8601=seconds)" "$name"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" eval/eval_pose_local_refiner.py \
    --config "$config" \
    --diffusion_ckpt "$stage1_checkpoint" \
    --refiner_ckpt "$refiner_checkpoint" \
    --split test \
    --seed 42 \
    --val_particles 64 \
    --sample_steps 20 \
    --top_k 1 \
    --pose_selection kde \
    --topk_aggregate none \
    --cache_refiner_image \
    --log_every 100 \
    --output_json "$output_dir/results.json" \
    2>&1 | tee "$output_dir/eval.log"
}

s3d_sem_stage1="$(run_stage1 \
  PoseQueryDiffusion_SemRayLoc_SemanticOneHot.yaml \
  pose_query_diffusion_s3d_semantic_onehot_seed42)"
s3d_sem_refiner="$(run_refiner \
  PoseLocalRefiner_SemRayLoc_SemanticOneHot_Dense.yaml \
  pose_local_refiner_s3d_semantic_onehot_dense_seed42 \
  "$s3d_sem_stage1")"
run_full_test \
  s3d_semantic_onehot \
  PoseLocalRefiner_SemRayLoc_SemanticOneHot_Dense.yaml \
  "$s3d_sem_stage1" \
  "$s3d_sem_refiner"

zind_gray_stage1="$(run_stage1 \
  PoseQueryDiffusion_ZInD_MainDiffusion.yaml \
  pose_query_diffusion_zind_gray_seed42)"
zind_gray_refiner="$(run_refiner \
  PoseLocalRefiner_ZInD_MainDiffusion.yaml \
  pose_local_refiner_zind_gray_dense_seed42 \
  "$zind_gray_stage1")"
run_full_test \
  zind_gray \
  PoseLocalRefiner_ZInD_MainDiffusion.yaml \
  "$zind_gray_stage1" \
  "$zind_gray_refiner"

zind_sem_stage1="$(run_stage1 \
  PoseQueryDiffusion_ZInD_SemanticOneHot.yaml \
  pose_query_diffusion_zind_semantic_onehot_seed42)"
zind_sem_refiner="$(run_refiner \
  PoseLocalRefiner_ZInD_SemanticOneHot_Dense.yaml \
  pose_local_refiner_zind_semantic_onehot_dense_seed42 \
  "$zind_sem_stage1")"
run_full_test \
  zind_semantic_onehot \
  PoseLocalRefiner_ZInD_SemanticOneHot_Dense.yaml \
  "$zind_sem_stage1" \
  "$zind_sem_refiner"

printf '[%s] All seed-42 reruns and full tests completed.\n' \
  "$(date --iso-8601=seconds)"

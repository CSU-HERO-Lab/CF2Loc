#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-1}"
python_bin="${CF2LOC_PYTHON:-python}"
output_dir="outputs/extended_pipeline_11m/full_test"
log_dir="logs/extended_pipeline_11m/full_test_gpu${gpu_id}"

mkdir -p "$output_dir" "$log_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"

wait_for_checkpoint() {
  local checkpoint="$1"
  while [[ ! -e "$checkpoint" ]]; do
    sleep 60
  done
}

run_eval() {
  local name="$1"
  local config="$2"
  local diffusion_checkpoint="$3"
  local disco_checkpoint="$4"
  local refiner_config="$5"
  local refiner_checkpoint="$6"
  local output="$output_dir/${name}.json"
  local log="$log_dir/${name}.log"

  if [[ -e "$output" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP $name" | tee -a "$log"
    return
  fi

  wait_for_checkpoint "$diffusion_checkpoint"
  wait_for_checkpoint "$disco_checkpoint"
  wait_for_checkpoint "$refiner_checkpoint"
  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$log"
  "$python_bin" -u eval/eval_pose_diffusion_disco_rerank.py \
    --config "$config" \
    --diffusion-ckpt "$diffusion_checkpoint" \
    --disco-ckpt "$disco_checkpoint" \
    --depth-ckpt checkpoints/depth_anything_v2_vits.pth \
    --refiner-config "$refiner_config" \
    --refiner-ckpt "$refiner_checkpoint" \
    --split test \
    --seed 0 \
    --batch-size 4 \
    --mode-counts 5 \
    --nms-xy-m 1.0 \
    --nms-theta-deg 30 \
    --crop-sizes-m 5 \
    --disco-weights 0.75 \
    --output-json "$output" \
    2>&1 | tee -a "$log"
  echo "[$(date --iso-8601=seconds)] DONE $name" | tee -a "$log"
}

run_eval \
  s3d_no_sem_diffusion_disco_refiner11m \
  configs/PoseQueryDiffusion_S3D.yaml \
  checkpoints/release_s3d/s3d_no_sem_stage1_best.ckpt \
  checkpoints/extended_pipeline_11m/disco_s3d_no_sem_5m_seed42/best.ckpt \
  configs/experiments/extended_pipeline_11m/refiner_s3d_no_sem_11m_seed42.yaml \
  checkpoints/extended_pipeline_11m/refiner_s3d_no_sem_11m_seed42/best.ckpt

run_eval \
  s3d_semantic_diffusion_disco_refiner11m \
  configs/PoseQueryDiffusion_SemRayLoc_SemanticOneHot.yaml \
  checkpoints/release_s3d/s3d_semantic_onehot_stage1_best.ckpt \
  checkpoints/extended_pipeline_11m/disco_s3d_semantic_5m_seed42/best.ckpt \
  configs/experiments/extended_pipeline_11m/refiner_s3d_semantic_11m_seed42.yaml \
  checkpoints/extended_pipeline_11m/refiner_s3d_semantic_11m_seed42/best.ckpt

run_eval \
  zind_no_sem_diffusion_disco_refiner11m \
  configs/PoseQueryDiffusion_ZInD_MainDiffusion.yaml \
  checkpoints/release_zind/zind_no_sem_stage1_best.ckpt \
  checkpoints/extended_pipeline_11m/disco_zind_no_sem_5m_seed42/best.ckpt \
  configs/experiments/extended_pipeline_11m/refiner_zind_no_sem_11m_seed42.yaml \
  checkpoints/extended_pipeline_11m/refiner_zind_no_sem_11m_seed42/best.ckpt

run_eval \
  zind_semantic_diffusion_disco_refiner11m \
  configs/PoseQueryDiffusion_ZInD_SemanticOneHot.yaml \
  checkpoints/release_zind/zind_semantic_onehot_stage1_best.ckpt \
  checkpoints/extended_pipeline_11m/disco_zind_semantic_5m_seed42/best.ckpt \
  configs/experiments/extended_pipeline_11m/refiner_zind_semantic_11m_seed42.yaml \
  checkpoints/extended_pipeline_11m/refiner_zind_semantic_11m_seed42/best.ckpt

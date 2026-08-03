#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID QUEUE_ID" >&2
  exit 2
fi

GPU_ID="$1"
QUEUE_ID="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CF2LOC_PYTHON:-python}"
CONFIG_DIR="configs/experiments/extended_pipeline_11m"
LOG_DIR="logs/extended_pipeline_11m/queue_gpu${GPU_ID}"

cd "$ROOT"
mkdir -p "$LOG_DIR" checkpoints/extended_pipeline_11m
export CUDA_VISIBLE_DEVICES="$GPU_ID"
if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

config_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
value = config
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

run_job() {
  local kind="$1"
  local name="$2"
  local config="$CONFIG_DIR/${name}.yaml"
  local checkpoint_dir
  checkpoint_dir="$(config_value "$config" checkpoint_dir)"
  local best="$checkpoint_dir/best.ckpt"
  local last="$checkpoint_dir/last.ckpt"
  local log="$LOG_DIR/${name}.log"

  if [[ -e "$best" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP completed $name" | tee -a "$log"
    return
  fi

  local entrypoint
  case "$kind" in
    stage1) entrypoint="training/train_pose_query_diffusion.py" ;;
    disco) entrypoint="training/train_disco_model.py" ;;
    refiner) entrypoint="training/train_pose_local_refiner.py" ;;
    *) echo "Unknown job kind: $kind" >&2; return 2 ;;
  esac

  local resume_args=()
  if [[ -e "$last" ]]; then
    resume_args=(--ckpt_path "$last")
  fi
  echo "[$(date --iso-8601=seconds)] START $name on physical GPU $GPU_ID" \
    | tee -a "$log"
  "$PYTHON" -u "$entrypoint" --config "$config" "${resume_args[@]}" \
    2>&1 | tee -a "$log"
  echo "[$(date --iso-8601=seconds)] DONE $name" | tee -a "$log"
}

run_palms_fold() {
  local fold="$1"
  run_job stage1 "stage1_palms_fold${fold}_seed42"
  run_job refiner "refiner_palms_fold${fold}_11m_seed42"
  run_job disco "disco_palms_fold${fold}_5m_seed42"
}

case "$QUEUE_ID" in
  0)
    run_job disco disco_s3d_semantic_5m_seed42
    run_palms_fold 1
    run_palms_fold 2
    ;;
  1)
    run_job refiner refiner_s3d_semantic_11m_seed42
    run_palms_fold 3
    run_palms_fold 4
    ;;
  2)
    run_job disco disco_zind_no_sem_5m_seed42
    run_job refiner refiner_zind_no_sem_11m_seed42
    run_job disco disco_zind_semantic_5m_seed42
    run_job refiner refiner_zind_semantic_11m_seed42
    run_palms_fold 5
    ;;
  *)
    echo "QUEUE_ID must be 0, 1, or 2" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_bigcornea_quick_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

# Duration-SmolVLA smoke finetune on bigcornea. GPU id is physical id before CUDA remapping.
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
PILOT_EPISODES="${PILOT_EPISODES:-[2,8,13,14,382]}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_duration_sidecar_pilot_ep2_8_13_14_382.parquet}"

PRETRAINED="${PRETRAINED:-/home/jongwoopark/lerobot/smolvla_libero}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"
RESUME="${RESUME:-false}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

build_run_id
RUN_NAME="${RUN_NAME:-smolvla_hiva_duration_token_bigcornea_smoke_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_NAME}}"
guard_train_output_dir "${OUTPUT_DIR}" "${RESUME}"
build_wandb_args

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-5}"
EVAL_FREQ="${EVAL_FREQ:-0}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "PILOT_EPISODES=${PILOT_EPISODES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "STEPS=${STEPS}"
echo "EVAL_FREQ=${EVAL_FREQ}"
echo "RESUME=${RESUME}"
print_wandb_config
echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "LEROBOT_TRAIN_BIN=${LEROBOT_TRAIN_BIN}"

"${ACCELERATE_BIN}" launch --num_processes=1 --mixed_precision=bf16 "${LEROBOT_TRAIN_BIN}" \
  --policy.type=smolvla \
  --policy.pretrained_path="${PRETRAINED}" \
  --policy.expert_width_multiplier=0.5 \
  --policy.num_vlm_layers=0 \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.use_duration_head=true \
  --policy.duration_train_reuse_prefix_cache=true \
  --policy.duration_loss_weight=0.1 \
  --policy.duration_sidecar_path="${SIDECAR}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq="${STEPS}" \
  --num_workers=0 \
  --policy.scheduler_warmup_steps=1 \
  --policy.scheduler_decay_steps="${STEPS}" \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.n_action_steps=8 \
  --dataset.repo_id="${DATA_REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.episodes="${PILOT_EPISODES}" \
  --rename_map='{"observation.images.agentview":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
  --env.type=libero \
  --env.control_mode=relative \
  --env.task="${TASKS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --resume="${RESUME}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq="${EVAL_FREQ}" \
  "${WANDB_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Example launch on bigbrain with explicit hyperparameters:
#   GPU_IDS=0,1,2,3 \
#   NUM_GPUS=4 \
#   BATCH_PER_GPU=192 \
#   S=2 \
#   WANDB_ENABLE=true \
#   WANDB_PROJECT=lerobot \
#   bash server_scripts/bigbrain/finetune_bigbrain_ckpt_20k_d.sh
#
# You can also override the schedule directly for a smoke test:
#   GPU_IDS=0 NUM_GPUS=1 NUM_PROCESSES=1 BATCH_PER_GPU=4 STEPS=1 SAVE_FREQ=1 \
#   bash server_scripts/bigbrain/finetune_bigbrain_ckpt_20k_d.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_bigbrain_20k_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

###############################################################################
# Automatic step and scheduler scaling logic
#
# Baseline training setting:
#   - 1 GPU
#   - batch size 64 per GPU
#   - 20,000 training steps
#
# This script receives:
#   - NUM_GPUS: number of GPUs used for training
#   - BATCH_PER_GPU: batch size on each GPU
#   - S: extra sample-interaction reduction factor
#
# The global batch size is:
#   GLOBAL_BATCH_SIZE = NUM_GPUS * BATCH_PER_GPU
#
# The baseline global batch size is:
#   BASE_GLOBAL_BATCH_SIZE = 1 * 64 = 64
#
# To keep the same number of sample interactions after changing the global
# batch size, the number of steps scales inversely with global batch size:
#   STEPS = BASE_STEPS * BASE_GLOBAL_BATCH_SIZE / GLOBAL_BATCH_SIZE
#
# To intentionally reduce training exposure because of overfitting, divide
# that step count by S:
#   STEPS = BASE_STEPS * BASE_GLOBAL_BATCH_SIZE / GLOBAL_BATCH_SIZE / S
#
# With the bigbrain defaults:
#   NUM_GPUS=4, BATCH_PER_GPU=192, GLOBAL_BATCH_SIZE=768
#   S=2 -> 834 steps
#
# Scheduler defaults:
#   - warmup steps = 3% of STEPS
#   - decay steps = STEPS
#   - final decay learning rate = 2.5e-6
#
# Offline finetuning does not need LIBERO/MuJoCo env construction. Keep
# EVAL_FREQ=0 unless you explicitly want in-training simulation evaluation.
###############################################################################

# Full duration-SmolVLA finetune on bigbrain. GPU ids are physical ids before CUDA remapping.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-192}"
S="${S:-2}"

BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
BASE_STEPS="${BASE_STEPS:-20000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
EVAL_FREQ="${EVAL_FREQ:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
SIDECAR="${SIDECAR:-/nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_all_episodes.parquet}"

PRETRAINED="${PRETRAINED:-/home/jongwoopark/lerobot/smolvla_libero}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

RUN_NAME="${RUN_NAME:-smolvla_hiva_duration_token_bigbrain_full_s${S}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_NAME}}"
build_wandb_args

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))

STEPS="${STEPS:-$(python - <<PY
import math

base_steps = int("${BASE_STEPS}")
base_global_batch = int("${BASE_GLOBAL_BATCH_SIZE}")
global_batch = int("${GLOBAL_BATCH_SIZE}")
s = float("${S}")

steps = math.ceil(base_steps * base_global_batch / global_batch / s)
print(max(1, steps))
PY
)}"

BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"

SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$(python - <<PY
import math

steps = int("${STEPS}")
warmup_ratio = float("${WARMUP_RATIO}")

warmup_steps = math.ceil(steps * warmup_ratio)
print(max(1, warmup_steps))
PY
)}"

SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"

SAVE_FREQ="${SAVE_FREQ:-$(python - <<PY
steps = int("${STEPS}")
print(max(1, (steps + 1) // 2))
PY
)}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE}"
echo "S=${S}"
echo "BASE_STEPS=${BASE_STEPS}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "STEPS=${STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "SCHEDULER_WARMUP_STEPS=${SCHEDULER_WARMUP_STEPS}"
echo "SCHEDULER_DECAY_STEPS=${SCHEDULER_DECAY_STEPS}"
echo "SCHEDULER_DECAY_LR=${SCHEDULER_DECAY_LR}"
echo "EVAL_FREQ=${EVAL_FREQ}"
print_wandb_config
echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "LEROBOT_TRAIN_BIN=${LEROBOT_TRAIN_BIN}"

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
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
  --save_freq="${SAVE_FREQ}" \
  --num_workers=0 \
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}" \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}" \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.n_action_steps=8 \
  --dataset.repo_id="${DATA_REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
  --rename_map='{"observation.images.agentview":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
  --env.type=libero \
  --env.control_mode=relative \
  --env.task="${TASKS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq="${EVAL_FREQ}" \
  "${WANDB_ARGS[@]}"

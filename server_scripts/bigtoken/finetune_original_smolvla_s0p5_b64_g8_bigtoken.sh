#!/usr/bin/env bash
set -euo pipefail

# Finetune the original SmolVLA policy from the official LIBERO checkpoint with
# S=0.5 on 8 GPUs, batch size 64/GPU. This mirrors the original SmolVLA
# bigcornea sweep settings while using explicit milestone checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/server_scripts/common_wandb.sh"
build_run_id

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-${RUN_ID}}"
RUN_NAME="${RUN_NAME:-smolvla_original_bigcornea_s0p5_b64_g8_${RUN_TIMESTAMP}}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

###############################################################################
# Runtime / scaling
###############################################################################

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
S="${S:-0.5}"

BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
BASE_STEPS="${BASE_STEPS:-20000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
EVAL_FREQ="${EVAL_FREQ:-0}"
RESUME="${RESUME:-false}"

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))

calc_steps() {
  python - <<PY
import math
base_steps = int("${BASE_STEPS}")
base_global_batch = int("${BASE_GLOBAL_BATCH_SIZE}")
global_batch = int("${GLOBAL_BATCH_SIZE}")
s = float("${S}")
print(max(1, math.ceil(base_steps * base_global_batch / global_batch / s)))
PY
}

STEPS="${STEPS_OVERRIDE:-$(calc_steps)}"
WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS_OVERRIDE:-$(python - <<PY
import math
print(max(1, math.ceil(int("${STEPS}") * float("${WARMUP_RATIO}"))))
PY
)}"
DECAY_STEPS="${SCHEDULER_DECAY_STEPS_OVERRIDE:-${STEPS}}"

# Requested fractions of total steps:
#   0.6x, 0.625x, 0.7x, 0.75x, 0.875x, 1.0x of 5000
SAVE_STEPS="${SAVE_STEPS:-[3000,3125,3500,3750,4375,5000]}"

###############################################################################
# Paths / data
###############################################################################

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
PRETRAINED="${PRETRAINED:-/home/jongwoopark/lerobot/smolvla_libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jongwoopark/hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

WANDB_ENABLE=true
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
WANDB_NOTES="${WANDB_NOTES:-original SmolVLA no-duration finetune; S=${S}; bs/gpu=${BATCH_PER_GPU}; gpus=${NUM_GPUS}; steps=${STEPS}; save_steps=${SAVE_STEPS}}"
build_wandb_args

check_inputs() {
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Missing DATA_ROOT directory: ${DATA_ROOT}" >&2
    exit 1
  fi
  if [[ ! -d "${PRETRAINED}" ]]; then
    echo "Missing PRETRAINED directory: ${PRETRAINED}" >&2
    exit 1
  fi
  if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Missing or non-executable ACCELERATE_BIN: ${ACCELERATE_BIN}" >&2
    exit 1
  fi
  if [[ ! -x "${LEROBOT_TRAIN_BIN}" ]]; then
    echo "Missing or non-executable LEROBOT_TRAIN_BIN: ${LEROBOT_TRAIN_BIN}" >&2
    exit 1
  fi
}

check_inputs
guard_train_output_dir "${OUTPUT_DIR}" "${RESUME}"

echo "Logging to ${LOG_FILE}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE}"
echo "S=${S}"
echo "BASE_STEPS=${BASE_STEPS}"
echo "STEPS=${STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "DECAY_STEPS=${DECAY_STEPS}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAINED=${PRETRAINED}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
print_wandb_config

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
  --policy.use_duration_head=false \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq=0 \
  --save_steps="${SAVE_STEPS}" \
  --num_workers=0 \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${DECAY_STEPS}" \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}" \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.n_action_steps=1 \
  --dataset.repo_id="${DATA_REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
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

echo "===== Finished ${RUN_NAME} at $(date) ====="

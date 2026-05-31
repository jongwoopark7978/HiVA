#!/usr/bin/env bash
set -euo pipefail

# Sequential full-GPU finetune sweep for the original SmolVLA policy on
# bigcornea. Despite the historical filename, this script now runs training,
# not evaluation.
#
# Runs S=8,4,2,1 sequentially on GPU 0-7. S controls the same sample-exposure
# reduction used by finetune_bigcornea_ckpt_20k_d.sh:
#   steps = ceil(BASE_STEPS * BASE_GLOBAL_BATCH_SIZE / GLOBAL_BATCH_SIZE / S)
#
# Example:
#   nohup setsid bash server_scripts/bigcornea/eval_dist_bigcornea.sh \
#     > outputs/train_logs/original_smolvla_sweep_bigcornea_$(date +%Y%m%d_%H%M%S).outer.log 2>&1 < /dev/null &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/server_scripts/common_wandb.sh"
build_run_id

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
SWEEP_TIMESTAMP="${SWEEP_TIMESTAMP:-${RUN_ID}}"
LOG_FILE="${LOG_DIR}/original_smolvla_sweep_bigcornea_${SWEEP_TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

###############################################################################
# Cluster/runtime defaults
###############################################################################

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-48}"
S_VALUES="${S_VALUES:-8 4 2 1}"

BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
BASE_STEPS="${BASE_STEPS:-20000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
EVAL_FREQ="${EVAL_FREQ:-0}"
RESUME="${RESUME:-false}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
PRETRAINED="${PRETRAINED:-/home/jongwoopark/lerobot/smolvla_libero}"

source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache

RUN_PREFIX="${RUN_PREFIX:-smolvla_original_bigcornea}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train}"

# Always enable WandB for this sweep. WANDB_MODE remains configurable so you can
# choose online/offline, but the training process will always initialize W&B.
WANDB_ENABLE=true
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"

calc_steps() {
  local s="$1"
  python - <<PY
import math

base_steps = int("${BASE_STEPS}")
base_global_batch = int("${BASE_GLOBAL_BATCH_SIZE}")
global_batch = int("${GLOBAL_BATCH_SIZE}")
s = float("${s}")

steps = math.ceil(base_steps * base_global_batch / global_batch / s)
print(max(1, steps))
PY
}

calc_warmup_steps() {
  local steps="$1"
  python - <<PY
import math

steps = int("${steps}")
warmup_ratio = float("${WARMUP_RATIO}")

warmup_steps = math.ceil(steps * warmup_ratio)
print(max(1, warmup_steps))
PY
}

calc_save_freq() {
  local steps="$1"
  python - <<PY
steps = int("${steps}")
print(max(1, (steps + 1) // 2))
PY
}

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

print_sweep_config() {
  echo "Host: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
  echo "BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE}"
  echo "S_VALUES=${S_VALUES}"
  echo "BASE_STEPS=${BASE_STEPS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "WARMUP_RATIO=${WARMUP_RATIO}"
  echo "SCHEDULER_DECAY_LR=${SCHEDULER_DECAY_LR}"
  echo "EVAL_FREQ=${EVAL_FREQ}"
  echo "RESUME=${RESUME}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "DATA_REPO_ID=${DATA_REPO_ID}"
  echo "TASKS=${TASKS}"
  echo "PRETRAINED=${PRETRAINED}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "RUN_PREFIX=${RUN_PREFIX}"
  echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
  echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
  echo "LEROBOT_TRAIN_BIN=${LEROBOT_TRAIN_BIN}"
  print_wandb_config
}

run_one_s() {
  local s="$1"
  local steps="${STEPS_OVERRIDE:-$(calc_steps "${s}")}"
  local warmup_steps="${SCHEDULER_WARMUP_STEPS_OVERRIDE:-$(calc_warmup_steps "${steps}")}"
  local decay_steps="${SCHEDULER_DECAY_STEPS_OVERRIDE:-${steps}}"
  local save_freq="${SAVE_FREQ_OVERRIDE:-$(calc_save_freq "${steps}")}"
  local run_name="${RUN_PREFIX}_s${s}_${SWEEP_TIMESTAMP}"
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  guard_train_output_dir "${output_dir}" "${RESUME}"

  WANDB_NOTES="${WANDB_NOTES_BASE:-original SmolVLA no-duration sequential S sweep on bigcornea; S=${s}; steps=${steps}; global_batch=${GLOBAL_BATCH_SIZE}}"
  build_wandb_args

  echo
  echo "===== Starting ${run_name} at $(date) ====="
  echo "S=${s}"
  echo "STEPS=${steps}"
  echo "SAVE_FREQ=${save_freq}"
  echo "SCHEDULER_WARMUP_STEPS=${warmup_steps}"
  echo "SCHEDULER_DECAY_STEPS=${decay_steps}"
  echo "OUTPUT_DIR=${output_dir}"
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
    --steps="${steps}" \
    --log_freq=1 \
    --save_checkpoint=true \
    --save_freq="${save_freq}" \
    --num_workers=0 \
    --policy.scheduler_warmup_steps="${warmup_steps}" \
    --policy.scheduler_decay_steps="${decay_steps}" \
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
    --output_dir="${output_dir}" \
    --job_name="${run_name}" \
    --resume="${RESUME}" \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --eval_freq="${EVAL_FREQ}" \
    "${WANDB_ARGS[@]}"

  echo "===== Finished ${run_name} at $(date) ====="
}

check_inputs
build_wandb_args
print_sweep_config

for s in ${S_VALUES}; do
  run_one_s "${s}"
done

echo
echo "All original SmolVLA bigcornea S sweep runs finished at $(date)."
